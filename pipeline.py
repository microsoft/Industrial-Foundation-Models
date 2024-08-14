import os
import torch
import traceback
import datetime
from itertools import product

from transformers import (
    set_seed,
    enable_full_determinism,
)

import torch.distributed as dist

from accelerate import init_empty_weights

from src.config import Config, parse_args_to_config
from src.data import DataManager, DataCollatorForTabLM
from src.tokenizer import AutoTokenizerForTabLM
from src.models import AutoTabCausalLM
from src.trainer import Trainer
from src.callbacks import MetricsCallback, TaskCoverageCallback, MemoryCallback, ModelCheckpoint, EarlyStopping, WandbCallback
from src.evaluator import Evaluator
from src.utils import save_success, dump_pickle, load_datasets_from_config

import warnings
warnings.filterwarnings('ignore')

def only_enable_default_print_for_master(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

class Pipeline:
    """The pipeline class for training and inference of tabular foundation models"""

    def __init__(self, config: Config):
        self.config = config
        
        if config.enable_fsdp:
            self.init_distributed_mode()
        
        self.rank = get_rank()
        self._setup_seed(config.seed)
        self.tokenizer = self._prepare_tokenizer(config.model_path)
        
        if self.config.tabfm_stage == 'generate_data':
            self._prepare_data()
        else:
            # Skip the following steps if we only want to generate data
            self._prepare_evaluator()
            self._prepare_data_collator()
            self._prepare_model()
            self._setup_trainer()
    
    def init_distributed_mode(self):
        backend = 'nccl'
        dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        world_size = 1
        
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            rank = int(os.environ['SLURM_PROCID'])
            gpu = rank % torch.cuda.device_count()
        else:
            print('Not using distributed mode')
            return
         
        torch.cuda.set_device(gpu)
        print('| Distributed init (rank {}): {}, gpu {}'.format(rank, dist_url, gpu), flush=True)
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(hours=6)
        )
        torch.distributed.barrier()

        only_enable_default_print_for_master(rank == 0)

        # Set environment variables for easy debugging
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    
    def _setup_seed(self, seed):
        config = self.config
        # Set the seed for reproducibility
        if config.full_deterministic:
            enable_full_determinism(seed=seed)
        else:
            set_seed(seed)
    
    @classmethod
    def _prepare_tokenizer(cls, path):
        tokenizer = AutoTokenizerForTabLM.from_pretrained(path)
        tokenizer.define_numerical_tokens()
        tokenizer.add_additional_special_tokens()
        return tokenizer
    
    def _prepare_data_collator(self):
        self.data_collator = DataCollatorForTabLM(
            self.tokenizer._tokenizer,  # the wrapper class does not include the padding functionality from PreTrainedTokenizer
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        )
    
    def _prepare_model(self):
        config = self.config
        if self.rank == 0:
            model = AutoTabCausalLM(config)
        else:
            with init_empty_weights():
                model = AutoTabCausalLM(config)
        self.model = model
    
    def _init_callbacks(self):
        self.metrics_callback = MetricsCallback()
        self.task_coverage_callback = TaskCoverageCallback()
        self.memory_callback = MemoryCallback()
        self.model_checkpoint = ModelCheckpoint()
        self.earlystopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            metric=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better
        )
        self.wandb_callback = WandbCallback(
            project_name=self.config.wandb_project,
            run_name=self.config.wandb_run_name,
            config=self.config.to_dict()
        )
        self.callbacks = [
            self.metrics_callback,
            self.task_coverage_callback,
            self.earlystopping,
            self.model_checkpoint,
            self.memory_callback,
            self.wandb_callback,
        ]
    
    def _setup_trainer(self):
        self._init_callbacks()
        self.trainer = Trainer(
            config=self.config,
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=self.callbacks,
            evaluator=self.evaluator
        )
    
    def _prepare_data(self):
        self.data_manager = DataManager(self.config, self.tokenizer)
        self.data, self.task_info = self.data_manager.load_data()
    
    def _prepare_evaluator(self):
        self.evaluator = Evaluator(self.config, self.tokenizer)
    
    def _build_dataloader(self):
        self._prepare_data()
        self.trainer._prepare_dataloader(self.data, self.task_info)
    
    def pretrain(self):
        self._build_dataloader()
        self.trainer.train()
        if self.rank == 0:
            save_success(self.config.output_path, filename="pretrain_completed")
    
    def evaluation(self, run_tag):
        try:
            # Rebuild data based on the new data config
            self._build_dataloader()
            
            test_results_path = os.path.join(self.config.output_path, f"test_info.pkl.{run_tag}")
            if os.path.exists(test_results_path):
                print(f'The experiment in {test_results_path} already completes, skip')
                return
            
            test_results = self.trainer.test()
            if self.rank == 0 and test_results is not None:
                dump_pickle(test_results, test_results_path)
        except Exception as err:
            tb = traceback.format_exc()
            print(f"Fail to eval for setting {run_tag}, err={err}\n{tb}")
    
    def holdout_evaluation(self):
        """
            This function can support zero-shot, and in-context evaluation simultaneously.
        """
        config = self.config
        
        # Reload the pretrained model if needed
        if self.config.reload_ckpt_folder is not None:
            self.trainer._load_model_from(self.config.reload_ckpt_folder)
        else:
            self.trainer.logging("No pretrained model is reloaded.")
        
        if self.config.reload_directly:
            database_name = "-".join([d.split('/')[-1] for d in self.config.test_database.split(',')])
            run_tag = f"database_{database_name}"
            self.evaluation(run_tag)
        else:
            # Reset database and load evaluation datasets independently
            self.config.database = ""
            if config.eval_database != "":
                eval_datasets = load_datasets_from_config(config.eval_database)
            else:
                eval_datasets = [s for s in config.eval_datasets.split(',')]
            eval_seeds = [int(s) for s in config.eval_seeds.split(',')]
            eval_contexts = [int(s) for s in config.eval_contexts.split(',')]
            eval_context_seeds = [int(s) for s in config.eval_context_seeds.split(',')]
            
            for new_seed, new_context, new_context_seed, new_dataset in \
                product(eval_seeds, eval_contexts, eval_context_seeds, eval_datasets):
                # Reset config for each setting
                self.config.seed = int(new_seed)
                self.config.datasets = new_dataset
                self.config.in_context_count = new_context
                self.config.context_sample_seed = new_context_seed
                self.config.generate_train_seed_by_task_settings()
                self.trainer.config = self.config
                run_tag = f"dataset_{new_dataset}-seed_{new_seed}-context_{new_context}-contextseed_{new_context_seed}"
                
                self.evaluation(run_tag)
        
        if self.rank == 0:        
            save_success(self.config.output_path, filename="eval_completed")
    
if __name__ == '__main__':
    config = parse_args_to_config()
    pipeline = Pipeline(config)

    if config.tabfm_stage == 'pretrain':
        pipeline.pretrain()
    elif config.tabfm_stage == 'eval':
        pipeline.holdout_evaluation()
    elif config.tabfm_stage == 'generate_data':
        print(f"Finished generating data")
    else:
        raise ValueError(f"Unknown tabfm_stage: {config.tabfm_stage}")