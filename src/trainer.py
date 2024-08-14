import os
import time
from tqdm import tqdm
from typing import List

import functools
import torch
import transformers
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from transformers.optimization import get_scheduler
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from src.models import AutoTabCausalLM
from src.config import Config
from src.callbacks import Callback
from src.evaluator import Evaluator
from src.sampler import DistributedOrderedSampler
from src.utils import (
    clip_grad_norm,
    distributed_concat_and_move_to_cpu,
    gather_tensor,
    batch_move_to_cpu,
    load_model_and_optimizer_sharded,
    dump_pickle
)

class Trainer:
    def __init__(
        self,
        config: Config = None,
        model: AutoTabCausalLM = None,
        tokenizer: transformers.tokenization_utils.PreTrainedTokenizer = None,
        data_collator: transformers.DataCollatorForSeq2Seq = None,
        callbacks: List[Callback] = None,
        evaluator: Evaluator = None
    ):
        """
        This class is implemented to finetune llama-2 from a pretrained weights. 
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.callbacks = callbacks
        self.evaluator = evaluator

        self._init_scaler()
        self._enable_fsdp()
        self._prepare_model()

        # Initialize internal states to allow skipping training and directly run evaluation
        self.global_step = 0
        self.train_data_idx = 0
        self.data_step = 0
        self.epoch = 1
        self.epoch_step = 0
        self.enable_early_stopping = False
        self._setup_callbacks()
    
    def logging(self, *args, **kwargs):
        if self.rank == 0:
            print(time.asctime(), *args, **kwargs)
    
    def _init_scaler(self):
        # Create a gradient scaler for fp16
        if self.config.use_fp16 and self.config.enable_fsdp:
            self.scaler = ShardedGradScaler()
        elif self.config.use_fp16 and not self.config.enable_fsdp:
            self.scaler = torch.cuda.amp.GradScaler() 
    
    def _enable_fsdp(self):
        if self.config.enable_fsdp:
            # Initializing process groups
            self.local_rank = int(os.environ.get("LOCAL_RANK") or 0)
            self.rank = int(os.environ.get("RANK") or 0)
            self.world_size = int(os.environ.get("WORLD_SIZE") or 1)
            self.device = torch.cuda.current_device()
            assert torch.distributed.is_initialized()

            # Enable debug info
            os.environ['NCCL_BLOCKING_WAIT'] = "1"
            os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.per_device_batch_size = self.config.batch_size // self.world_size
    
    def _prepare_dataloader(self, data, task_info):
        self.datasets = data
        self.task_info = task_info
        
        self.dataloaders = {'train':[], 'val': [], 'test':[]}
        batch_size_dict = {
           'train': self.config.micro_batch_size,
           'val': self.config.val_batch_size,
           'test': self.config.test_batch_size
        }
        
        for data_type, datasets in data.items():
            for i, dataset in enumerate(datasets):
                if data_type == 'train':
                    drop_last = True
                    sampler = DistributedSampler(
                        dataset,
                        rank=dist.get_rank(),
                        num_replicas=dist.get_world_size(),
                        shuffle=True,
                        drop_last=drop_last
                    )
                else:
                    drop_last = False
                    sampler = DistributedOrderedSampler(
                        dataset,
                        rank=dist.get_rank(),
                        num_replicas=dist.get_world_size(),
                        shuffle=False,
                        drop_last=drop_last
                    )
                
                if not self.config.enable_fsdp:
                    sampler = None

                # Assign batch size for different datasets
                batch_size_list = batch_size_dict[data_type].split(',')
                if len(batch_size_list) == 1:
                    batch_size = int(batch_size_list[0])
                else:
                    assert len(batch_size_list) == len(datasets)
                    batch_size = int(batch_size_list[i])
                
                self.dataloaders[data_type].append(
                    DataLoader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=1,
                        pin_memory=True,
                        sampler=sampler,
                        drop_last=drop_last,
                        collate_fn=self.data_collator
                    )
                )
    
    def _prepare_model(self):
        model = self.model
        config = self.config
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logging(f"INFO: Base Model {config.model_path} has {total_params / 1e6} Million params")

        # Initialize custom modules after loading to ensure that our module parameters are not set to int8
        if config.enable_fsdp:
            model.init_custom_modules(self.tokenizer, "cpu")
        else:
            model.init_custom_modules(self.tokenizer, self.device)

        # Ensure pure_bf16 and use_fp16 are not used at the same time
        assert not (self.config.pure_bf16 and self.config.use_fp16)
        if self.config.pure_bf16:
            self.logging("INFO: Converting model to bfloat16")
            model.to(torch.bfloat16)
        elif self.config.use_fp16:
            self.logging("INFO: Converting model to float16")
            model.to(torch.float16)
        
        if config.enable_fsdp:
            # If we need FSDP + LoRA, please refer to llama receipes
            # The following code only supports full-parameter FSDP

            layers_to_be_wrapped = {
                LlamaDecoderLayer,
                Phi3DecoderLayer,
                # include more decoder layers here to support other LLMs
            }

            custom_auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=layers_to_be_wrapped,
            )
    
            self.logging("INFO: Wrapping model with FSDP")
            model = FSDP(
                model,
                # this policy wraps desired modules with FSDP from bottom to top
                auto_wrap_policy=custom_auto_wrap_policy,
                # only specify the mixed precision when we want different parameter, buffer, and gradient reduction dtypes
                mixed_precision=None,
                # we prefer to shard all of parameters, gradients, and optimizer states to reduce the memory usage
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                # this init fn helps to move the model parameters from cpu to gpu
                param_init_fn=lambda x: x.to_empty(device=self.device, recurse=False),
                device_id=self.device,
                limit_all_gathers=True,
                sync_module_states=True,
            )

            if self.config.use_act_ckpt:
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    # Always use NO_REENTRANT, https://pytorch.org/docs/stable/checkpoint.html
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )

                act_ckpt_cnt = 0
                def do_act_ckpt(submodule):
                    nonlocal act_ckpt_cnt
                    if any([isinstance(submodule, layer) for layer in layers_to_be_wrapped]):
                        act_ckpt_cnt += 1
                        if act_ckpt_cnt % self.config.act_ckpt_interval == 0:
                            return True
                    return False

                self.logging("INFO: Wrapping model with activation checkpointing")
                apply_activation_checkpointing(
                    model,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=do_act_ckpt,
                )
            
            # check whether the model is wrapped as you have expected
            self.logging("INFO: Wrapped Model:\n", model)
        elif not config.load_in_kbit and not config.enable_fsdp:
            model.to(self.device)
        
        self.model = model
    
    def _load_model_from(self, path):
        if not os.path.exists(path):
            self.logging(f"Warning: {path} does not exist, skip loading")
            return
        
        if self.config.enable_fsdp:
            if self.config.tabfm_stage == 'pretrain' and self.config.reload_optimizer:
                load_model_and_optimizer_sharded(self.model, self.rank, path, self.optimizer)
            else:
                load_model_and_optimizer_sharded(self.model, self.rank, path)
            self.logging(f"Succeeded in loading pretrained model from {path}")
        else:
            self.logging(f"Warning: no mechanism to store or load non-FSDP checkpoints")
    
    def recover_training_info_from_global_step(self):
        current_step = 0
        # Recover the dataset for continuous training
        for data_idx in range(len(self.dataloaders['train'])):
            per_epoch_steps = len(self.dataloaders['train'][data_idx])
            max_train_steps = self.config.num_epochs * per_epoch_steps
            max_train_steps_data = int(self.config.max_train_steps.split(',')[data_idx])
            if max_train_steps_data > 0:
                max_train_steps = min(max_train_steps, max_train_steps_data)
            
            if current_step + max_train_steps >= self.global_step:
                self.train_data_idx = data_idx
                self.data_step = self.global_step - current_step
                self.epoch = (self.data_step - 1) // per_epoch_steps + 1
                self.epoch_step = (self.data_step - 1) % per_epoch_steps + 1
                break
            current_step += max_train_steps
        self.logging(", ".join([
            f"INFO: Recovered training info from global step {self.global_step}",
            f"train_data_idx={self.train_data_idx}",
            f"epoch={self.epoch}",
            f"epoch_step={self.epoch_step}"
        ]))
    
    @classmethod
    def _get_step_sorted_ckpt_folders(cls, exp_path):
        ckpt_folders = [
            os.path.join(exp_path, fn) for fn in os.listdir(exp_path) if fn.startswith('checkpoint')
        ]
        ckpt_folders.sort(key=lambda x: int(x.split('step-')[-1]))
        return ckpt_folders
    
    def _load_model_from_last_ckpt(self, exp_path):
        ckpt_folder = None
        ckpt_folders = self._get_step_sorted_ckpt_folders(exp_path)
        if len(ckpt_folders) > 0:
            ckpt_folder = ckpt_folders[-1]
        
        if ckpt_folder is None:
            self.logging(f"INFO: there is no checkpoint to reload for this exp.")
        else:
            self._load_model_from(ckpt_folder)
            # Reload training information
            if self.config.tabfm_stage == 'pretrain':
                self.global_step = int(ckpt_folder.split('step-')[-1].split('_')[0])
                self.recover_training_info_from_global_step()
                self.task_coverage_callback.reload_coverage(self)
                self.earlystop_callback.reload_earlystop(self)
        return ckpt_folder

    def _prepare_optimizer(self):
        model = self.model

        # Initialize the optimizer and learning rate scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            # foreach=False     # We only disable foreach when optimizer.step() leads to OOM
        )

        if self.config.lr_schedule_type != 'none':
            self.warmup_steps = self.config.warmup_steps
            self.scheduler = get_scheduler(
                self.config.lr_schedule_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.config.warmup_steps
            )
        else:
            self.scheduler = None
    
    def train_step(self, batch, batch_idx):
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)
            if self.config.pure_bf16 and batch[key].dtype == torch.float:
                batch[key] = batch[key].to(torch.bfloat16)
            elif self.config.use_fp16 and batch[key].dtype == torch.float:
                batch[key] = batch[key].to(torch.float16)
        loss = self.model(**batch).loss
        return loss
    
    def train_epoch(self, epoch, data_idx):
        self.train_data_idx = data_idx
        train_dataset = self.datasets['train'][data_idx]
        train_dataloader = self.dataloaders['train'][data_idx]
        train_dataloader.sampler.set_epoch(epoch)
        
        max_train_steps = int(self.config.max_train_steps.split(',')[data_idx])
        micro_batch_size = train_dataloader.batch_size
        gradient_accumulation_steps = self.per_device_batch_size // micro_batch_size
        
        self.on_train_epoch_start(epoch)
        for step, batch in enumerate(tqdm(train_dataloader, colour="blue", desc=f"Training Dataset{data_idx} Epoch{epoch}")):
            # When reloading from a checkpoint, skip the steps that have been trained
            if step == self.epoch_step:
                self.on_train_step_start(step)
                loss = self.train_step(batch, step)
                loss = loss / gradient_accumulation_steps
                if self.config.use_fp16:
                    # If fp16 is enabled, use gradient scaler to handle gradient update
                    self.scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if self.config.use_clip_grad:
                            clip_grad_norm(self.model, self.optimizer, self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        if self.scheduler is not None:
                            self.scheduler.step()
                else:
                    # Regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if self.config.use_clip_grad:
                            clip_grad_norm(self.model, self.optimizer, self.config.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if self.scheduler is not None:
                            self.scheduler.step()
                self.train_step_loss = loss.detach().float()
                # Calculate loss sum of an epoch
                self.train_epoch_loss += self.train_step_loss * gradient_accumulation_steps * micro_batch_size
                # Store task indices for analysing data coverage
                self.train_step_task_indices = batch['task_idx'].cpu().numpy().tolist()
                self.on_train_step_end()

            # Skip validation for the previous training steps
            if step + 1 < self.epoch_step:
                continue
            
            # Run validation every n steps
            if (self.config.val_every_n_steps > 0) and (self.data_step % self.config.val_every_n_steps == 0):
                self.validation()
            # Check early stopping
            if self.earlystop_callback.should_stop:
                break
            if max_train_steps > 0 and self.data_step >= max_train_steps:
                break
        
        # If there's more than one CUDA device, reduce train loss across all devices
        if torch.cuda.device_count() > 1 and self.config.enable_fsdp:
            dist.all_reduce(self.train_epoch_loss, op=dist.ReduceOp.SUM)

        # Calculate average loss in an epoch over all samples
        self.train_epoch_loss = self.train_epoch_loss / len(train_dataset)
        self.on_train_epoch_end()
    
    def train(self, metric_save_suffix=''):
        self._prepare_optimizer()
        self._setup_callbacks()
        
        # Reload pretrained model from the last checkpoint 
        if self.config.reload_model_before_training:
            if self.config.reload_ckpt_folder is not None:
                self._load_model_from(self.config.reload_ckpt_folder)
            else:
                self._load_model_from_last_ckpt(self.config.output_path)
        
        # Validation before training
        if self.config.val_before_training:
            self.validation()

        # This flag will control whether to check early stopping
        self.enable_early_stopping = True
        for data_idx in range(self.train_data_idx, len(self.dataloaders['train'])):
            for epoch in range(self.epoch, self.config.num_epochs + 1):
                self.train_epoch(epoch, data_idx)
                if self.config.val_per_epoch:
                    self.validation()
                if self.earlystop_callback.should_stop:
                    break
                self.epoch_step = 0
            self.data_step = 0
        self.enable_early_stopping = False

        # Save metrics collected during training
        if self.rank == 0:
            save_path = os.path.join(self.config.output_path, f"train_info.pkl{metric_save_suffix}")
            dump_pickle(self.metrics_callback.metrics, path=save_path)
    
    def predict_step(self, batch, batch_idx, max_generate_length=10):
        # TODO: we may consider restrict input features here to accelerate inference
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)
            if self.config.pure_bf16 and batch[key].dtype == torch.float:
                batch[key] = batch[key].to(torch.bfloat16)
            elif self.config.use_fp16 and batch[key].dtype == torch.float:
                batch[key] = batch[key].to(torch.float16)
        # Ensure no gradients are computed for this scope to save memory
        if self.config.use_legacy_generate:
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = self.model(
                    is_generate=True,
                    max_generate_length=max_generate_length,
                    **batch
                )
                loss = outputs.loss
        else:
            # A weird bug when using pytorch FSDP with huggingface's generate method
            # See https://github.com/pytorch/pytorch/issues/82461
            # and https://github.com/pytorch/pytorch/issues/100069 for workaround methods
            # if batch_idx == 0:
            #     self.model(**batch)
            with FSDP.summon_full_params(self.model, writeback=False, recurse=False):
                outputs = self.model.generate(
                    return_dict=True,
                    pad_to_max_length=True,
                    max_new_tokens=max_generate_length,
                    synced_gpus=True,
                    **batch
                )
            loss = outputs.loss
        return outputs, loss
    
    def validation(self):
        if 'val' not in self.dataloaders or len(self.dataloaders['val']) == 0:
            return None
        
        self.all_val_outputs = {}
        for val_data_idx in range(len(self.dataloaders['val'])):
            self.val_data_idx = val_data_idx
            self.on_validation_epoch_start()
            if self.metrics_callback.should_skip_val:
                self.logging(f"INFO: Skip validation dataset {val_data_idx} on global step {self.global_step}")
                continue
            else:
                val_dataset = self.datasets['val'][val_data_idx]
                val_dataloader = self.dataloaders['val'][val_data_idx]
                max_generate_length = int(self.config.val_max_generate_length.split(',')[val_data_idx])
                # local_outputs, local_labels = None, None
                gathered_outputs, gathered_labels = [], []
                for step, batch in enumerate(tqdm(val_dataloader, colour="green", desc=f"Validation Dataset{val_data_idx}")):
                    if self.config.debug_mode and step >= self.config.debug_max_val_steps:
                        break
                    self.on_validation_step_start()
                    outputs, loss = self.predict_step(batch, step, max_generate_length)
                    if loss is not None:
                        self.val_epoch_loss += loss.detach().float() * batch['input_ids'].shape[0]
                    labels = batch['label_tabular']
                    if torch.cuda.device_count() > 1 and self.config.enable_fsdp:
                        outputs = distributed_concat_and_move_to_cpu(tensor=outputs, valid_keys=outputs.get_eval_output_keys())
                        labels = distributed_concat_and_move_to_cpu(labels)
                    else:
                        outputs = batch_move_to_cpu(outputs, valid_keys=outputs.get_eval_output_keys())
                        labels = batch_move_to_cpu(labels)
                    gathered_outputs.append(outputs)
                    gathered_labels.append(labels)
                    self.on_validation_step_end()

                # If there's more than one CUDA device, reduce validation loss across all devices
                if torch.cuda.device_count() > 1 and self.config.enable_fsdp:
                    dist.all_reduce(self.val_epoch_loss, op=dist.ReduceOp.SUM)

                # Calculate average loss in an epoch over all samples
                self.val_epoch_loss = self.val_epoch_loss / len(val_dataset)

                # Gather all outputs and labels for evaluation
                gathered_outputs = gather_tensor(gathered_outputs, limit=len(val_dataset))
                gathered_labels = gather_tensor(gathered_labels, limit=len(val_dataset))
                self.val_metric_info = self.evaluator(
                    outputs=gathered_outputs,
                    labels=gathered_labels, 
                    eval_task_info=self.task_info['val'][val_data_idx],
                    stage='Val'
                )
            self.all_val_outputs[val_data_idx] = {
                'outputs': gathered_outputs,
                'labels': gathered_labels,
                'metrics': self.val_metric_info
            }
            self.is_val_finished = True if val_data_idx == len(self.dataloaders['val']) - 1 else False
            self.on_validation_epoch_end()
        return self.all_val_outputs
    
    def test(self):
        if 'test' not in self.dataloaders or len(self.dataloaders['test']) == 0:
            return None
        
        self.all_test_metric_info = {}
        self.all_test_outputs = {}
        for test_data_idx in range(len(self.dataloaders['test'])):
            self.test_data_idx = test_data_idx
            self.on_test_epoch_start()
            if self.metrics_callback.should_skip_test:
                self.logging(f"INFO: Skip test dataset {test_data_idx} on global step {self.global_step}")
                continue
            else:
                self.test_dataset = self.datasets['test'][test_data_idx]
                self.test_dataloader = self.dataloaders['test'][test_data_idx]
                max_generate_length = int(self.config.test_max_generate_length.split(',')[test_data_idx])

                gathered_outputs, gathered_labels = [], []
                for step, batch in enumerate(tqdm(self.test_dataloader,colour="green", desc=f"Test Dataset{test_data_idx}")):
                    if self.config.debug_mode and step >= self.config.debug_max_test_steps:
                        break
                    self.on_test_step_start()
                    outputs, loss = self.predict_step(batch, step, max_generate_length)
                    if loss is not None:
                        self.test_epoch_loss += loss.detach().float() * batch['input_ids'].shape[0]
                    labels = batch['label_tabular']
                    if torch.cuda.device_count() > 1 and self.config.enable_fsdp:
                        outputs = distributed_concat_and_move_to_cpu(tensor=outputs, valid_keys=outputs.get_eval_output_keys())
                        labels = distributed_concat_and_move_to_cpu(labels)
                    else:
                        outputs = batch_move_to_cpu(outputs, valid_keys=outputs.get_eval_output_keys())
                        labels = batch_move_to_cpu(labels)
                    gathered_outputs.append(outputs)
                    gathered_labels.append(labels)
                    self.on_test_step_end()

                # If there's more than one CUDA device, reduce test loss across all devices
                if torch.cuda.device_count() > 1 and self.config.enable_fsdp:
                    dist.all_reduce(self.test_epoch_loss, op=dist.ReduceOp.SUM)

                # Calculate average loss in an epoch over all samples
                self.test_epoch_loss = self.test_epoch_loss / len(self.test_dataset)

                # Gather all outputs
                gathered_outputs = gather_tensor(gathered_outputs, limit=len(self.test_dataset))
                gathered_labels = gather_tensor(gathered_labels, limit=len(self.test_dataset))
                self.test_metric_info = self.evaluator(
                    outputs=gathered_outputs,
                    labels=gathered_labels, 
                    eval_task_info=self.task_info['test'][test_data_idx],
                    stage='Test'
                )
            
            self.all_test_metric_info[test_data_idx] = self.test_metric_info
            self.all_test_outputs[test_data_idx] = {
                'outputs': gathered_outputs,
                'labels': gathered_labels,
                'metrics': self.test_metric_info
            }
            self.is_test_finished = True if test_data_idx == len(self.dataloaders['test']) - 1 else False
            self.on_test_epoch_end()
        return self.all_test_outputs
    
    def _setup_callbacks(self):
        for c in self.callbacks:
            c.setup(self)
            setattr(self, c.name, c)
    
    def on_train_step_start(self, step):
        for c in self.callbacks:
            c.on_train_batch_start(self)
    
    def on_train_step_end(self):
        self.global_step += 1
        self.data_step += 1
        self.epoch_step += 1
        for c in self.callbacks:
            c.on_train_batch_end(self)
    
    def on_train_epoch_start(self, epoch):
        self.model.train()
        self.epoch = epoch
        self.train_epoch_loss = torch.tensor(0.0, device=self.device)
        self.is_best_epoch_so_far = False
        for c in self.callbacks:
            c.on_train_epoch_start(self)
    
    def on_train_epoch_end(self):
        for c in self.callbacks:
            c.on_train_epoch_end(self)
    
    def on_validation_step_start(self):
        for c in self.callbacks:
            c.on_validation_batch_start(self)
    
    def on_validation_step_end(self):
        for c in self.callbacks:
            c.on_validation_batch_end(self)
    
    def on_validation_epoch_start(self):
        self.model.eval()
        self.val_preds = []
        self.val_epoch_loss = torch.tensor(0.0, device=self.device)
        for c in self.callbacks:
            c.on_validation_epoch_start(self)
    
    def on_validation_epoch_end(self):
        for c in self.callbacks:
            c.on_validation_epoch_end(self)
    
    def on_test_step_start(self):
        for c in self.callbacks:
            c.on_test_batch_start(self)
    
    def on_test_step_end(self):
        for c in self.callbacks:
            c.on_test_batch_end(self)
    
    def on_test_epoch_start(self):
        self.model.eval()
        self.test_preds = []
        self.test_epoch_loss = torch.tensor(0.0, device=self.device)
        for c in self.callbacks:
            c.on_test_epoch_start(self)
    
    def on_test_epoch_end(self):
        for c in self.callbacks:
            c.on_test_epoch_end(self)