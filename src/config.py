# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

import os
import json
import argparse
import hashlib

from .utils import thread_safe_log


class Config(object):
    _default_file_name = 'main_config.json'

    def __init__(self):
        # ------- Basic Arguments -------
        self.seed = 0

    def update_by_dict(self, config_dict):
        for key, val in config_dict.items():
            setattr(self, key, val)

    def to_dict(self):
        return dict(self.__dict__)
    
    def generate_train_seed_by_task_settings(self):
        # We want different context settings to have different set of training data
        # So we generate a random seed for each context setting automatically
        hash_key = f'{self.in_context_count}_{self.context_sample_seed}'
        if self.train_seed_change_by_template:
            hash_key += f'_{self.template}'
        if self.specified_label_idx != 0:
            hash_key += f'_{self.specified_label_idx}'
        self.train_sample_seed = int(hashlib.sha256(hash_key.encode()).hexdigest(), 16) % 1000000
    
    def generate_component_save_path(self, component, root_path=None, data_idx=None, global_step=None, rank=None, file_type=None):
        if '_' in component:
            # raise ValueError('Component name should not contain "_"')
            component = component.replace('_', '-')
        return os.path.join(
            self.output_path if root_path is None else root_path,
            f'{component}' + 
            (f'_data-{data_idx}' if data_idx is not None else '') +
            (f'_step-{global_step}' if global_step is not None else '') +
            (f'_rank-{rank}' if rank is not None else '') +
            (f'.{file_type}' if file_type is not None else '')
        )
    
    def update_by_pretrained_config(self):
        # Load pretrained model and overwrite config
        overwrite_keys = [
            'enable_fsdp',
            'load_in_kbit', 
            'use_fp16', 
            'pure_bf16',
            'use_numeric_token_loss',
            'use_numeric_head',
        ]
        pretrain_root_path = self.reload_ckpt_folder.rsplit('/', 1)[0]
        pretrain_config = self.load_from(pretrain_root_path).to_dict()
        overwrite_config = {
            k: pretrain_config[k] for k in overwrite_keys
            if k in pretrain_config and k in self.__dict__
        }
        # Print the arguments as name-value pairs
        thread_safe_log('>>> Overwrite arguments:')
        for k, v in overwrite_config.items():
            thread_safe_log(f"{k}: {v}")
        self.update_by_dict(overwrite_config)
    
    def save(self):
        output_dir = getattr(self, 'output_path', None)
        if output_dir is None:
            raise Exception('output_path is not specified')
        else:
            os.makedirs(output_dir, exist_ok=True)
            fp = os.path.join(output_dir, self._default_file_name)
            with open(fp, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load_from(cls, path):
        if os.path.isdir(path):
            path = os.path.join(path, cls._default_file_name)

        with open(path, 'r') as f:
            config_dict = json.load(f)
        config = cls()
        config.update_by_dict(config_dict)
        return config


def strtobool(str_val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    str_val = str_val.lower()
    if str_val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif str_val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (str_val,))


def add_config_to_argparse(config, arg_parser):
    """The helper for adding configuration attributes to the argument parser"""
    for key, val in config.to_dict().items():
        if isinstance(val, bool):
            arg_parser.add_argument('--' + key, type=strtobool, default=val)
        elif isinstance(val, (int, float, str)):
            arg_parser.add_argument('--' + key, type=type(val), default=val)
        else:
            raise Exception('Do not support value ({}) type ({})'.format(val, type(val)))


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='LLM meets tabular data.')
    # Training data arguments
    parser.add_argument('--data_path', type=str, default='data/raw_data',
                        help='Path to huggingface dataset.')
    parser.add_argument('--meta_info_path', type=str, default='data/meta_info',
                        help='Path to meta information.')
    parser.add_argument('--database_config_path', type=str, default='data/database_config',
                        help='Path to database config.')
    parser.add_argument('--database', type=str, default='',
                        help='Database config name for training.')
    parser.add_argument('--datasets', type=str,
                        help='Datasets for training. For example, you can use "diabetes,health" to load multiple datasets.')
    parser.add_argument('--load_synthetic', type=strtobool, default=False,
                        help='Whether to load synthetic data.')
    parser.add_argument('--tmp_path', type=str, default='/data/tmp',
                        help='Temporal path for huggingface datasets cache.')
    # Reload data arguments
    parser.add_argument('--reload_data', type=strtobool, default=True,
                        help='Whether to reload processed data with prompts.')
    parser.add_argument('--reload_data_path', type=str, default='data/processed',
                        help='Path to reload processed data with prompts.')
    parser.add_argument('--save_data', type=strtobool, default=True,
                        help='Whether to save processed data with prompts.')
    parser.add_argument('--save_data_path', type=str, default='data/processed',
                        help='Path to save processed data with prompts.')
    parser.add_argument('--reload_directly', type=strtobool, default=False,
                        help='Whether to load database directly from absolute path.')
    parser.add_argument('--train_database', type=str, default="",
                        help='Database config for training.')
    parser.add_argument('--val_database', type=str, default="",
                        help='Database config for validation.')
    parser.add_argument('--test_database', type=str, default="",
                        help='Database config for test.')
    # Model arguments
    parser.add_argument('--model_path', type=str, help='Path to LLaMA weights.')
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2",
                        help='Implementation of attention layer. Options: ["eager", "flash_attention_2", "sdpa"]. Refer to https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM.from_config.attn_implementation for details.')
    parser.add_argument('--output_path', type=str, default='./exps',
                        help='Path to save the checkpoints and metrics.')
    parser.add_argument('--use_legacy_generate', type=strtobool, default=False,
                        help='Whether to use legacy generation.')
    # Workflow arguments
    parser.add_argument('--tabfm_stage', type=str, default='pretrain',
                        choices=['pretrain', 'eval', 'generate_data'], help='Stage of TabFM')
    parser.add_argument('--eval_datasets', type=str, default='diabetes',
                        help='Only used when tabfm_stage=eval, datasets for evaluation. For example, you can use "income,breast-cancer".')
    parser.add_argument('--eval_database', type=str, default='',
                        help='Only used when tabfm_stage=eval, datasets for evaluation. For example, you can use "holdout_demo".')
    parser.add_argument('--eval_seeds', type=str, default='0',
                        help='Only used when tabfm_stage=eval, seeds for data split. For example, you can use "0,1,2,3,4".')
    parser.add_argument('--eval_contexts', type=str, default='0',
                        help='Only used when tabfm_stage=eval, the number of context examples for each sample. For example, you can use "0,4,8".')
    parser.add_argument('--eval_context_seeds', type=str, default='0',
                        help='Only used when tabfm_stage=eval, the seed of global context. For example, you can use "0,1,2,3,4".')
    # Accelerator arguments
    parser.add_argument('--enable_fsdp', type=strtobool, default=True,
                        help='Whether to use FSDP.')
    parser.add_argument('--use_act_ckpt', type=strtobool, default=True,
                        help='Whether to use activation checkpointing.')
    parser.add_argument('--act_ckpt_interval', type=int, default=1,
                        help='The interval of performing activation checkpointing operations.')
    parser.add_argument('--load_in_kbit', type=strtobool, default=False,
                        help='Whether to quantize the base model.')
    parser.add_argument('--use_fp16', type=strtobool, default=False,
                        help='Whether to use fp16.')
    parser.add_argument('--pure_bf16', type=strtobool, default=True,
                        help='Whether to use pure bf16.')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='Optimizer type, adamw or anyprecision.')
    parser.add_argument('--use_clip_grad', type=strtobool, default=True,
                        help='Whether to clip gradient for preventing exploding gradients.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm (for gradient clipping).')
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enabling wandb logging.')
    parser.add_argument('--wandb_project', type=str, default='LLaMA2-GTL',
                        help='Wandb project name.')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name.')
    parser.add_argument('--wandb_key_file', type=str, default=None,
                        help='Wandb API key file.')
    # Debug arguments
    parser.add_argument('--debug_mode', type=strtobool, default=False,
                        help="Debug mode for data and evaluation")
    parser.add_argument('--debug_max_val_steps', type=int, default=4,
                        help='Maximum validation steps for debug mode.')
    parser.add_argument('--debug_max_test_steps', type=int, default=4,
                        help='Maximum test steps for debug mode.')
    # Data arguments
    parser.add_argument('--test_sample_seed', type=int, default=0,
                        help='Random seed for sample test data.')
    parser.add_argument('--train_sample_seed', type=int, default=-1,
                        help='Random seed for sample train data.')
    parser.add_argument('--num_shots_test', type=str, default='16',
                        help='Number of samples for test.')
    parser.add_argument('--num_shots_train', type=str, default='64',
                        help='Number of samples for training.')
    parser.add_argument('--norm', type=strtobool, default=True,
                        help='Whether to normalize numerical features.')
    parser.add_argument('--remove_special_token', type=strtobool, default=True,
                        help='Whether to remove special tokens of numeric value templates.')
    parser.add_argument('--max_decimal', type=int, default=4,
                        help='Maximum number of decimal places to keep for numerical values.')
    # Task arguments
    parser.add_argument('--specified_label', type=str, default="",
                        help='Specified label to generate task.')
    parser.add_argument('--specified_label_idx', type=int, default=0,
                        help='Specified label index in meta info to generate task.')
    parser.add_argument('--random_select_label', type=strtobool, default=False,
                        help='Whether to randomly select a label from candidates.')
    parser.add_argument('--random_drop_features', type=strtobool, default=False,
                        help='Whether to drop features randomly.')
    parser.add_argument('--drop_features_ratio', type=float, default=0.1,
                        help='Ratio of dropping features.')
    # In-context learning
    parser.add_argument('--in_context_count', type=int, default=0,
                        help='The number of in-context learning examples.')
    parser.add_argument('--use_balanced_context', type=strtobool, default=True,
                        help='Whether to use balanced context samples for in-context learning.')
    parser.add_argument('--context_sample_mode', type=str, default='global', choices=['random', 'global'],
                        help='Mode of sampling context.')
    parser.add_argument('--global_context_groups', type=int, default=1,
                        help='Number of groups sampling global context.')
    parser.add_argument('--context_sample_seed', type=int, default=0,
                        help='Random seed for sample context.')
    # Template arguments
    parser.add_argument('--template', type=str, default='table', choices=['language', 'table', 'anonymous_table'],
                        help='Prompt template.')
    parser.add_argument('--train_seed_change_by_template', type=strtobool, default=False,
                        help='Whether to change train seed by template.')
    # Training arguments
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--full_deterministic', action='store_true',
                        help='Whether to enable fully deterministic training.')
    parser.add_argument('--cutoff_len', type=int, default=4096,
                        help='Maximum length of input tokens')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Maximum number of epochs for each training datasets.')
    parser.add_argument('--max_train_steps', type=str, default='0',
                        help='Maximum training steps. For example, you can use "8192,4096" for diffrerent train datasets.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Global batch size')
    parser.add_argument('--micro_batch_size', type=str, default='4',
                        help='Train batch size per GPU. For example, you can use "4,2,1" for diffrerent train datasets.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--lr_schedule_type', type=str, default='none',
                        help='Learning rate scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warm-up steps')
    parser.add_argument('--reload_model_before_training', type=strtobool, default=True,
                        help='Whether to reload a checkpoint before training.')
    parser.add_argument('--reload_ckpt_folder', type=str, default=None,
                        help='Specify a exact checkpoint folder to model reloading.')
    parser.add_argument('--reload_optimizer', type=strtobool, default=True,
                        help='Whether to reload optimizer when reload pretrained model.')
    # Evaluation arguments
    parser.add_argument('--val_max_generate_length', type=str, default='10',
                        help='Maximum length of generated tokens for validation. For example, you can use "2,10" for diffrerent val datasets.')
    parser.add_argument('--val_batch_size', type=str, default='4',
                        help='Val batch size per GPU. For example, you can use "4,2,1" for diffrerent validation datasets.')
    parser.add_argument('--val_every_n_steps', type=int, default=0,
                        help='Test for every n trainig steps. If set to 0, we will not evaluate during a training epoch.')
    parser.add_argument('--val_per_epoch', type=strtobool, default=True,
                        help='Whether to run test per epoch end.')
    parser.add_argument('--val_before_training', type=strtobool, default=True,
                        help='Whether to run test before training.')
    parser.add_argument('--test_max_generate_length', type=str, default='10',
                        help='Maximum length of generated tokens for test. For example, you can use "2,10" for diffrerent test datasets.')
    parser.add_argument('--test_batch_size', type=str, default='4',
                        help='Test batch size per GPU. For example, you can use "4,2,1" for diffrerent test datasets.')
    # Early stopping arguments
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Earlystopping patience (the number of training epochs)')
    parser.add_argument('--metric_for_best_model', type=str, default='All/AUROC',
                        help='Metric for the best model')
    parser.add_argument('--greater_is_better', type=strtobool, default=True,
                        help='Whether the metric for the best model should be maximized or not.')
    # Save arguments
    parser.add_argument('--save_config', type=strtobool, default=True,
                        help='Whether to save the config json into the experiment directory.')
    parser.add_argument('--save_single_dataset', type=strtobool, default=False,
                        help='Whether to save single dataset.')
    parser.add_argument('--save_model', type=strtobool, default=True,
                        help='Whether to save checkpoint.')
    parser.add_argument('--save_optimizer', type=strtobool, default=True,
                        help='Whether to save optimizer.')
    parser.add_argument('--save_eval_metrics', type=strtobool, default=True,
                        help='Whether to save evaluation metrics.')
    parser.add_argument('--save_best_ckpt', type=strtobool, default=False,
                        help='Whether to save every best checkpoint.')
    parser.add_argument('--save_ckpt_every_n_steps', type=int, default=0,
                        help='Number of global training steps to save a checkpoint.')
    parser.add_argument('--save_ckpt_per_epoch', type=strtobool, default=True,
                        help='Whether to save a checkpoint after each train epoch.')
    parser.add_argument('--save_ckpt_per_each_val', type=strtobool, default=True,
                        help='Whether to save a checkpoint before each validation epoch.')
    # TabCausalLM arguments
    parser.add_argument('--change_rope_base_freq', type=strtobool, default=False,
                        help='Whether to change the base frequency of rotary embeddings in the model to support longer sequences')
    parser.add_argument('--use_numeric_token_loss', type=strtobool, default=True,
                        help='Whether to calculate numeric token loss')
    parser.add_argument('--numeric_token_loss_weight', type=float, default=1,
                        help='Weight of the loss of predicting numerical tokens.')
    parser.add_argument('--use_numeric_head', type=strtobool, default=False,
                        help='Whether to use numeric head to recover the numeric value.')
    parser.add_argument('--numeric_feat_loss_weight', type=float, default=1,
                        help='Weight of the loss of predicting numerical features.')
    parser.add_argument('--use_weighted_numeric_loss', type=strtobool, default=False,
                        help='Whether to calculate weighted numeric loss for numeric tokens to mimic decimal values.')
    

    args = parser.parse_args(args=args)
    os.environ['TMPDIR'] = args.tmp_path

    # Print the arguments as name-value pairs
    thread_safe_log('>>> Arguments:')
    for arg in vars(args):
        thread_safe_log(f"{arg}: {getattr(args, arg)}")
    return args


def parse_args_to_config():
    args = parse_args()
    config = Config()
    config.update_by_dict(args.__dict__)
    if config.train_sample_seed == -1:
        config.generate_train_seed_by_task_settings()
    if config.reload_ckpt_folder is not None:
        config.update_by_pretrained_config()
    if config.save_config:
        config.save()
    return config