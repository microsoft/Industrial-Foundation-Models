# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

import math
import numpy as np
from typing import Set, Union, Tuple, Optional, List
from datasets import DatasetDict, Dataset

class TaskInfo:
    
    num_features: List[str] = None
    cat_features: List[str] = None
    label_column: str = None
    task_type: str = None
    class_num: int = None
    num_feature_indices: List[int] = None
    cat_feature_indices: List[int] = None
    label_idx: int = None
    label_mean: float = None
    label_std: float = None
    context_cnt: int = None
    context_seed: int = None
    context_sample_mode: str = None
    context_balanced: bool = None
    template: str = None
    
    def __init__(self, config, meta_info):
        self.num_features = meta_info['basic_info']['num_features']
        self.cat_features = meta_info['basic_info']['cat_features'] + meta_info['basic_info']['other_features']
        self.label_column = meta_info['basic_info']['label_candidates'][0]
        self.task_type = meta_info['task_info'][self.label_column]['task_type']
        self.num_feature_indices = list(range(len(self.num_features)))
        self.cat_feature_indices = list(range(len(self.cat_features)))
        # Set task-agnostic config to task info
        self.context_cnt = config.in_context_count
        self.context_seed = config.context_sample_seed
        self.context_sample_mode = config.context_sample_mode
        self.context_balanced = config.use_balanced_context
        self.template = config.template

    def __str__(self):
        return str(self.__dict__)
        

class TaskGenerator:
    
    def __init__(
        self,
        tokenizer,
        config,
        meta_info
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.meta_info = meta_info
        # Extract original information from meta_info
        self.label_candidates = meta_info['basic_info']['label_candidates']
        self.default_label_column = self.label_candidates[0]
        self.num_feat_size = len(meta_info['basic_info']['num_features'])
        self.cat_feat_size = len(meta_info['basic_info']['cat_features'] + meta_info['basic_info']['other_features'])
    
    def select_label(self, task_info: TaskInfo):
        if self.config.specified_label != "":
            task_info.label_column = self.config.specified_label
        elif self.config.specified_label_idx != 0:
            if self.config.specified_label_idx >= len(self.label_candidates):
                raise ValueError(f"Label index {self.config.specified_label_idx} out of range")
            task_info.label_column = self.label_candidates[self.config.specified_label_idx]
        elif self.config.random_select_label:
            task_info.label_column = np.random.choice(self.label_candidates)
        else:
            task_info.label_column = self.default_label_column

        # Set new feature info
        if task_info.label_column in task_info.num_features:
            task_info.label_idx = task_info.num_features.index(task_info.label_column)
            task_info.task_type = 'regression'
            task_info.class_num = 1
            task_info.num_features.pop(task_info.label_idx)
            task_info.num_feature_indices.pop(task_info.label_idx)
        elif task_info.label_column in task_info.cat_features:
            task_info.label_idx = task_info.cat_features.index(task_info.label_column)
            task_info.task_type = 'classification'
            task_info.class_num = self.meta_info['task_info'][task_info.label_column]['class_num']
            task_info.cat_features.pop(task_info.label_idx)
            task_info.cat_feature_indices.pop(task_info.label_idx)
        else:
            raise ValueError(f"Label column {task_info.label_column} not found")
        
        return task_info
    
    def drop_features(self, task_info: TaskInfo):
        if self.config.random_drop_features:
            # Randomly drop numerical features
            if len(task_info.num_features) > 0:
                num_feat_drop_cnt = max(1, math.ceil(self.config.drop_features_ratio * len(task_info.num_features)))
                num_feat_drop_indices = np.random.choice(len(task_info.num_features), num_feat_drop_cnt, replace=False)
                num_feat_drop_mask = np.zeros(len(task_info.num_features), dtype=np.bool)
                num_feat_drop_mask[num_feat_drop_indices] = True
                task_info.num_features = np.array(task_info.num_features)[~num_feat_drop_mask].tolist()
                task_info.num_feature_indices = np.array(task_info.num_feature_indices)[~num_feat_drop_mask].tolist()
            
            # Randomly drop categorical features
            if len(task_info.cat_features) > 0:
                cat_feat_drop_cnt = max(1, math.ceil(self.config.drop_features_ratio * len(task_info.cat_features)))
                cat_feat_drop_indices = np.random.choice(len(task_info.cat_features), cat_feat_drop_cnt, replace=False)
                cat_feat_drop_mask = np.zeros(len(task_info.cat_features), dtype=np.bool)
                cat_feat_drop_mask[cat_feat_drop_indices] = True
                task_info.cat_features = np.array(task_info.cat_features)[~cat_feat_drop_mask].tolist()
                task_info.cat_feature_indices = np.array(task_info.cat_feature_indices)[~cat_feat_drop_mask].tolist()
        
        return task_info
    
    def construct_task_data(self, data: Dataset, task_info: TaskInfo):
        notes_indices = task_info.num_feature_indices + (np.array(task_info.cat_feature_indices) + self.num_feat_size).tolist()
        
        # Reset label
        if task_info.task_type == 'regression':
            data = data.filter(lambda sample: sample['num_feats_mask'][task_info.label_idx] != 0)
            label_tabular = np.array(data['num_feats'])[:, task_info.label_idx]
        else:
            if self.config.load_synthetic:
                label_tabular = np.array(data['cat_feats'])[:, task_info.label_idx].astype(float)
                data = data.filter(lambda sample:
                    sample['cat_feats_mask'][task_info.label_idx] == 1
                )
            else:
                class_index_dict = self.meta_info['task_info'][task_info.label_column]['class_index_dict']
                valid_class_values = class_index_dict.keys()
                data = data.filter(lambda sample:
                    sample['cat_feats_mask'][task_info.label_idx] == 1 and
                    sample['cat_feats'][task_info.label_idx] in valid_class_values
                )
                map_func = np.vectorize(lambda x: class_index_dict[x])
                label_tabular = map_func(np.array(data['cat_feats'])[:, task_info.label_idx]).astype(float)
        
        # Reset features and notes
        task_data = Dataset.from_dict({
            'sample_idx': data['sample_idx'],
            'task_idx': data['task_idx'],
            'num_feats': np.array(data['num_feats'])[:, task_info.num_feature_indices],
            'num_feats_mask': np.array(data['num_feats_mask'])[:, task_info.num_feature_indices],
            'cat_feats': np.array(data['cat_feats'])[:, task_info.cat_feature_indices],
            'cat_feats_mask': np.array(data['cat_feats_mask'])[:, task_info.cat_feature_indices],
            # 'notes_template': np.array(data['notes_template'])[:, notes_indices],
            'label_tabular': label_tabular,
        })
    
        return task_data
    
    def generate_task_info(self):
        # Generate task info
        task_info = TaskInfo(self.config, self.meta_info)
        task_info = self.select_label(task_info)
        task_info = self.drop_features(task_info)
        return task_info
