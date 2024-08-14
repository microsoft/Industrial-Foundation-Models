import numpy as np
from typing import List, Tuple
from datasets import Dataset, DatasetDict

class ContextSampler:
    def __init__(self, config, task_info):
        self.config = config
        self.task_info = task_info
        self.sample_cnt = self.config.in_context_count
        self.is_balanced = self.config.use_balanced_context
        self.sample_mode = self.config.context_sample_mode
        self.global_context_groups = self.config.global_context_groups
        self.sample_seed = self.config.context_sample_seed
    
    @property
    def context_features(self):
        return ['num_feats', 'norm_feats', 'num_feats_mask', 'cat_feats', 'cat_token_ids', 'cat_feats_mask']
    
    def gather_context_features(self, ori_data: Dataset, context_data: Dataset, context_indices: np.array) -> Dataset:
        # Construct data with context samples
        data = {
            'context_sample_indices': context_indices,
            'context_data_indices': np.array(context_data['sample_idx'])[context_indices],
            'context_labels': np.array(context_data['label_tabular'])[context_indices],
        }
        for feat in ori_data.features.keys():
            if feat in self.context_features:
                data[feat] = np.concatenate(
                    [np.array(context_data[feat])[context_indices].reshape(len(ori_data), -1), ori_data[feat]], axis=1)
            else:
                data[feat] = ori_data[feat]
        return Dataset.from_dict(data)
    
    def get_label_indices(self, data) -> Tuple[np.array, List[np.array]]:
        labels, inverse_indices = np.unique(data['label_tabular'], return_inverse=True)
        label_indices = []
        for i in range(len(labels)):
            label_indices.append(np.where(inverse_indices == i)[0])
        return labels, label_indices
    
    def sample_balanced_context_indices(self, context_data: Dataset, num_rows: int, seed: int) -> np.array:
        rng = np.random.default_rng(seed)
        labels, label_indices = self.get_label_indices(context_data)
        min_label_cnt = int(self.config.in_context_count // len(labels))
        min_label_value_cnt = len(labels) - int(self.config.in_context_count % len(labels))
        label_cnt_list = [min_label_cnt+1] * (len(labels)-min_label_value_cnt) + [min_label_cnt] * min_label_value_cnt

        # sample
        sample_indices = []
        for i in range(len(labels)):
            indices = np.array(label_indices[i])
            repeat_indices = np.repeat(indices.reshape(1, -1), num_rows, axis=0)
            label_sample_indices = rng.permuted(repeat_indices, axis=1)[:,:min(label_cnt_list[i], len(indices))]
            sample_indices.append(label_sample_indices)
        sample_indices = np.concatenate(sample_indices, axis=1)
        # Whether to keep the label order in context or permute
        sample_indices = rng.permuted(sample_indices, axis=1)
        return sample_indices

    def sample_context(self, context_data: Dataset, num_rows: int) -> np.array:
        if self.config.use_balanced_context and self.task_info.task_type == 'classification':
            context_sample_indices = self.sample_balanced_context_indices(context_data, num_rows, self.sample_seed)
        else:
            rng = np.random.default_rng(self.sample_seed)
            context_sample_indices = np.repeat(np.arange(len(context_data)).reshape(1, -1), num_rows, axis=0)
            context_sample_indices = rng.permuted(context_sample_indices, axis=1)[:,:min(self.sample_cnt, len(context_data))]
        return context_sample_indices

    def get_context_sample_indices(self, context_data: Dataset, num_rows: int) -> np.array:
        if self.sample_mode == 'global':
            global_context_indices = self.sample_context(context_data, self.global_context_groups)
            context_sample_indices = np.repeat(
                np.expand_dims(global_context_indices, 0), num_rows // self.global_context_groups + 1, axis=0
            ).reshape(-1, global_context_indices.shape[-1])[:num_rows]
        elif self.sample_mode == 'random':
            context_sample_indices = self.sample_context(context_data, num_rows)
        else:
            raise ValueError(f"Unsupported context sample mode: {self.sample_mode}")
        return context_sample_indices
    
    def add_context_samples(self, data_dict: DatasetDict) -> DatasetDict:
        context_data = data_dict['context']
        if len(context_data) < self.sample_cnt // 2:
            raise ValueError(f"Context data size {len(context_data)} is too small")
        for t in data_dict.keys():
            if t == 'context':
                continue
            context_sample_indices = self.get_context_sample_indices(context_data, len(data_dict[t]))
            data_dict[t] = self.gather_context_features(data_dict[t], context_data, context_sample_indices)
        
        return data_dict