# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

# ---------------------------------------------------------------------------------
# This file contains some parts inspired by the huggingface transformers library.
# - Source: https://github.com/huggingface/transformers

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

from typing import List, Dict
from transformers import BatchEncoding, DataCollatorForSeq2Seq

class DataCollatorForTabLM(DataCollatorForSeq2Seq):
    """
    The feature input to this class should be aligned with the output from Prompter.generate_prompt().
    Each feature should have the following keys:
        {
            'sample_idx': int,
            'task_idx': int,
            'label_mean': float,
            'label_std': float,
            'context_indices': List[int],
            'context_labels': List[int],
            'label_tabular': Union[int, float],
            'norm_label_tabular': float,
            'num_feats': List[Union[int, float]],
            'norm_feats': List[Union[int, float]],
            'num_feats_mask': List[int],
            'cat_feats': List[str],
            'cat_token_ids': List[int],
            'prompt': str,
            'input_ids': List[int],
            'attention_mask': List[int],
            'numeric_mask': List[int],
            'answer_mask': List[int],
            'labels': List[int],
        }
    """
    keys_without_num: List[str] = [
        'input_ids',
        'attention_mask',
        'numeric_mask',
        'answer_mask',
        'labels',
    ]
    
    key_num_feats: List[str] = [
        'num_feats',
        'norm_feats',
        'num_feats_mask',
    ]
    keys_others: List[str] = [
        'sample_idx',
        'task_idx',
        'label_tabular',
        'norm_label_tabular',
    ]

    def pad_input_sequences(self, features: List[Dict[str, List[int]]], return_tensors=None) -> BatchEncoding:
        fea_key2pad_id = {
            'input_ids': self.tokenizer.pad_token_id,
            'attention_mask': 0,
            'numeric_mask': 0,
            'answer_mask': 0,
            'labels': self.tokenizer.pad_token_id,
        }
        max_length = None

        padding_outputs = {}
        for fea_key in self.keys_without_num:
            pad_id = fea_key2pad_id[fea_key]
            padding_outputs[fea_key] = []
            
            if max_length is None:
                max_length = max(len(feat[fea_key]) for feat in features)
                if self.pad_to_multiple_of is not None and (max_length % self.pad_to_multiple_of != 0):
                    max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
            
            for feat in features:
                difference = max_length - len(feat[fea_key])
                if self.tokenizer.padding_side == "right":
                    padding_outputs[fea_key].append(feat[fea_key] + [pad_id] * difference)
                elif self.tokenizer.padding_side == "left":
                    padding_outputs[fea_key].append([pad_id] * difference + feat[fea_key])
                else:
                    raise ValueError("Invalid padding strategy:" + str(self.tokenizer.padding_side))
        
        return BatchEncoding(padding_outputs, tensor_type=return_tensors)
    
    def pad_numeric_features(self, features: List[Dict], pad_id: int = 0, return_tensors=None) -> BatchEncoding:
        keys = features[0].keys()
        base_key = list(keys)[0]
        max_feature_len = max(len(feat[base_key]) for feat in features)
        
        # Padding each feature for each example to max_feature_len
        padding_outputs = {}
        for key in keys:
            padding_outputs[key] = []
            for example in features:
                output = example[key] + [pad_id] * (max_feature_len - len(example[key]))
                padding_outputs[key].append(output)
        
        return BatchEncoding(padding_outputs, tensor_type=return_tensors)
    
    def __call__(self, features: List[Dict], return_tensors=None) -> BatchEncoding:
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Collating inputs without numerical features
        features_without_num = []
        for feat in features:
            fea = {}
            for key in self.keys_without_num:
                fea[key] = feat[key]
            features_without_num.append(fea)
        batch = self.pad_input_sequences(features_without_num, return_tensors=return_tensors)
        
        # Collating feature inputs
        num_features = []
        for feat in features:
            fea = {}
            for key in self.key_num_feats:
                fea[key] = feat[key]
            num_features.append(fea)
        batch_num_features = self.pad_numeric_features(num_features, return_tensors=return_tensors)
        batch.update(batch_num_features)
        
        # Collating other inputs
        batch_others = {key: [feat[key] for feat in features] for key in self.keys_others}
        batch.update(BatchEncoding(batch_others, tensor_type=return_tensors))

        return batch