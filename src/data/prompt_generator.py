# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

import numpy as np
from typing import List, Dict
from .template import template_cls


class PrompteGenerator(object):

    def __init__(
        self,
        tokenizer,
        config,
        meta_info,
        task_info
    ):    
        self.tokenizer = tokenizer
        self.config = config
        self.meta_info = meta_info
        self.task_info = task_info
        self.template = template_cls(self.config.template)(self.tokenizer, self.config, self.meta_info, self.task_info)
        self.token_id_dict = self.tokenizer.special_token_ids

    def add_mask_by_special_token(
        self,
        result,
        mask_key,
        begin_token=None,
        end_token=None
    ):
        mask_arr = np.zeros(len(result['input_ids']))
        if begin_token is not None and end_token is not None:
            begin_indices = np.where(np.array(result['input_ids']) == self.token_id_dict[begin_token])[0]
            end_indices = np.where(np.array(result['input_ids']) == self.token_id_dict[end_token])[0]
            if len(begin_indices) > 0:
                # Eg: <BEGIN>-0.44<END> will mask tokens in ['_-', 0, ., 4, 4]
                # Eg: <BEGIN>0.44<END> will mask tokens in ['_', 0, ., 4, 4]
                mask_indices = np.concatenate([
                    np.arange(begin_indices[i]+1, end_indices[i])
                        for i in range(len(begin_indices))
                ])
                mask_arr[mask_indices] = 1
                # Keep space before each number
                remove_special_tokens = np.concatenate([begin_indices, end_indices], axis=0)
            else:
                remove_special_tokens = np.array([])
        else:
            raise ValueError(f"No available special token to locat mask position.")
        result[mask_key] = mask_arr.astype(int).tolist()
        return result, remove_special_tokens

    def remove_tokens(self, result, remove_keys, remove_tokens):
        if self.config.remove_special_token:
            for key in remove_keys:
                result[key] = np.delete(np.array(result[key]), remove_tokens.astype(int)).tolist()

    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        
        if len(result['input_ids']) == self.config.cutoff_len:
            raise ValueError(f"prompt length exceeds cutoff length")
        
        # Add numeric and answer mask
        numeric_mask_key = 'numeric_mask'
        answer_mask_key = 'answer_mask'
        result, remove_num_tokens = self.add_mask_by_special_token(
            result, numeric_mask_key, begin_token=self.tokenizer.num_begin_token, end_token=self.tokenizer.num_end_token
        )
        result, remove_answer_tokens = self.add_mask_by_special_token(
            result, answer_mask_key, begin_token=self.tokenizer.answer_begin_token, end_token=self.tokenizer.answer_end_token
        )
        
        # Remove special tokens
        remove_keys = [numeric_mask_key, answer_mask_key, 'input_ids', 'attention_mask']
        remove_tokens = np.concatenate([remove_num_tokens, remove_answer_tokens], axis=0) 
        self.remove_tokens(result, remove_keys, remove_tokens)
        
        # Replace result keys
        id_key, mask_key, label_key = 'input_ids', 'attention_mask', 'labels'

        # Add eos token
        if (
            result[id_key][-1] != self.tokenizer.eos_token_id
            and len(result[id_key]) < self.config.cutoff_len
            and add_eos_token
        ):
            result[id_key].append(self.tokenizer.eos_token_id)
            result[mask_key].append(1)
            result[numeric_mask_key].append(0)
            result[answer_mask_key].append(1)
        result[label_key] = result[id_key].copy()

        return result

    def generate_feature_prompt(self, sample: Dict):
        sample = self.template.generate_feature_prompt(sample)
        return sample

    def generate_full_prompt(self, sample: Dict, context_samples: List[Dict] = None):
        sample = self.template.generate_full_prompt(sample, context_samples)
        
        # Remove useless content
        useless_keys = ['feature_prompt']
        for key in useless_keys:
            sample.pop(key)
        
        # Tokenization
        sample.update(self.tokenize(sample['prompt']))

        # Remove special token in prompts
        for t in self.tokenizer.additional_special_tokens:
            sample['prompt'] = sample['prompt'].replace(t, ' ')

        # After processing, sample has the following keys:
        # {
        #     'sample_idx': int,
        #     'task_idx': int,
        #     'context_indices': List[int],
        #     'context_labels': List[int],
        #     'num_perm_indices': int,
        #     'label_tabular': Union[int, float],
        #     'norm_label_tabular': float,
        #     'num_feats': List[Union[int, float]],
        #     'norm_feats': List[Union[int, float]],
        #     'num_feats_mask': List[int],
        #     'cat_feats': List[str],
        #     'cat_token_ids': List[int],
        #     'prompt': str,
        #     'prompt_with_num': str,
        #     'input_ids': List[int],
        #     'attention_mask': List[int],
        #     'numeric_mask': List[int],
        #     'answer_mask': List[int],
        #     'labels': List[int],
        # }
        return sample