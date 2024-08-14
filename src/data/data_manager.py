import os
import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Dataset, Sequence, Value

from .task_generator import TaskGenerator
from .context_sampler import ContextSampler
from .prompt_generator import PrompteGenerator

from src.utils import load_json, save_json

import datasets
datasets.disable_caching()

class DataManager:

    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.rank = int(os.environ.get("RANK") or 0)

    def load_multi_datasets(self, database): 
        multi_datasets = []
        config_path = f"{self.config.database_config_path}/{database}"
        with open(config_path, 'r') as f:
            for dataset in f.readlines():
                multi_datasets.append(dataset.strip())
        return multi_datasets

    def load_database(self, database=None, datasets=None, data_type=['train', 'test']):
        # Load parameters from config
        database = database or self.config.database
        datasets = datasets or self.config.datasets
        
        # Synthetic data is only used for pre-training
        if self.config.load_synthetic:
            data_type = ['train']
        
        # Reload database
        dataset_dict, task_info_dict = self.reload_data(database, data_type)
        if dataset_dict is not None and task_info_dict is not None:
            database_task_info = task_info_dict
        else:
            # Init data config and task_info
            database_task_info = {}
            multi_datasets = self.load_multi_datasets(database) if database != "" else datasets.split(",")
            
            # Load multiple datasets
            dataset_list_dict = {}
            for task_idx, dataset in enumerate(multi_datasets):
                try:
                    print(f"Load dataset {dataset}...")
                    # Reload single dataset
                    data_dict, task_info_dict = self.reload_data(dataset, data_type, task_idx)
                    
                    if data_dict is None or task_info_dict is None:
                        # Load single dataset
                        meta_info = self.load_meta_info(dataset)
                        data_dict = self.load_single_raw_dataset(dataset, meta_info, task_idx)
                        
                        # Generate task
                        data_dict, task_info = self.generate_task(data_dict, meta_info)
                        statics_set = 'train' if 'train' in data_dict else 'context'
                        data_dict = self.process_dataset(data_dict, task_info, statics_set)

                        # Generate prompt
                        data_dict = self.generate_prompt(data_dict, meta_info, task_info, dataset)
                        
                        # Transform feature type for datasets with null values
                        data_dict = self.transform_feature_type(data_dict)
                        
                        # Save single dataset
                        task_info_dict = task_info.__dict__
                        if self.config.save_single_dataset:
                            self.save_data(data_dict, dataset, task_info_dict)
                        data_dict.cleanup_cache_files()
                
                except Exception as err:
                    tb = traceback.format_exc()
                    print(f"Fail to load {dataset}, err={err}\n{tb}")
                    continue
                
                database_task_info[task_idx] = (dataset, task_info_dict)

                # Add datasets
                for k, v in data_dict.items():
                    if k not in dataset_list_dict:
                        dataset_list_dict[k] = []
                    dataset_list_dict[k].append(v)
            
            # Merge datasets
            dataset_dict = DatasetDict({
                t: concatenate_datasets(dataset_list_dict[t]) for t in dataset_list_dict.keys()
            })
            
            # Save database
            if len(database_task_info) > 1:
                self.save_data(dataset_dict, database, database_task_info)
        
        # Log
        print(f"Load {len(database_task_info)} datasets")
        for t in dataset_dict.keys():
            print(f"{t} sample num: {len(dataset_dict[t])}")
        
        # Debug
        if self.config.debug_mode:
            print(dataset_dict[list(dataset_dict.keys())[0]][0])

        return dataset_dict, database_task_info

    def load_data(self):
        data = {'train': [], 'val': [], 'test': []}
        task_info = {'train': [], 'val': [], 'test': []}
        if not self.config.reload_directly:
            dataset_dict, database_task_info = self.load_database(
                database=self.config.database,
                datasets=self.config.datasets
            )
            if 'val' not in dataset_dict and 'test' in dataset_dict:
                dataset_dict['val'] = dataset_dict['test']
            for data_type, datasets in dataset_dict.items():
                data[data_type].append(datasets)
                task_info[data_type].append(database_task_info)
        else:
            # Load train/val/test data directly
            database_dict = {
                'train': self.config.train_database,
                'val': self.config.val_database,
                'test': self.config.test_database
            }
            for data_type, database_str in database_dict.items():
                load_type = 'train' if data_type == 'train' else 'test'
                for db in database_str.split(','):
                    if db == "":
                        break
                    dataset_dict, database_task_info = self.load_database(
                        database=db,
                        data_type=[load_type]
                    )
                    data[data_type].append(dataset_dict[load_type])
                    task_info[data_type].append(database_task_info)
        return data, task_info

    def load_meta_info(self, dataset):
        return load_json(f"{self.config.meta_info_path}/{dataset}.json")

    def load_numerical_features(self, df, meta_info):
        num_cols = meta_info['basic_info']['num_features']
        if len(num_cols) == 0:
            num_feats_mask = [[] for i in range(len(df))]
            num_feats = [[] for i in range(len(df))]
        else:
            num_feats_mask = df[num_cols].notna().astype(int).values.tolist()
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            num_feats = df[num_cols].astype(float).values.tolist()
        return num_feats, num_feats_mask

    def load_categorical_features(self, df, meta_info):
        # regard non-numerical features as categorical features
        cat_cols = meta_info['basic_info']['cat_features'] + meta_info['basic_info']['other_features']
        if len(cat_cols) == 0:
            cat_feats_mask = [[] for i in range(len(df))]
            cat_feats = [[] for i in range(len(df))]
        else:
            cat_feats_mask = df[cat_cols].notna().astype(int).values.tolist()
            df[cat_cols] = df[cat_cols].fillna('unknown').astype(str)
            cat_feats = df[cat_cols].values.tolist()
        return cat_feats, cat_feats_mask

    def sample_data(self, data):
        valid_data_dict = {}
        used_data_indices = []
        
        # Sample test data
        if self.config.num_shots_test != '0':
            test_data = self.sample_few_shot_data(data, self.config.num_shots_test, self.config.test_sample_seed)
            valid_data_dict['test'] = test_data
            used_data_indices = np.concatenate([used_data_indices, np.array(test_data['sample_idx'])])
            unused_data = data.select(np.setdiff1d(range(data.num_rows), used_data_indices))
        
        # Sample train data
        if self.config.num_shots_train != '0' and len(unused_data) > 0:
            train_data = self.sample_few_shot_data(unused_data, self.config.num_shots_train, self.config.train_sample_seed)
            valid_data_dict['train'] = train_data
            used_data_indices = np.concatenate([used_data_indices, np.array(train_data['sample_idx'])])
        
        # Sample context data
        if self.config.in_context_count > 0:
            context_data = data.select(np.setdiff1d(range(data.num_rows), used_data_indices))
            if len(context_data) <= self.config.in_context_count // 2:
                raise ValueError(f"Context pool size {len(context_data)} is too small")
            if self.config.context_sample_mode == 'global':
                # To accelerate, we only sample 4x samples for context data
                context_data = self.sample_few_shot_data(context_data, int(self.config.in_context_count) * 4, self.config.context_sample_seed)
            else:
                context_data = self.sample_few_shot_data(context_data, int(self.config.in_context_count) * int(self.config.num_shots_train), self.config.context_sample_seed)
            valid_data_dict['context'] = context_data

        data_dict = DatasetDict(valid_data_dict)
        data_dict.cleanup_cache_files()
        return data_dict

    def sample_synthetic_data(self, data, meta_info):
        self.config.in_context_count = meta_info['basic_info']['train_num']
        self.config.train_num = meta_info['basic_info']['test_num']
        context_data = data.select(list(range(self.config.in_context_count)))
        train_data = data.select(list(range(self.config.in_context_count, self.config.in_context_count+self.config.train_num)))
        data_dict = DatasetDict({'train': train_data, 'context': context_data})
        data_dict.cleanup_cache_files()
        return data_dict

    def load_single_raw_dataset(self, dataset, meta_info, task_idx):
        # Load
        df = pd.read_csv(f'{self.config.data_path}/{dataset}.csv')
        
        # Construct data
        num_feats, num_feats_mask = self.load_numerical_features(df, meta_info)
        cat_feats, cat_feats_mask = self.load_categorical_features(df, meta_info)
        data = Dataset.from_dict({
            'sample_idx': list(range(0, len(df))),
            'task_idx': [task_idx]*len(df),
            'num_feats': num_feats,
            'cat_feats': cat_feats,
            'num_feats_mask': num_feats_mask,
            'cat_feats_mask': cat_feats_mask
        })
        if not self.config.load_synthetic:
            data_dict = self.sample_data(data)
        else:
            data_dict = self.sample_synthetic_data(data, meta_info)
        
        return data_dict
    
    def add_missing_category_samples(self, data_all, data, class_num):
        data_categories = np.unique(data['label_tabular'])
        if len(data_categories) == class_num:
            return data
        # We only add one sample for each missing category
        min_category_samples_num = 1
        missing_categories = list(set(range(class_num)) - set(data_categories))
        samples = []
        for i, label in enumerate(missing_categories):
            label_idxs = [idx for idx, sample in enumerate(data_all)
                            if sample['label_tabular'] == label]
            sample_idxs = list(np.random.choice(label_idxs, min_category_samples_num))
            samples.append(data_all.select(sample_idxs))
        sample_data = concatenate_datasets(samples)
        return concatenate_datasets([data, sample_data])
    
    def get_balanced_samples(self, data, n_samples, seed):
        # Get balanced samples num 
        labels = np.unique(data['label_tabular'])
        min_label_cnt = int(n_samples // len(labels))
        min_label_value_cnt = len(labels) - int(n_samples % len(labels))
        label_cnt_list = [min_label_cnt+1] * (len(labels)-min_label_value_cnt) + [min_label_cnt] * min_label_value_cnt
        # Sample per label
        rng = np.random.default_rng(seed)
        sample_indices = []
        for i, label in enumerate(labels):
            label_indices = np.where(data['label_tabular'] == label)[0]
            sample_indices.append(
                rng.choice(label_indices, size=min(label_cnt_list[i], len(label_indices)), replace=False)
            )
        sampled_dataset = data.select(np.concatenate(sample_indices))
        return sampled_dataset
    
    def sample_few_shot_data(self, data, num_shots, sample_seed, is_balanced=False):
        if num_shots == 'all' or int(num_shots) > len(data):
            return data
        if is_balanced:
            sampled_dataset = self.get_balanced_samples(data, int(num_shots), seed=sample_seed)
        else:
            rng = np.random.default_rng(sample_seed)
            sampled_dataset = data.select(rng.choice(len(data), size=int(num_shots), replace=False))
        return sampled_dataset

    def generate_task(self, data_dict, meta_info):
        task_generator = TaskGenerator(
            tokenizer=self.tokenizer,
            config=self.config,
            meta_info=meta_info
        )
        task_info = task_generator.generate_task_info()
        for t in data_dict.keys():
            data_dict[t] = task_generator.construct_task_data(data_dict[t], task_info)
        return data_dict, task_info

    def process_dataset(self, data_dict, task_info, statics_set='train'):
        # Process features and labels
        self.tokenize_cat_features(data_dict, statics_set)
        self.normalize_num_features(data_dict, statics_set)
        self.normalize_label_tabular(data_dict, task_info, statics_set)
        
        data_dict.cleanup_cache_files()
        return data_dict
    
    def transform_feature_type(self, data_dict):
        for k, v in data_dict.items():
            new_features = v.features.copy()
            new_features['num_feats'] = Sequence(feature=Value('float64'))
            new_features['norm_feats'] = Sequence(feature=Value('float64'))
            new_features['num_feats_mask'] = Sequence(feature=Value('int64'))
            new_features['cat_feats'] = Sequence(feature=Value('string'))
            new_features['cat_token_ids'] = Sequence(feature=Value('int64'))
            new_features['cat_feats_mask'] = Sequence(feature=Value('int64'))
            new_features['label_tabular'] = Value('float64')
            new_features['norm_label_tabular'] = Value('float64')
            data_dict[k] = v.cast(new_features)
        return data_dict

    def normalize_num_features(self, data_dict, statics_set):
        if statics_set not in data_dict or len(data_dict[statics_set]['num_feats'][0]) == 0:
            scaler = FunctionTransformer()
        else:
            scaler = StandardScaler()
            scaler.fit(data_dict[statics_set]['num_feats'])
        for t in data_dict.keys():
            data_dict[t] = data_dict[t].add_column(
                name='norm_feats',
                column=scaler.transform(np.array(data_dict[t]['num_feats'])).tolist()
            )
    
    def normalize_label_tabular(self, data_dict, task_info, statics_set):
        if statics_set not in data_dict or task_info.task_type != 'regression':
            scaler = FunctionTransformer()
        else:
            scaler = StandardScaler()
            scaler.fit(np.array(data_dict[statics_set]['label_tabular']).reshape(-1, 1))
            label_mean = scaler.mean_[0]
            label_std = scaler.scale_[0]
            task_info.label_mean = label_mean
            task_info.label_std = label_std
        for t in data_dict.keys():
            # Set label statics
            data_dict[t] = data_dict[t].add_column(
                name='norm_label_tabular',
                column=scaler.transform(np.array(data_dict[t]['label_tabular']).reshape(-1, 1)).reshape(-1).tolist()
            )
    
    def tokenize_cat_features(self, data_dict, statics_set):
        # If there is no train set or context set, we use test set to tokenize categorical feature3s
        if statics_set not in data_dict:
            statics_set = 'test'
        if len(data_dict[statics_set]['cat_feats'][0]) == 0:
            tokenizer = FunctionTransformer()
        else:
            tokenizer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            tokenizer.fit(data_dict[statics_set]['cat_feats'])
        for t in data_dict.keys():
            data_dict[t] = data_dict[t].add_column(
                name='cat_token_ids',
                column=np.array(tokenizer.transform(data_dict[t]['cat_feats'])).tolist()
            )

    def add_context_samples(self, data_dict, task_info):
        context_sampler = ContextSampler(self.config, task_info)
        data_dict = context_sampler.add_context_samples(data_dict)
        return data_dict

    def generate_prompt(self, data_dict, meta_info, task_info, dataset):
        prompt_generator = PrompteGenerator(
            tokenizer=self.tokenizer,
            config=self.config,
            meta_info=meta_info,
            task_info=task_info
        )
        # Generate feature prompts
        data_dict = data_dict.map(prompt_generator.generate_feature_prompt)
        
        # Add context samples
        if self.config.in_context_count > 0:
            data_dict = self.add_context_samples(data_dict, task_info)

        # Generate full prompts with context samples
        prompt_data_dict = {}
        for t in data_dict.keys():
            if t == 'context':
                continue
            num_proc = len(data_dict[t])
            prompt_data_dict[t] = data_dict[t].map(lambda sample:
                prompt_generator.generate_full_prompt(
                    sample=sample,
                    context_samples=data_dict['context'].select(sample['context_sample_indices'])
                        if self.config.in_context_count > 0 else None
                ),
                num_proc=min(num_proc, 16), desc=f'generate_full_prompt_{dataset}'
            )
        prompt_data_dict = DatasetDict(prompt_data_dict)

        return prompt_data_dict
    
    def get_data_config_path(self, data_dir, dataset):
        config_name = f'seed{self.config.seed}' \
                      f'_data_seed{self.config.train_sample_seed}_{self.config.test_sample_seed}' \
                      f'_shot{self.config.num_shots_train}_{self.config.num_shots_test}' \
                      f'_context_m{self.config.context_sample_mode}' \
                      f'_g{self.config.global_context_groups}' \
                      f'_n{self.config.in_context_count}' \
                      f'_s{self.config.context_sample_seed}' \
                      f'_{self.config.template}' \
                      f'_tc{self.config.train_seed_change_by_template}' \
                      f'_c{self.config.cutoff_len}'
        if self.config.specified_label != "":
            config_name += f'_lab{self.config.specified_label}'
        elif self.config.specified_label_idx != 0:
            config_name += f'_labid{self.config.specified_label_idx}'
        elif self.config.random_select_label:
            config_name += f'_labrand'
        if self.config.random_drop_features:
            config_name += f'_drop{self.config.drop_features_ratio}'
        return os.path.join(data_dir, dataset, config_name)
    
    def reload_data(self, dataset, data_type, task_idx=None):
        if not self.config.reload_data or dataset is None or dataset == "":
            return None, None

        if self.config.reload_directly:
            # We directly reload test set from the path
            # Note that we should construct test set and merge task info offline manually for this case
            data_path = dataset
            assert os.path.exists(data_path), f"Test set {data_path} not found"
        else:
            data_path = self.get_data_config_path(self.config.reload_data_path, dataset)
        try:
            print(f"Reload dataset {dataset} from {data_path}")
            data_dict = DatasetDict({t: load_from_disk(f'{data_path}/{t}') for t in data_type})
            task_info = load_json(f"{data_path}/task_info.json")
            # Reset task index to ensure consistency with task_info
            if task_idx is not None:
                for t in data_type:
                    data_dict[t] = data_dict[t].remove_columns(["task_idx"])
                    data_dict[t] = data_dict[t].add_column(name='task_idx', column=[task_idx]*data_dict[t].num_rows)
        except Exception as err:
            print(f"Fail to reload {data_path}, err={err}")
            data_dict, task_info = None, None
        return data_dict, task_info
    
    def save_data(self, data_dict, dataset, task_info):
        if not self.config.save_data or dataset is None:
            return
        # save data for reload directly
        if self.rank == 0:
            save_path = self.get_data_config_path(self.config.save_data_path, dataset)
            print(f"Save dataset {dataset} to {save_path}")
            data_dict.save_to_disk(save_path, max_shard_size="3GB")
            save_json(task_info, f"{save_path}/task_info.json")
