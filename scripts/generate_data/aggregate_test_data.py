# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

import os
import json
import argparse
import numpy as np
import datasets
import traceback
from datasets import load_from_disk, concatenate_datasets, DatasetDict
datasets.disable_caching()


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate Data')
    parser.add_argument('--aggregate_config', type=str)
    
    # Load config
    args = parser.parse_args()
    config = load_json(args.aggregate_config)
    source_dir = config['source_dir']
    source_data_list = config['source_data_list']
    target_dir = config['target_dir']
    data_type = config['data_type']
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    # Process
    output_data_dict = {t: [] for t in data_type}
    output_task_info = {}
    task_index_offset = 0
    for d in source_data_list:
        print(f"Processing database: {d}")
        try:
            data_path = f'{source_dir}/{d}'
            task_info = load_json(f"{data_path}/task_info.json")
            max_task_index = max([int(k) for k in task_info.keys()]) + 1
            # Update task info with new task index
            for k, v in task_info.items():
                new_task_idx = int(k) + task_index_offset
                output_task_info[new_task_idx] = v
            # Update task index in data
            for t in data_type:
                data = load_from_disk(f'{data_path}/{t}')
                new_task_idx = (np.array(data['task_idx']) + task_index_offset).tolist()
                data = data.remove_columns(["task_idx"])
                data = data.add_column(name='task_idx', column=new_task_idx)
                output_data_dict[t].append(data)
        except Exception as err:
            tb = traceback.format_exc()
            print(f"Fail to load {d}, err={err}\n{tb}")
        task_index_offset += max_task_index
    # Merge databases
    save_json(output_task_info, os.path.join(target_dir, 'task_info.json'))
    for t in data_type:
        output_data_dict[t] = concatenate_datasets(output_data_dict[t])
    merged_data_dict = DatasetDict(output_data_dict)
    merged_data_dict.save_to_disk(target_dir, max_shard_size="3GB")
    print(f"Finish aggregating data to {target_dir}")
