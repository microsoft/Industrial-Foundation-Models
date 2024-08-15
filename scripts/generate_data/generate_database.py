# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

import os
import time
import shlex
import subprocess
import argparse
import numpy as np
from itertools import product
from datetime import timedelta


def run(cmds, cuda_id):
    _cur = 0

    def recycle_devices():
        for cid in cuda_id:
            if cuda_id[cid] is not None:
                proc = cuda_id[cid]
                if proc.poll() is not None:
                    cuda_id[cid] = None

    def available_device_id():
        for cid in cuda_id:
            if cuda_id[cid] is None:
                return cid

    def submit(cmd, cid):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = cid

        args = shlex.split(cmd)
        exp_dir = args[-1]
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)
        log = open('{}/log.txt'.format(exp_dir), 'w')
        print(time.asctime(), ' '.join(args))

        proc = subprocess.Popen(args, env=env, stdout=log, stderr=log)

        cuda_id[cid] = proc

    while _cur < len(cmds):
        recycle_devices()
        cid = available_device_id()

        if cid is not None:
            print(f'CUDA {cid} available for job ({_cur+1} of {len(cmds)})')
            submit(cmds[_cur], cid)
            _cur += 1

        time.sleep(1)
    
    while any([v is not None for v in cuda_id.values()]):
        recycle_devices()
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Tabular Data')
    parser.add_argument('--cuda_ids', type=str, default='0-1-2-3')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_data_path', type=str)
    parser.add_argument('--root_exp_dir', type=str, default='exps')
    parser.add_argument('--data_path', type=str, default='data/raw_data')
    parser.add_argument('--meta_info_path', type=str, default='data/meta_info')
    parser.add_argument('--database_config_path', type=str, default='data/database_config')
    parser.add_argument('--database', type=str)
    parser.add_argument('--project_name', type=str, default='GenerateData')
    parser.add_argument('--load_synthetic', type=bool, default=False)
    parser.add_argument('--reload_data', type=bool, default=False)
    parser.add_argument('--save_single_dataset', type=bool, default=False)
    parser.add_argument('--test_sample_seed', type=int, default=0)
    parser.add_argument('--num_shots_train', type=int, default=64)
    parser.add_argument('--num_shots_test', type=int, default=64)
    parser.add_argument('--specified_label_idx_options', type=str, default='0')
    parser.add_argument('--in_context_count_options', type=str, default='0,4,8,16,32,64')
    parser.add_argument('--context_sample_mode_options', type=str, default='global')
    parser.add_argument('--global_context_groups', type=int, default=1)
    parser.add_argument('--context_sample_seed_options', type=str, default='0')
    parser.add_argument('--template_options', type=str, default='table,language')
    parser.add_argument('--cutoff_len', type=int, default=4096)

    args = parser.parse_args()
    cuda_dict = dict([(str(i), None) for i in args.cuda_ids.split('-')])
    model_path = args.model_path
    save_data_path = args.save_data_path
    root_exp_dir = args.root_exp_dir
    data_path = args.data_path
    meta_info_path = args.meta_info_path
    database_config_path = args.database_config_path
    database = args.database
    project_name = args.project_name
    specified_label_idx_options = args.specified_label_idx_options.split(',')
    in_context_count_options = args.in_context_count_options.split(',')
    context_sample_mode_options = args.context_sample_mode_options.split(',')
    global_context_groups = args.global_context_groups
    context_sample_seed_options = args.context_sample_seed_options.split(',')
    template_options = args.template_options.split(',')
    load_synthetic = args.load_synthetic
    reload_data = args.reload_data
    reload_data_path = args.save_data_path
    save_single_dataset = args.save_single_dataset
    
    test_sample_seed = args.test_sample_seed
    num_shots_train = args.num_shots_train
    num_shots_test = args.num_shots_test
    cutoff_len = args.cutoff_len

    cmds = []
    idx = 0
    for specified_label_idx, in_context_count, context_sample_mode, context_sample_seed, template in \
        product(specified_label_idx_options, in_context_count_options, context_sample_mode_options, context_sample_seed_options, template_options):
        output_path = os.path.join(
            root_exp_dir,
            project_name,
            f'generate_{database.split("/")[-1]}_ds{test_sample_seed}_shots_{num_shots_train}_{num_shots_test}_lab_{specified_label_idx}_context_{in_context_count}_{context_sample_mode}_{global_context_groups}_{context_sample_seed}_{template}_{cutoff_len}'
        )
        # Skip zero-shot with anonymous template 
        if int(in_context_count) == 0:
            if int(context_sample_seed) != 0 or template == 'anonymous_table':
                continue
        # Random select a port
        master_port = np.random.randint(15333, 23333)
        items = [
            'torchrun', '--nnodes 1', '--nproc_per_node 1', f'--master_port {master_port}',
            'pipeline.py',
            # Generate Data
            '--tabfm_stage generate_data',
            # Tokenizer
            f'--model_path {model_path}',
            # Task related
            f'--test_sample_seed {test_sample_seed}',
            f'--num_shots_train {num_shots_train}',
            f'--num_shots_test {num_shots_test}',
            f'--cutoff_len {cutoff_len}',
            f'--specified_label_idx {specified_label_idx}',
            f'--in_context_count {in_context_count}',
            f'--context_sample_mode {context_sample_mode}',
            f'--global_context_groups {global_context_groups}',
            f'--context_sample_seed {context_sample_seed}',
            f'--template {template}',
            f'--data_path {data_path}',
            f'--meta_info_path {meta_info_path}',
            f'--database_config_path {database_config_path}',
            f'--database {database}',
            f'--load_synthetic {load_synthetic}',
            f'--reload_data {reload_data}',
            f'--reload_data_path {reload_data_path}',
            f'--save_single_dataset {save_single_dataset}',
            f'--save_data True',
            f'--save_data_path {save_data_path}',
            # IMPORTANT: always put the output path at last,
            # so that we can set proper log file paths in submit()
            f'--output_path {output_path}',
        ]
        cmd = ' '.join(items)
        # We must set exp_dir at last to set proper log file paths
        cmds.append(cmd)
        idx += 1

    start = time.time()
    run(cmds, cuda_dict)
    end = time.time()
    print(f'Total elapsed time: {str(timedelta(seconds=end-start))}')