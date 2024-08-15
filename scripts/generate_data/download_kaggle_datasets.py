# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

import os
import functools
import signal
import pandas as pd
import argparse
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import kaggle
kaggle.api.authenticate()


def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func
    return decorator

@timeout(120)
def download_dataset(username, dataset, save_dir):
    kaggle.api.dataset_download_files(f'{username}/{dataset}', unzip=True, path=save_dir)

def is_dataset_exists(dataset_dir):
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.find(".csv") >= 0:
                return True
    return False

def load_dataset_to_df(dataset_dir):
    """
        Load all valid csv files in the dataset_dir and merge them into one dataframe.
    """
    features = []
    df_list = []
    sep_options = [',', ';']
    load_files = []
    try:
        for root, dirs, files in os.walk(dataset_dir):
            for f in files:
                # try to load each file in the directory
                try:
                    if f.find(".csv") >= 0:
                        # judge sep and index_col
                        path = os.path.join(root, f)
                        sep = ','
                        is_first_col_index = True
                        with open(path, 'r') as fp:
                            cols_content = fp.readline().strip('\n')
                            # find seperator
                            for s in sep_options:
                                cols = cols_content.split(s)
                                if len(cols) > 1:
                                    sep = s
                                    break
                            if len(cols[0]) > 0:
                                is_first_col_index = False
                        # load csv
                        df = pd.read_csv(path, index_col=0 if is_first_col_index else None, sep=sep)
                        f_cols = df.columns
                        if len(f_cols) > len(features):
                            # use data with maximum features
                            print(f"Update data from {f} with num_features={len(f_cols)}")
                            features = f_cols
                            df_list = [df]
                            load_files = [f]
                        elif len(f_cols) == len(features):
                            if len(df) == len(df_list[-1]) and dataset_dir.split('/')[-1].lower() not in ['train', 'val', 'validation', 'test']:
                                # maybe duplicated
                                print(f"Ignore duplicated data of {f}")
                                continue
                            else:
                                # different data count or train/val/test, merge all
                                print(f"Add data of {f}")
                                df_list.append(df)
                                load_files.append(f)
                except Exception as err:
                    print(f"[Error] Fail to load data csv in {f}, error={err}")
                    continue
        df_all = pd.concat(df_list)
    except Exception as err:
        df_all = None
        print(f"[Error] Fail to load data csv in {dataset_dir}, error={err}")
    return df_all, load_files

def main():
    parser = argparse.ArgumentParser(description='Download Kaggle Datasets')
    parser.add_argument('--url_path', type=str)
    parser.add_argument('--download_dir', type=str)
    parser.add_argument('--raw_data_save_dir', type=str)
    
    args = parser.parse_args()
    url_path = args.url_path
    download_dir = args.download_dir
    raw_data_save_dir = args.raw_data_save_dir
    
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    if not os.path.exists(raw_data_save_dir):
        os.mkdir(raw_data_save_dir)
    
    # Load url list
    url_list = []
    with open(url_path, 'r') as f:
        for l in f.readlines():
            url_list.append(l.strip())
    
    pbar = tqdm(url_list)
    success_cnt, fail_cnt, download_cnt = 0, 0, 0
    for url in pbar:
        try:
            username, dataset = url.split('/')[-2], url.split('/')[-1]
            
            download_dataset_dir = f'{download_dir}/{dataset}'
            if not os.path.exists(download_dataset_dir):
                os.mkdir(download_dataset_dir)
                
            # Download dataset
            if not is_dataset_exists(download_dataset_dir):
                pbar.set_description(f"Download {dataset}")
                download_dataset(username, dataset, download_dataset_dir)
                download_cnt += 1
            else:
                print(f"Dataset {dataset} already exists, skip download.")
            
            # Load dataset
            save_path = os.path.join(raw_data_save_dir, f'{dataset}.csv')
            if not os.path.exists(save_path):
                df, load_files = load_dataset_to_df(download_dataset_dir)
                if df is not None:
                    df.to_csv(save_path)
                    success_cnt += 1
                else:
                    fail_cnt += 1
            else:
                print(f"Raw data {dataset} already exists, skip load.")
        
        except Exception as err:
            print(f'Download dataset {dataset} failed, check the url {url}, err={err}')
            fail_cnt += 1
        
        pbar.set_postfix({
            'download_cnt': download_cnt,
            'success': success_cnt,
            'fail': fail_cnt,
            'success_rate': float(success_cnt / max(1, success_cnt + fail_cnt))
        })

if __name__ == '__main__':
    main()