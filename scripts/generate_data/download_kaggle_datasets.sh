#! /bin/bash

DATA_VERSION=$1

if [ $DATA_VERSION == "demo" ]; then
    URL_PATH=data/datasets_url/holdout_demo_datasets
elif [ $DATA_VERSION == "holdout" ]; then
    URL_PATH=data/datasets_url/holdout_kaggle_datasets
elif [ $DATA_VERSION == "all" ]; then
    URL_PATH=data/datasets_url/all_kaggle_datasets
else
    echo "Invalid model size"
    exit 1
fi

DOWNLOAD_DIR=data/download
RAW_DATA_SAVE_DIR=data/raw_data

python scripts/generate_data/download_kaggle_datasets.py --url_path $URL_PATH --download_dir $DOWNLOAD_DIR --raw_data_save_dir $RAW_DATA_SAVE_DIR