#! /bin/bash

MODEL_PATH=$1
AGG_DATABASE_PATH=$2
CUTOFF_LEN=4096
SAVE_DATABASE_PATH=data/processed_database/llama2_4k
AGG_CONFIG_DIR=data/aggregate_config/llama2_4k
PROJECT_NAME=GenerateData-llama2-4k
DATABASE_TRAIN_CLS=pretrain_cls
DATABASE_TRAIN_REG=pretrain_reg
TARGET_DATABASE=pretrain

# Genearate data for several settings
python scripts/generate_data/generate_database.py \
    --model_path $MODEL_PATH --cutoff_len $CUTOFF_LEN \
    --save_data_path $SAVE_DATABASE_PATH --database $DATABASE_TRAIN_CLS \
    --project_name $PROJECT_NAME \
    --specified_label_idx_options 0,1,2,3 --num_shots_train 64 --num_shots_test 16 --context_sample_seed_options 0 \
    --test_sample_seed 0 --template_options table,language,anonymous_table \
    --in_context_count_options 0,4,8,16,32,64

python scripts/generate_data/generate_database.py \
    --model_path $MODEL_PATH --cutoff_len $CUTOFF_LEN \
    --save_data_path $SAVE_DATABASE_PATH --database $DATABASE_TRAIN_REG \
    --project_name $PROJECT_NAME \
    --specified_label_idx_options 0,1,2,3 --num_shots_train 64 --num_shots_test 4 --context_sample_seed_options 0 \
    --test_sample_seed 0 --template_options table,language,anonymous_table \
    --in_context_count_options 0,4,8,16,32,64

# Aggregate all settings processed data to one database
python scripts/generate_data/gen_aggregate_config.py \
    --source_dir $SAVE_DATABASE_PATH --source_database_names $DATABASE_TRAIN_CLS,$DATABASE_TRAIN_REG \
    --target_dir $AGG_DATABASE_PATH/$TARGET_DATABASE --data_type train \
    --agg_config_dir $AGG_CONFIG_DIR --agg_config_name $TARGET_DATABASE

python scripts/generate_data/aggregate_test_data.py \
    --aggregate_config $AGG_CONFIG_DIR/$TARGET_DATABASE.json
