#! /bin/bash

MODEL_PATH=$1
AGG_DATABASE_PATH=$2
CUTOFF_LEN=4096
SAVE_DATABASE_PATH=data/processed_database/llama2_4k
AGG_CONFIG_DIR=data/aggregate_config/llama2_4k
PROJECT_NAME=GenerateData-llama2-4k
DATABASE_EVAL_CLS=holdout_cls
DATABASE_EVAL_REG=holdout_reg

# (Option 1) If you want to reproduce the in-context learning results, you only need to generate the data with table template 
# python scripts/generate_data/generate_database.py \
#     --model_path $MODEL_PATH --cutoff_len $CUTOFF_LEN \
#     --save_data_path $SAVE_DATABASE_PATH --database $DATABASE_EVAL_CLS \
#     --project_name $PROJECT_NAME \
#     --num_shots_train 64 --num_shots_test 64 --context_sample_seed_options 0,1,2 \
#     --test_sample_seed 0 --template_options table \
#     --in_context_count_options 4,8,16,32,64

# python scripts/generate_data/generate_database.py \
#     --model_path $MODEL_PATH --cutoff_len $CUTOFF_LEN \
#     --save_data_path $SAVE_DATABASE_PATH --database $DATABASE_EVAL_REG \
#     --project_name $PROJECT_NAME \
#     --num_shots_train 64 --num_shots_test 64 --context_sample_seed_options 0,1,2 \
#     --test_sample_seed 0 --template_options table \
#     --in_context_count_options 4,8,16,32,64

# (Option 2) Genearate holdout data for all settings
python scripts/generate_data/generate_database.py \
    --model_path $MODEL_PATH --cutoff_len $CUTOFF_LEN \
    --save_data_path $SAVE_DATABASE_PATH --database $DATABASE_EVAL_CLS \
    --project_name $PROJECT_NAME \
    --num_shots_train 64 --num_shots_test 64 --context_sample_seed_options 0,1,2 \
    --test_sample_seed 0 --template_options table,language,anonymous_table \
    --in_context_count_options 0,4,8,16,32,64

python scripts/generate_data/generate_database.py \
    --model_path $MODEL_PATH --cutoff_len $CUTOFF_LEN \
    --save_data_path $SAVE_DATABASE_PATH --database $DATABASE_EVAL_REG \
    --project_name $PROJECT_NAME \
    --num_shots_train 64 --num_shots_test 16 --context_sample_seed_options 0,1,2 \
    --test_sample_seed 0 --template_options table,language,anonymous_table \
    --in_context_count_options 0,4,8,16,32,64

# Aggregate all settings processed data to one database
# holdout cls
python scripts/generate_data/gen_aggregate_config.py \
    --source_dir $SAVE_DATABASE_PATH --source_database_names $DATABASE_EVAL_CLS \
    --target_dir $AGG_DATABASE_PATH/$DATABASE_EVAL_CLS --data_type test \
    --agg_config_dir $AGG_CONFIG_DIR --agg_config_name $DATABASE_EVAL_CLS

python scripts/generate_data/aggregate_test_data.py \
    --aggregate_config $AGG_CONFIG_DIR/$DATABASE_EVAL_CLS.json

# holdout reg
python scripts/generate_data/gen_aggregate_config.py \
    --source_dir $SAVE_DATABASE_PATH --source_database_names $DATABASE_EVAL_REG \
    --target_dir $AGG_DATABASE_PATH/$DATABASE_EVAL_REG --data_type test \
    --agg_config_dir $AGG_CONFIG_DIR --agg_config_name $DATABASE_EVAL_REG

python scripts/generate_data/aggregate_test_data.py \
    --aggregate_config $AGG_CONFIG_DIR/$DATABASE_EVAL_REG.json