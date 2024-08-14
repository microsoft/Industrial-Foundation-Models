CHECKPOINT_PATH=$1
DATABASE_PATH=$2
OUTPUT_PATH=$3
NUM_GPUS=$4

torchrun --nnodes 1 --nproc_per_node $NUM_GPUS pipeline.py \
      --tabfm_stage eval --model_path $CHECKPOINT_PATH \
      --reload_data True --reload_directly True \
      --test_database $DATABASE_PATH/holdout_cls,$DATABASE_PATH/holdout_reg \
      --test_batch_size 8,8 --test_max_generate_length 2,10 \
      --output_path $OUTPUT_PATH