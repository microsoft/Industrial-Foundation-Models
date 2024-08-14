LLAMA_MODEL_PATH=$1
DATABASE_PATH=$2
OUTPUT_PATH=$3
NUM_GPUS=$4

torchrun --nnodes 1 --nproc_per_node $NUM_GPUS pipeline.py \
      --tabfm_stage pretrain --model_path $LLAMA_MODEL_PATH \
      --reload_data True --reload_directly True \
      --train_database $DATABASE_PATH/pretrain \
      --num_epochs 1 --max_train_steps 8192 --batch_size 64 --micro_batch_size 4 --learning_rate 1e-5 --early_stopping_patience 3 \
      --val_database $DATABASE_PATH/holdout_cls,$DATABASE_PATH/holdout_reg \
      --val_batch_size 8,8 --val_max_generate_length 2,10 --val_before_training False --val_every_n_steps 2048 \
      --output_path $OUTPUT_PATH