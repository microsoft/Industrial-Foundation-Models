#! /bin/bash

LLAMA_MODEL_PATH=$1
CKPT_SAVE_DIR=$2
MODEL_SIZE=$3

if [ $MODEL_SIZE == "7B" ]; then
    MODEL_NAME=LLaMA-2-7b-GTL-Delta
elif [ $MODEL_SIZE == "13B" ]; then
    MODEL_NAME=LLaMA-2-13b-GTL-Delta
else
    echo "Invalid model size"
    exit 1
fi

# Download LLaMA-2-GTL checkpoint difference
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --local-dir $CKPT_SAVE_DIR/$MODEL_NAME microsoft/$MODEL_NAME

# Recover LLaMA-2-GTL from raw LLaMA-2 and checkpoint difference
python scripts/manage_ckpt/recover_hf_model_from_weights_diff.py \
    --hf_model_path $LLAMA_MODEL_PATH \
    --weight_diff_path $CKPT_SAVE_DIR/$MODEL_NAME \
    --recover_model_save_path $CKPT_SAVE_DIR/LLaMA-2-GTL/$MODEL_SIZE