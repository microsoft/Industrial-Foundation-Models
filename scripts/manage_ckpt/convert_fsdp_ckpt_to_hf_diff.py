from copy import deepcopy
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    load_state_dict
)

def load_auto_model(path):
    model = AutoModelForCausalLM.from_pretrained(path)
    return model

def compute_weight_diff(state_dict1, state_dict2):
    weight_diff = {}
    for key in state_dict1.keys():
        if key in state_dict2:
            weight_diff[key] = state_dict2[key] - state_dict1[key]
        else:
            raise KeyError(f"Key {key} not found in both state dictionaries")
    return weight_diff

def load_sharded_model_single_gpu(model, path, prefix="causal_lm."):
    # Load checkpoint into the state dict with prefix
    state_dict = {"model": model.state_dict()}
    new_state_dict = {"model": {}}
    for key, value in state_dict["model"].items():
        new_state_dict["model"][f'{prefix}{key}'] = value
    
    load_state_dict(
        state_dict=new_state_dict,
        storage_reader= FileSystemReader(path),
        no_dist=True,
    )
    
    # Remove the prefix from the state dict
    for key, value in new_state_dict["model"].items():
        if key.startswith(prefix):
            ori_key = key[len(prefix):]
            state_dict["model"][ori_key] = value
    
    model.load_state_dict(state_dict["model"])
    print(f"Sharded state checkpoint loaded from {path}")
    return model

def main():
    parser = argparse.ArgumentParser(description='Tabular checkpoint')
    parser.add_argument('--hf_model_path', type=str)
    parser.add_argument('--fsdp_ckpt_path', type=str)
    parser.add_argument('--hf_ckpt_save_path', type=str)
    parser.add_argument('--save_ckpt_diff', type=bool, default=False)
    parser.add_argument('--hf_ckpt_diff_save_path', type=str)
    
    args = parser.parse_args()
    hf_model_path = args.hf_model_path
    fsdp_ckpt_path = args.fsdp_ckpt_path
    hf_ckpt_save_path = args.hf_ckpt_save_path
    hf_ckpt_diff_save_path = args.hf_ckpt_diff_save_path
    
    # Load the original HF model
    model = load_auto_model(hf_model_path)
    
    # Load the FSDP sharded checkpoints into the model
    tmp_model = deepcopy(model)
    hf_ckpt = load_sharded_model_single_gpu(tmp_model, fsdp_ckpt_path)
    hf_ckpt.save_pretrained(hf_ckpt_save_path)
    print(f"HuggingFace model checkpoints has been saved in {hf_ckpt_save_path}")
    
    # Loading the tokenizer form the model_path and save to the new path
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    tokenizer.save_pretrained(hf_ckpt_save_path)
    
    if args.save_ckpt_diff:
        # Compute and save weights difference
        weights_diff = compute_weight_diff(model.state_dict(), hf_ckpt.state_dict())
        hf_ckpt_diff = deepcopy(model)
        hf_ckpt_diff.load_state_dict(weights_diff)
        hf_ckpt_diff.save_pretrained(hf_ckpt_diff_save_path)
        print(f"HuggingFace model checkpoints weights difference has been saved in {hf_ckpt_diff_save_path}")

        tokenizer.save_pretrained(hf_ckpt_diff_save_path)

if __name__ == "__main__":
    main()