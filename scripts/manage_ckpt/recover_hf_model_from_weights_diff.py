# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_auto_model(path):
    model = AutoModelForCausalLM.from_pretrained(path)
    return model

def recover_from_weight_diff(state_dict1, state_dict2):
    recover_weights = {}
    for key in state_dict1.keys():
        if key in state_dict2:
            recover_weights[key] = state_dict2[key] + state_dict1[key]
        else:
            raise KeyError(f"Key {key} not found in both state dictionaries")
    return recover_weights

def main():
    parser = argparse.ArgumentParser(description='Tabular checkpoint')
    parser.add_argument('--hf_model_path', type=str)
    parser.add_argument('--weight_diff_path', type=str)
    parser.add_argument('--recover_model_save_path', type=str)
    
    args = parser.parse_args()
    hf_model_path = args.hf_model_path
    weight_diff_path = args.weight_diff_path
    recover_model_save_path = args.recover_model_save_path
    
    # Load the original Transformers model
    model = load_auto_model(hf_model_path)

    # Load the weights difference
    weights_diff = load_auto_model(weight_diff_path)

    # Compute the fine-tuned model weights
    recover_weights = recover_from_weight_diff(model.state_dict(), weights_diff.state_dict())

    # Load the fine-tuned model with the recovered weights
    model.load_state_dict(recover_weights)
    model.save_pretrained(recover_model_save_path)
    
    # Loading the tokenizer form the model_path and save to the new path
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    tokenizer.save_pretrained(recover_model_save_path)

if __name__ == "__main__":
    main()