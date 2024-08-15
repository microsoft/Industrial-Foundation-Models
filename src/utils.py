# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

# ---------------------------------------------------------------------------------
# This file contains some parts inspired by the llama-recipes library.
# - Source: https://github.com/meta-llama/llama-recipes

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import os
import json
import torch
import pickle

from pathlib import Path
from typing import Any, List
from collections.abc import Mapping

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType
)
from torch.distributed.checkpoint.optimizer import (
    load_sharded_optimizer_state_dict
)
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    DefaultLoadPlanner,
    DefaultSavePlanner,    
    save_state_dict,
    load_state_dict
)


def thread_safe_log(text):
    if 'LOCAL_RANK' in os.environ:
        if os.environ['LOCAL_RANK'] == '0':
            print(text, flush=True)
    else:
        print(text, flush=True)


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def load_datasets_from_config(config_path):
    datasets = []
    with open(config_path, 'r') as f:
        for l in f.readlines():
            datasets.append(l.strip().replace('\n', ''))
    return datasets


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_success(path, filename="success"):
    fp = os.path.join(path, filename)
    with open(fp, 'w') as f:
        f.write("Job finished.")


def save_model_and_optimizer_sharded(model, rank, path, optimizer=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    save_dir = Path.cwd() / path
    print(f"Saving model to {save_dir}")
    
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        
        state_dict = {"model": model.state_dict()}
        if optimizer is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optimizer)

        save_state_dict(
            state_dict=state_dict,
            storage_writer=FileSystemWriter(save_dir),
            planner=DefaultSavePlanner()
        )
    dist.barrier()
    print(f"Sharded state checkpoint saved to {save_dir}")
        
        
def load_model_and_optimizer_sharded(model, rank, path, optimizer=None):
    """load model and optimizer via sharded_state_dict from load_dir"""
    load_dir = Path.cwd() / path
    if not load_dir.exists():
        raise ValueError(f"No sharded_state_dict checkpoint directory found...")
    print(f"Loading model from {load_dir}")

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {"model": model.state_dict()}

        load_state_dict(
            state_dict=state_dict,
            storage_reader=FileSystemReader(load_dir),
            planner=DefaultLoadPlanner(),
        )
        model.load_state_dict(state_dict["model"])
    
        # Load optimizer
        if optimizer is not None:
            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=FileSystemReader(load_dir)
            )
            flattened_optimizer_state = FSDP.optim_state_dict_to_load(
                model, optimizer, optim_state["optim"]
            )
            optimizer.load_state_dict(flattened_optimizer_state)
    print(f"Sharded state checkpoint loaded from {load_dir}")


def clip_grad_norm(model, optimizer, max_grad_norm):
    if hasattr(optimizer, "clip_grad_norm"):
        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
        optimizer.clip_grad_norm(max_grad_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def distributed_concat_and_move_to_cpu(tensor: Any, valid_keys: List[str] = None) -> Any:
    """
        Reference from huggingface transformers.
        https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L189
    """
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(distributed_concat_and_move_to_cpu(t) for t in tensor)
        if isinstance(tensor, Mapping):
            valid_outputs = {}
            for k, t in tensor.items():
                if (valid_keys is None or k in valid_keys) and t is not None:
                    valid_outputs[k] = distributed_concat_and_move_to_cpu(t)
            return valid_outputs
        tensor = torch.atleast_1d(tensor).contiguous()
        output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0).detach().cpu().to(dtype=torch.float32)
    
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def batch_move_to_cpu(tensor: Any, valid_keys: List[str] = None) -> Any:
    """
        Reference from huggingface transformers.
        https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L189
    """
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(batch_move_to_cpu(t) for t in tensor)
        if isinstance(tensor, Mapping):
            valid_outputs = {}
            for k, t in tensor.items():
                if (valid_keys is None or k in valid_keys) and t is not None:
                    valid_outputs[k] = batch_move_to_cpu(t)
            return valid_outputs

        return tensor.detach().cpu().to(dtype=torch.float32)
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def gather_tensor(tensor: List[Any], limit: int = None) -> Any:
    l = len(tensor)
    if isinstance(tensor[0], (tuple, list)):
        return type(tensor[0])(gather_tensor([t[i] for t in tensor], limit) for i in range(l))
    if isinstance(tensor[0], Mapping):
        return {k: gather_tensor([t[k] for t in tensor], limit) for k in tensor[0].keys()}
    if l == 0 or tensor[0] is None:
        return None
    if tensor[0].dim() == 0:
        concat = torch.stack(tensor, dim=0)
    else:
        concat = torch.cat(tensor, dim=0)
    if limit is not None:
        concat = concat[:limit]

    return concat


def extract_masked_continuous_values(value_tensor, mask_tensor, padding_value=0):  
    """
        Extract the continuous values from value_tensor using the mask_tensor.
        Transform each group of continuous values to a sample and pad all the samples to the same length.
        For example, given the following mask_tensor and value_tensor:
            value_tensor = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
            mask_tensor = [[1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
        The output will be:
            output = [[1, 2, 3, 0],
                      [6, 7, 0, 0],
                      [11, 12, 13, 14]]
        
        Args:
            value_tensor: (batch_size, seq_len) or (batch_size, seq_len, value_dim)
            mask_tensor: (batch_size, seq_len)
            padding_value: the value to pad the samples
        
        Returns:
            output: (group_size, max_group_length) or (group_size, max_group_length, value_dim)
    """
    # Clip the value_tensor and mask_tensor to the same length
    min_len = min(value_tensor.shape[1], mask_tensor.shape[1])
    value_tensor = value_tensor[:, :min_len]
    mask_tensor = mask_tensor[:, :min_len]
    
    # Find the group boundaries using torch.diff()  
    pad_tensor = torch.zeros_like(mask_tensor[:, :1])
    boundary_indices = torch.where(torch.diff(mask_tensor, prepend=pad_tensor, append=pad_tensor, dim=-1))  
  
    # Split the boundary_indices into start_indices and end_indices  
    start_indices = boundary_indices[1][::2]  
    end_indices = boundary_indices[1][1::2]  
    row_indices = boundary_indices[0][::2]  
  
    # Compute group lengths  
    group_lengths = end_indices - start_indices  
  
    # Compute the indices for each group using arange and broadcasting  
    group_indices = torch.arange(group_lengths.max(), device=mask_tensor.device) + start_indices[:, None]  
  
    # Clip the group indices to be within the valid range  
    clipped_indices = torch.clamp(group_indices, min=0, max=value_tensor.shape[1] - 1)  
    
    # Gather the values using the clipped indices and apply the mask to zero out invalid positions  
    is_valid_values = group_indices < end_indices[:, None]
    if len(value_tensor.shape) == 2:
        gathered_values = torch.gather(value_tensor[row_indices], dim=1, index=clipped_indices)
        gathered_values = torch.where(  
            is_valid_values,  
            gathered_values,  
            torch.ones_like(gathered_values, dtype=gathered_values.dtype) * padding_value  
        )  
    elif len(value_tensor.shape) == 3:
        gathered_values = torch.gather(value_tensor[row_indices], dim=1, index=clipped_indices[..., None].expand(-1, -1, value_tensor.shape[-1]))
        gathered_values = torch.where(  
            is_valid_values.unsqueeze(-1),
            gathered_values,  
            torch.ones_like(gathered_values, dtype=gathered_values.dtype) * padding_value  
        )
    else:
        raise ValueError(f"Unsupported value_tensor shape: {value_tensor.shape}")
  
    return gathered_values 


def shift_unmasked_values(value_tensor, tail_masked_tensor, padding_value=0, maintain_masked_values=False):
    """
        Shift the unmasked values in value_tensor to the right for alignment by the number of masked values in tail_masked_tensor.
        
        Args:
            value_tensor: (batch_size, seq_len)
            tail_masked_tensor: (batch_size, seq_len)
            padding_value: the value to pad new positions
            maintain_masked_values: whether to maintain the masked values in the shifted tensor
                if True, the shifted tensor will be extend to the the right with the maximum number of masked values
                if False, the shifted tensor will be truncated to the same length as the original tensor
        
        Returns:
            shifted_tensor: (batch_size, seq_len)
    """
    # Clip the value_tensor and mask_tensor to the same length
    min_len = min(value_tensor.shape[1], tail_masked_tensor.shape[1])
    value_tensor = value_tensor[:, :min_len]
    tail_masked_tensor = tail_masked_tensor[:, :min_len]
    
    # Calculate the number of masked values along each row
    num_masked_values = tail_masked_tensor.sum(dim=1)
    max_shift_length = num_masked_values.max()
    
    if maintain_masked_values:
        max_len = value_tensor.shape[1] + max_shift_length
    else:
        max_len = value_tensor.shape[1]
    
    # Create a range tensor to represent each index along the sequence dimension
    index_range = torch.arange(max_len).repeat(value_tensor.shape[0], 1).to(value_tensor.device)
  
    # Subtract the number of masked values from the index range tensor
    shifted_indices = index_range - num_masked_values.view(-1, 1)
  
    # Create a mask for valid indices after shifting
    valid_indices_mask = (shifted_indices >= 0) & (shifted_indices < value_tensor.shape[1])
  
    # Gather the shifted values using the valid indices mask and shifted_indices
    shifted_tensor = torch.where(
        valid_indices_mask,
        torch.gather(value_tensor, 1, shifted_indices * valid_indices_mask.long()),
        torch.ones_like(index_range) * padding_value
    )

    return shifted_tensor