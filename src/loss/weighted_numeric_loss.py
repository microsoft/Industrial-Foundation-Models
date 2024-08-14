import torch
import torch.nn as nn
import numpy as np
from typing import Set, Union, Tuple, Optional, List


class WeightedNumericTokenLoss(nn.Module):
    def __init__(
        self,
        numerical_token_ids: List[int],
        decay_method: str = 'exp',
        decay_rate: float = 0.5,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.base_ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.base_weight = 1.0
        self.decay_method = decay_method
        self.decay_rate = decay_rate
        self.numerical_token_ids = numerical_token_ids
        self.pad_token_id = pad_token_id
    
    def assign_weights_for_target_values(
        self,
        weights,
        label_token_ids,
        target_values: List[int],
        target_weights: Union[float, torch.FloatTensor] = 0.5,
        mode: str = 'in'
    ):  
        """
            Args:
                weights: (batch_size, seq_len)
                label_token_ids: (batch_size, seq_len)
                target_values: a list of target values
                target_weight: the weight of the target values
                mode: 'in' or 'out', indicating whether to assign the target_weight to the target values or the non-target values
            
            Returns:
                weights: (batch_size, seq_len)
        """
        # Check if the elements in the tensor belong to the target_values list
        target_values_tensor = torch.tensor(target_values, device=label_token_ids.device).view(1, 1, -1) 
        is_target_value = (label_token_ids.unsqueeze(-1) == target_values_tensor).any(dim=-1)  
    
        if mode == 'in':
            target_weights = torch.tensor(target_weights, dtype=torch.float32, device=label_token_ids.device)
            non_target_weights = weights
        elif mode == 'out':
            target_weights = weights
            non_target_weights = torch.tensor(target_weights, dtype=torch.float32, device=label_token_ids.device)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if isinstance(target_weights, float) or (isinstance(target_weights, torch.Tensor) and target_weights.numel() == 1):
            # Assign a weight to all target value tokens
            weights = torch.where(is_target_value, target_weights, non_target_weights)
        else:
            # Assign a group of weights maintaining the same order
            weights = torch.repeat_interleave(target_weights.unsqueeze(0), repeats=label_token_ids.shape[0], dim=0)
            
            # Compute the cumulative sum of the mask tensor along the last dimension
            target_cumsum = is_target_value.cumsum(dim=-1)
            target_weight_indices = target_cumsum - 1
            
            # Mask the indices tensor to only keep the indices where mask_tensor is True
            target_weight_indices = torch.where(is_target_value, target_weight_indices, torch.zeros_like(target_weight_indices))
             
            # Gather the values from the weight tensor using the masked_indices
            weights = torch.gather(weights, dim=-1, index=target_weight_indices)
    
        return weights 
    
    def forward(self, logits, label_token_ids):
        """
            Args:
                logits: (batch_size, seq_len, vocab_size)
                label_token_ids: (batch_size, seq_len)
            
            Returns:
                total_loss: the summation of the weighted loss of all numerical tokens.
        """
        batch_size, seq_len = label_token_ids.shape
        
        # TODO: support the expression of scientific notation
        if self.decay_method == 'exp': 
            decayed_weights = torch.logspace(
                start=np.log10(self.base_weight),
                end=np.log10(self.base_weight * self.decay_rate ** (seq_len - 1)),
                steps=seq_len
            )
        elif self.decay_method == 'linear':
            decayed_weights = torch.linspace(
                start=self.base_weight,
                end=self.base_weight - self.decay_rate * (seq_len - 1),
                steps=seq_len
            )
        else:
            raise ValueError(f"Unknown decay method: {self.decay_method}")
        
        # Assign the decayed weights to the numerical tokens
        weights = self.assign_weights_for_target_values(
            weights=torch.ones_like(label_token_ids, dtype=torch.float32),
            label_token_ids=label_token_ids,
            target_values=self.numerical_token_ids,
            target_weights=decayed_weights,
            mode='in'
        )
          
        # Assign a fixed weight to the non-numerical tokens
        # weights = self.assign_weights_for_target_values(
        #     weights, label_token_ids, self.numerical_token_ids, target_weights=self.base_weight, mode='out'
        # )
        
        # Assign zero weight to the padding tokens
        weights = self.assign_weights_for_target_values(
            weights, label_token_ids, [self.pad_token_id], target_weights=0.0, mode='in'
        )
  
        # Compute the CrossEntropyLoss  
        losses = self.base_ce_loss(logits.view(-1, logits.size(-1)), label_token_ids.view(-1))  
        losses = losses.view(*label_token_ids.size())
  
        # Apply the loss weights  
        weighted_losses = losses * weights  
        total_loss = torch.sum(weighted_losses) / torch.sum(weights)  
  
        return total_loss