# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

import math
from typing import TypeVar, Optional, Iterator

import torch
import torch.distributed as dist

__all__ = ["DistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)
from torch.utils.data import DistributedSampler

class DistributedOrderedSampler(DistributedSampler):
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        per_rank_total_size = self.total_size // self.num_replicas
        rank_idx_begin = self.rank * per_rank_total_size
        rank_idx_end = (self.rank+1) * per_rank_total_size
        indices = indices[rank_idx_begin:rank_idx_end]
        assert len(indices) == self.num_samples

        return iter(indices)