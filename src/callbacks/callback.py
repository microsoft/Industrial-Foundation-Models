from typing import Any, Dict, Type

import torch
from torch import Tensor
import torch.distributed as dist

callback_dict = {
    'MetricsCallback': 'metrics_callback',
    'TaskCoverageCallback': 'task_coverage_callback',
    'MemoryCallback': 'memory_callback',
    'ModelCheckpoint': 'checkpoint_callback',
    'EarlyStopping': 'earlystop_callback',
    'WandbCallback': 'wandb_callback'
}

class Callback:
    r"""
    Abstract base class to build callbacks.
    """
    
    def __init__(self) -> None:
        """Initialize callback"""
    
    @property
    def name(self) -> str:
        return callback_dict[self.__class__.__name__]
    
    def setup(self, trainer) -> None:
        """Called when setup callbacks"""
    
    def on_train_batch_start(self, trainer) -> None:
        """Called when train batch begins"""
    
    def on_train_batch_end(self, trainer) -> None:
        """Called when train batch ends"""
    
    def on_train_epoch_start(self, trainer) -> None:
        """Called when train epoch begins"""
    
    def on_train_epoch_end(self, trainer) -> None:
        """Called when train epoch ends"""
    
    def on_validation_batch_start(self, trainer) -> None:
        """Called when validation batch begins"""
    
    def on_validation_batch_end(self, trainer) -> None:
        """Called when validation batch ends"""
    
    def on_validation_epoch_start(self, trainer) -> None:
        """Called when validation epoch begins"""
    
    def on_validation_epoch_end(self, trainer) -> None:
        """Called when validation epoch ends"""
    
    def on_test_batch_start(self, trainer) -> None:
        """Called when test batch begins"""
    
    def on_test_batch_end(self, trainer) -> None:
        """Called when test batch ends"""
    
    def on_test_epoch_start(self, trainer) -> None:
        """Called when test epoch begins"""
    
    def on_test_epoch_end(self, trainer) -> None:
        """Called when test epoch ends"""