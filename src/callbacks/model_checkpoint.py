from .callback import *

import os
from src.utils import save_model_and_optimizer_sharded

class ModelCheckpoint(Callback):
    r"""
    Handle model checkpoint.
    """
    def __init__(self):
        super().__init__()
    
    def save_model_checkpoint(self, trainer, save_path):
        model, optimizer, config, rank = trainer.model, trainer.optimizer, trainer.config, trainer.rank

        if config.enable_fsdp:
            dist.barrier()

        if config.save_optimizer:
            save_model_and_optimizer_sharded(model, rank, save_path, optimizer=optimizer)
        else:
            save_model_and_optimizer_sharded(model, rank, save_path)
        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
        print("=====================================================")
        
        if config.enable_fsdp:
            dist.barrier()
    
    def on_train_batch_end(self, trainer) -> None:
        """Called when train batch ends"""
        if not trainer.config.save_model or trainer.global_step == 0:
            return
        
        save_every_n_steps = trainer.config.save_ckpt_every_n_steps
        if save_every_n_steps > 0 and (trainer.global_step % save_every_n_steps) == 0:
            save_path = trainer.config.generate_component_save_path(
                component='checkpoint',
                global_step=trainer.global_step
            )
            self.save_model_checkpoint(trainer, save_path)
    
    def on_train_epoch_end(self, trainer) -> None:
        """Called when train epoch ends"""
        if not trainer.config.save_model or trainer.global_step == 0:
            return
        
        if trainer.config.save_ckpt_per_epoch:
            save_path = trainer.config.generate_component_save_path(
                component='checkpoint',
                global_step=trainer.global_step
            )
            if not os.path.exists(save_path):
                self.save_model_checkpoint(trainer, save_path)
    
    def on_validation_epoch_start(self, trainer) -> None:
        """Called when validation epoch starts"""
        if not trainer.config.save_model or not trainer.config.save_ckpt_per_each_val or trainer.global_step == 0:
            return
        
        save_path = trainer.config.generate_component_save_path(
            component='checkpoint',
            global_step=trainer.global_step
        )
        if not os.path.exists(save_path):
            self.save_model_checkpoint(trainer, save_path)
    
    def on_validation_epoch_end(self, trainer) -> None:
        """Called when validation epoch ends"""
        if not trainer.config.save_model or trainer.global_step == 0 or not trainer.is_val_finished:
            return
        
        if trainer.config.save_best_ckpt and trainer.is_best_step_so_far:
            save_path = trainer.config.generate_component_save_path('best_checkpoint')
            self.save_model_checkpoint(trainer, save_path)
