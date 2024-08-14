from .callback import *
import os
from src.utils import dump_pickle, load_pickle


class TaskCoverageCallback(Callback):
    r"""
    Save and update task coverage in train stage.
    """
    def __init__(self):
        super().__init__()
    
    def setup(self, trainer) -> None:
        """Called when setup callbacks"""
        self.task_coverage_dict = {}
    
    def update_coverage(self, data_idx, task_indices) -> None:
        """Update task coverage"""
        for task_idx in task_indices:
            self.task_coverage_dict[(data_idx, task_idx)] = self.task_coverage_dict.get((data_idx, task_idx), 0) + 1
    
    def save_coverage(self, path):
        dump_pickle(self.task_coverage_dict, path)
    
    def reload_coverage(self, trainer):
        task_coverage_save_path = trainer.config.generate_component_save_path(
            component='task_coverage',
            data_idx=trainer.train_data_idx,
            global_step=trainer.global_step,
            rank=trainer.rank,
            file_type='pkl'
        )
        try:
            self.task_coverage_dict = load_pickle(task_coverage_save_path)
            trainer.logging(f"Reload task coverage from {task_coverage_save_path}")
        except FileNotFoundError:
            trainer.logging(f"Task coverage file not found in {task_coverage_save_path}")
    
    def on_train_batch_end(self, trainer) -> None:
        """Called when train batch ends"""
        self.update_coverage(trainer.train_data_idx, trainer.train_step_task_indices)
        
        if not trainer.config.save_model or trainer.global_step == 0:
            return
        
        save_every_n_steps = trainer.config.save_ckpt_every_n_steps
        if save_every_n_steps > 0 and (trainer.global_step % save_every_n_steps) == 0:
            task_coverage_save_path = trainer.config.generate_component_save_path(
                component='task_coverage',
                data_idx=trainer.train_data_idx,
                global_step=trainer.global_step,
                rank=trainer.rank,
                file_type='pkl'
            )
            self.save_coverage(task_coverage_save_path)
    
    def on_validation_epoch_start(self, trainer) -> None:
        """Called when validation epoch starts"""
        if not trainer.config.save_model or trainer.global_step == 0:
            return
        
        if trainer.config.save_ckpt_per_each_val:
            task_coverage_save_path = trainer.config.generate_component_save_path(
                component='task_coverage',
                data_idx=trainer.train_data_idx,
                global_step=trainer.global_step,
                rank=trainer.rank,
                file_type='pkl'
            )
            self.save_coverage(task_coverage_save_path)
    