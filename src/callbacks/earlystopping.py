from .callback import *
import os
import numpy as np
from src.utils import dump_pickle, load_pickle


class EarlyStopping(Callback):
    r"""
    Earlystop callback.
    """
    def __init__(self, patience=16, delta=0, metric=None, greater_is_better=False):
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.metric_data_idx, self.metric_name = metric.split('/')
        self.greater_is_better = greater_is_better
    
    def setup(self, trainer):
        self.counter = 0
        self.best_score = float("inf") if not self.greater_is_better else -float("inf")
        self.should_stop = False

    def update_state(self):
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True

    def check_earlystop(self, score):
        if not self.greater_is_better and score >= self.best_score - self.delta:
            self.update_state()
        elif self.greater_is_better and score <= self.best_score + self.delta:
            self.update_state()
        else:
            self.best_score = score
            self.counter = 0
    
    def save_earlystop(self, path):
        earlystop_info = {
            'counter': self.counter,
            'best_score': self.best_score,
            'should_stop': self.should_stop
        }
        dump_pickle(earlystop_info, path)
    
    def reload_earlystop(self, trainer):
        earlystop_save_path = trainer.config.generate_component_save_path(
            component='earlystop',
            global_step=trainer.global_step,
            file_type='pkl'
        )
        try:
            earlystop_info = load_pickle(earlystop_save_path)
            self.counter = earlystop_info['counter']
            self.best_score = earlystop_info['best_score']
            self.should_stop = earlystop_info['should_stop']
            trainer.logging(f"Reload earlystop from {earlystop_save_path}")
        except FileNotFoundError:
            trainer.logging(f"Earlystop file not found in {earlystop_save_path}")
    
    def on_train_batch_end(self, trainer) -> None:
        """Called when train batch ends"""
        if not trainer.config.save_model or trainer.global_step == 0 or not trainer.enable_early_stopping:
            return
        
        save_every_n_steps = trainer.config.save_ckpt_every_n_steps
        if save_every_n_steps > 0 and (trainer.global_step % save_every_n_steps) == 0:
            if trainer.rank == 0:
                earlystop_save_path = trainer.config.generate_component_save_path(
                    component='earlystop',
                    global_step=trainer.global_step,
                    file_type='pkl'
                )
                self.save_earlystop(earlystop_save_path)
    
    def on_validation_epoch_start(self, trainer) -> None:
        """Called when validation epoch starts"""
        if not trainer.config.save_model or trainer.global_step == 0 or not trainer.enable_early_stopping:
            return
        
        if trainer.config.save_ckpt_per_each_val and trainer.rank == 0:
            earlystop_save_path = trainer.config.generate_component_save_path(
                component='earlystop',
                global_step=trainer.global_step,
                file_type='pkl'
            )
            if not os.path.exists(earlystop_save_path) :
                self.save_earlystop(earlystop_save_path)
    
    def on_validation_epoch_end(self, trainer) -> None:
        """Called when validation epoch ends"""
        if not trainer.is_val_finished or not trainer.enable_early_stopping:
            return
        
        val_score_list = []
        metric_name = f'Val/All/{self.metric_name}'
        for k, v in trainer.all_val_outputs.items():
            metrics = v['metrics']
            if metric_name in metrics:
                if self.metric_data_idx == 'All' or k == int(self.metric_data_idx):
                    val_score_list.append(metrics[metric_name])
        val_score = np.mean(val_score_list)
        self.check_earlystop(val_score)
        
        # Best score
        if self.counter == 0:
            trainer.logging(
                f"Best score [{self.metric_data_idx}/{self.metric_name}] on data {trainer.train_data_idx}"
                f" with training {trainer.global_step} steps is {val_score}"
            )
            trainer.is_best_step_so_far = True
        else:
            trainer.is_best_step_so_far = False
        
        # Save earlystop info
        if trainer.config.save_ckpt_per_each_val and trainer.rank == 0:
            earlystop_save_path = trainer.config.generate_component_save_path(
                component='earlystop',
                global_step=trainer.global_step,
                file_type='pkl'
            )
            self.save_earlystop(earlystop_save_path)