# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

from .callback import *

import os
import numpy as np
from src.utils import dump_pickle, load_pickle

class MetricsCallback(Callback):
    r"""
    Save and update metrics in train, validation and test stage.
    """
    
    def setup(self, trainer) -> None:
        """Called when setup callbacks"""
        self.metrics = {}
        self.log_metrics = ['Val/All/AUROC', 'Val/All/NMAE', 'Test/All/AUROC', 'Test/All/NMAE']
    
    def update_metrics(self, metrics: Dict, global_step: int) -> None:
        if global_step not in self.metrics:
            self.metrics[global_step] = dict()
        self.metrics[global_step].update(metrics)
    
    def on_train_epoch_end(self, trainer) -> None:
        """Called when train epoch ends"""
        train_epoch_loss = trainer.train_epoch_loss
        train_perplexity = torch.exp(train_epoch_loss)
        train_epoch_metrics = {
            'train_loss': train_epoch_loss.item(),
            'train_perplexity': train_perplexity.item(),
        }
        if trainer.rank == 0:
            self.update_metrics(train_epoch_metrics, trainer.global_step)
    
    def on_validation_epoch_start(self, trainer) -> None:
        """Called when validation epoch starts"""
        outputs_save_path = trainer.config.generate_component_save_path(
            component=f'val_outputs',
            data_idx=trainer.val_data_idx,
            global_step=trainer.global_step,
            file_type='pkl'
        )
        if os.path.exists(outputs_save_path):
            self.should_skip_val = True
            trainer.all_val_outputs[trainer.val_data_idx] = load_pickle(outputs_save_path)
        else:
            self.should_skip_val = False
    
    def on_validation_epoch_end(self, trainer) -> None:
        """Called when validation epoch ends"""
        if self.should_skip_val:
            return
        
        global_step, train_data_idx, val_data_idx = trainer.global_step, trainer.train_data_idx, trainer.val_data_idx
        
        val_epoch_loss = trainer.val_epoch_loss
        val_ppl = torch.exp(val_epoch_loss)
        val_epoch_metrics = {
            'val_loss': val_epoch_loss.item(),
            'val_perplexity': val_ppl.item(),
        }
        val_epoch_metrics.update(trainer.val_metric_info)
        
        if trainer.rank == 0:
            self.update_metrics(val_epoch_metrics, global_step)
            
            # Print validation metrics
            log_metrics = ', '.join([f'{m}={val_epoch_metrics[m]:.4f}' for m in self.log_metrics if m in val_epoch_metrics])
            trainer.logging(f'Validation: Global step {global_step}, Train data {train_data_idx}, Val data {val_data_idx}, Metrics: {log_metrics}')

            if trainer.config.tabfm_stage == 'pretrain' and trainer.config.save_eval_metrics:
                outputs_save_path = trainer.config.generate_component_save_path(
                    component=f'val_outputs',
                    data_idx=val_data_idx,
                    global_step=global_step,
                    file_type='pkl'
                )
                dump_pickle(trainer.all_val_outputs[val_data_idx], outputs_save_path)
    
    def on_test_epoch_start(self, trainer) -> None:
        """Called when test epoch starts"""
        outputs_save_path = trainer.config.generate_component_save_path(
            component=f'test_outputs',
            data_idx=trainer.test_data_idx,
            global_step=trainer.global_step,
            file_type='pkl'
        )
        if os.path.exists(outputs_save_path):
            self.should_skip_test = True
            trainer.all_test_outputs[trainer.test_data_idx] = load_pickle(outputs_save_path)
        else:
            self.should_skip_test = False
    
    def on_test_epoch_end(self, trainer) -> None:
        """Called when test epoch ends"""
        if self.should_skip_test:
            return
        
        global_step, train_data_idx, test_data_idx = trainer.global_step, trainer.train_data_idx, trainer.test_data_idx
        
        test_epoch_loss = trainer.test_epoch_loss
        test_ppl = torch.exp(test_epoch_loss)
        test_epoch_metrics = {
            'test_loss': test_epoch_loss.item(),
            'test_perplexity': test_ppl.item(),
        }
        test_epoch_metrics.update(trainer.test_metric_info)
        
        if trainer.rank == 0:
            self.update_metrics(test_epoch_metrics, global_step)
            
            # Print test metrics
            log_metrics = ', '.join([f'{m}={test_epoch_metrics[m]:.4f}' for m in self.log_metrics if m in test_epoch_metrics])
            trainer.logging(f'Test: Global step {global_step}, Train data {train_data_idx}, Test data {test_data_idx}, Metrics: {log_metrics}')

            if trainer.config.tabfm_stage == 'pretrain' and trainer.config.save_eval_metrics:
                outputs_save_path = trainer.config.generate_component_save_path(
                    component=f'test_outputs',
                    data_idx=test_data_idx,
                    global_step=global_step,
                    file_type='pkl'
                )
                dump_pickle(trainer.all_test_outputs[test_data_idx], outputs_save_path)