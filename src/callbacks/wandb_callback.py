from .callback import *
import wandb
import os

class WandbCallback(Callback):
    r"""
    Save and update metrics in train, validation and test stage.
    """
    
    def __init__(
        self,
        project_name: str = 'LLaMA2-GTL',
        run_name: str = None,
        config: Dict = None
    ):
        super().__init__()
        self.project_name = project_name
        self.run_name = run_name
        self.config = config
        self.run_tag = None
    
    def setup(self, trainer) -> None:
        if trainer.config.use_wandb and trainer.rank == 0:
            # Login
            if trainer.config.wandb_key_file is not None and os.path.exists(trainer.config.wandb_key_file):
                with open(trainer.config.wandb_key_file, 'r') as f:
                    api_key = f.read().strip()
                wandb.login(key=api_key)
            # Init experiment
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                # Track hyperparameters and run metadata
                config=self.config
            )
    
    def report_metrics(self, metrics: Dict, step: int) -> None:
        if self.run_tag is not None:
            metrics = {
                f'{self.run_tag}/{k}': v for k, v in metrics.items()
            }
        wandb.log(metrics, step=step)
    
    def on_train_batch_end(self, trainer) -> None:
        """Called when train batch ends"""
        if trainer.config.use_wandb and trainer.rank == 0:
            metrics = {
                'train/step_loss': trainer.train_step_loss,
                'train/epoch': trainer.epoch,
                'train/learning_rate': trainer.config.learning_rate if trainer.scheduler is None else trainer.scheduler.get_last_lr()[0]
            }
            self.report_metrics(metrics, step=trainer.global_step)
    
    def on_train_epoch_end(self, trainer) -> None:
        """Called when train epoch ends"""
        if trainer.config.use_wandb and trainer.rank == 0:
            metrics = {
                'train/epoch_loss': trainer.train_epoch_loss,
            }
            self.report_metrics(metrics, step=trainer.global_step)
    
    def on_validation_epoch_end(self, trainer) -> None:
        """Called when validation epoch ends"""
        if trainer.config.use_wandb and trainer.rank == 0:
            metrics = {
                'val/epoch_loss': trainer.val_epoch_loss,
            }
            metrics.update(trainer.val_metric_info)
            self.report_metrics(metrics, step=trainer.global_step)
    
    def on_test_epoch_end(self, trainer) -> None:
        """Called when test epoch ends"""
        if trainer.config.use_wandb and trainer.rank == 0:
            metrics = {
                'test/epoch_loss': trainer.test_epoch_loss,
            }
            metrics.update(trainer.test_metric_info)
            self.report_metrics(metrics, step=trainer.global_step)
            