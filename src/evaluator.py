from transformers.tokenization_utils import PreTrainedTokenizer
from scipy.special import softmax
import os
import torch
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from .config import Config
from .utils import isfloat, isint


class Evaluator:
    def __init__(
        self,
        config: Config,
        tokenizer: PreTrainedTokenizer
    ):
        self.config = config
        # We need tokenizer to recover numbers from generated tokens
        self.tokenizer = tokenizer
        self.end_token = tokenizer.decode(tokenizer.eos_token_id)
        self.rank = int(os.environ.get("RANK") or 0)
        self.eps = 1e-5

    @classmethod
    def decode_multiclass_probs_from_logits(cls, answer_logits: np.array, class_num: int, end_token: str = None):
        """
            Decode fixed-length logits to multi-class probabilities.
            
            Args:
                answer_logits (np.array): A 3D numpy array of logits, where is shape of (batch_size, token_length, valid_tokens_num).
                class_num (int): The number of classes.
                end_token (str): The end token of answer.
            
            Returns:
                probs (np.array): A 2D numpy array of probabilities, where is shape of (batch_size, class_num).
                preds: (np.array): A 1D numpy array of float predictions, where is shape of (batch_size).
        """
        # NOTE: for autoregressive decoding paradigm, this method is not reasonable for multi-token predictions
        # We only use this function to calculate 1-digit categories probabilities
        valid_tokens = list(f'{x}' for x in range(10))
        valid_token_indices = {t: i for i, t in enumerate(valid_tokens)}
        probs = np.ones([answer_logits.shape[0], class_num], dtype=np.float32)
        for i in range(answer_logits.shape[0]):
            for c in range(class_num):
                probs[i][c] = answer_logits[i][1][valid_token_indices[str(c)]]
        return softmax(probs, axis=-1), np.argmax(probs, axis=-1)

    @classmethod 
    def decode_answer_from_ids(
        cls,
        answer_ids: torch.LongTensor,
        tokenizer: PreTrainedTokenizer,
        end_token: str,
        answer_type: str = 'float'
    ):
        """
            Decode answer_ids generated auto-regressively to a float type answer.
            
            Args:
                answer_ids (torch.LongTensor): A 2D tensor of answer ids, where is shape of (batch_size, token_length).
                tokenizer: A PreTrainedTokenizer instance.
                end_token (str): The end token of answer.
                answer_type (str): The type of answer, 'float' or 'int'.
            
            Returns:
                preds: (np.array): A 1D numpy array of float predictions, where is shape of (batch_size).
        """
        # decode answer_ids to answer string
        preds = np.zeros([answer_ids.shape[0]])
        for i in range(answer_ids.shape[0]):
            full_answer = tokenizer.decode(answer_ids[i])
            possible_valid_answer = full_answer.split(end_token)[0]
            invalid_chars = [' ', ',', '#', '$', '%', '[', ']', '<', '>', '|', '{', '}', '\\', '^', '~', '`', '\n']
            for c in invalid_chars:
                possible_valid_answer = possible_valid_answer.replace(c, '')
            if answer_type == 'float' and isfloat(possible_valid_answer):
                preds[i] =  float(possible_valid_answer)
            if answer_type == 'int' and isint(possible_valid_answer):
                preds[i] =  int(possible_valid_answer)
        return preds

    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        eval_task_info: Dict[int, Tuple[str, Dict[str, Any]]],
        stage: str = None
    ):
        """
            outputs: dictionary of outputs, containing all the valid outputs in src.models.NumCausalLMOutputWithPast
            labels:  torch.Tensor, label_tabular of src.models.LlamaForCausalLM.forward
        """
        metric_info = {}
        
        # remove loss aggregation because it is not necessary for an evaluator
        # task_labels = labels.numpy()
        # for loss_key in ['loss', 'loss_answer', 'loss_numeric_feat', 'loss_numeric_token', 'loss_task']:
        #     if loss_key in outputs:
        #         metric_info.update({loss_key: np.mean(outputs[loss_key].numpy())})
        
        answer_task_idx = outputs['task_idx'].numpy()
        answer_logits = outputs['answer_logits'].numpy()
        answer_ids = outputs['answer_ids'].numpy().astype('int')
        answer_labels = labels.numpy()

        gathered_metrics = {}
        for task_idx, (dataset_name, task_info) in eval_task_info.items():
            task_idx = int(task_idx)
            task_valid_idx = np.where(answer_task_idx == task_idx)[0]
            if len(task_valid_idx) == 0:
                continue
            
            task_logits = answer_logits[task_valid_idx]
            task_labels = answer_labels[task_valid_idx]
            task_answer_ids = answer_ids[task_valid_idx]
            
            task_class_num = 1
            if task_info['task_type'] == 'classification':
                task_labels = task_labels.astype(np.int32)
                task_label_num = np.unique(task_labels).shape[0]
                task_class_num = task_info['class_num']
                if task_label_num > task_class_num:
                    print(f"[ERROR] dataset:{dataset_name}, task_label_num:{task_label_num}, task_class_num:{task_class_num}")
                    continue
            
            # Decode logits to predictions
            if task_class_num >= 10:
                task_preds = self.decode_answer_from_ids(task_answer_ids, self.tokenizer, self.end_token, answer_type='int')
            elif task_class_num >= 2:
                task_probs, task_preds = self.decode_multiclass_probs_from_logits(task_logits, task_class_num, self.end_token)
            else:
                task_preds = self.decode_answer_from_ids(task_answer_ids, self.tokenizer, self.end_token, answer_type='float')

            # Calculate metrics
            task_metrics = {}
            if task_class_num >= 10:
                task_metrics['Accuracy'] = accuracy_score(task_labels, task_preds)
            elif task_class_num > 2:
                task_metrics['Accuracy'] = accuracy_score(task_labels, task_preds)
                if np.unique(task_labels).shape[0] == task_class_num:
                    task_metrics['AUROC'] = roc_auc_score(task_labels, task_probs, multi_class='ovr')
            elif task_class_num == 2:
                task_metrics['Accuracy'] = accuracy_score(task_labels, task_preds)
                task_metrics['AUPRC'] = average_precision_score(task_labels, task_probs[:, 1])
                if np.unique(task_labels).shape[0] == task_class_num:
                    task_metrics['AUROC'] = roc_auc_score(task_labels, task_probs[:, 1])
            else:
                task_metrics['NRMSE'] = np.sqrt(np.mean((task_labels - task_preds) ** 2)) / (np.mean(np.abs(task_labels)) + self.eps)
                task_metrics['NMAE'] = np.mean(np.abs(task_labels - task_preds)) / (np.mean(np.abs(task_labels)) + self.eps)
            
            # Gather metrics
            for k, v in task_metrics.items():
                if k not in gathered_metrics:
                    gathered_metrics[k] = [v]
                else:
                    gathered_metrics[k].append(v)
                metric_info.update({f"{task_idx}-{dataset_name}/{k}": v})

        for k, v in gathered_metrics.items():
            metric = -1 if len(v) == 0 else np.mean(v)
            metric_info.update({f"All/{k}": metric})

        # Add stage as prefix
        if stage is not None:
            metric_info = {f'{stage}/{k}': v for k, v in metric_info.items()}

        return metric_info