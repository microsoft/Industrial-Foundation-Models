from re import S
import torch
import torch.nn as nn
import torch.distributed as dist
import math
import numpy as np

from typing import Set, Union, Tuple, Optional, List
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM, PreTrainedModel,
)
from transformers.utils import logging
from transformers.modeling_outputs import ModelOutput
from transformers.generation.utils import GenerationMixin, GenerateOutput

from src.loss import WeightedNumericTokenLoss
from src.utils import extract_masked_continuous_values, shift_unmasked_values
from src.tokenizer import AutoTokenizerForTabLM

logger = logging.get_logger(__name__)


@dataclass
class TabCausalLMOutputWithPast(ModelOutput):
    """
    This class is mimicking transformers.modeling_outputs.
    """
    # loss related
    loss: Optional[torch.FloatTensor] = None
    loss_answer: Optional[torch.FloatTensor] = None
    loss_numeric_token: Optional[torch.FloatTensor] = None
    loss_numeric_feat: Optional[torch.FloatTensor] = None

    # model related
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # prediction related
    answer_logits: torch.FloatTensor = None
    answer_ids: torch.LongTensor = None

    # index related to track back to an input sample
    sample_idx: torch.LongTensor = None
    task_idx: torch.LongTensor = None

    # TODO: not sure whether we need the following two fields in the output?
    label_mean: torch.FloatTensor = None
    label_std: torch.FloatTensor = None

    @classmethod
    def _get_valid_keys(cls):
        # only use valid outputs to evaluate
        return [k for k in cls.__dict__.keys() if not k.startswith("_")]
    
    @classmethod
    def get_eval_output_keys(cls):
        return [
            'sample_idx', 'task_idx', 'answer_logits', 'answer_ids',
        ]


class AutoTabCausalLM(nn.Module):
    """
    This class is a wrapper of transformers.models.auto.modeling_auto.AutoModelForCausalLM, with additional features for tabular data.
    """
    def __init__(self,
                 config):
        super().__init__()
        self.config = config

        if config.pure_bf16:
            dtype_kwargs = {"torch_dtype": torch.bfloat16}
        elif config.use_fp16:
            dtype_kwargs = {"torch_dtype": torch.float16}
        else:
            dtype_kwargs = {}

        self.causal_lm = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            load_in_8bit=True if config.load_in_kbit else None,
            device_map="auto" if config.load_in_kbit else None,
            attn_implementation=config.attn_implementation,
            **dtype_kwargs,
        )
        
        # Change the base frequency of rotary embeddings in the model to support longer sequences
        if config.change_rope_base_freq:
            for i in range(len(self.causal_lm.model.layers)):
                rotary_emb = self.causal_lm.model.layers[i].self_attn.rotary_emb
                dim = rotary_emb.dim
                device = rotary_emb.inv_freq.device
                self.base = 500000
                new_inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
                self.causal_lm.model.layers[i].self_attn.rotary_emb.register_buffer("inv_freq", new_inv_freq, persistent=False)
                self.causal_lm.model.layers[i].self_attn.rotary_emb.max_position_embeddings = 16384
                self.causal_lm.model.layers[i].self_attn.rotary_emb.max_seq_len_cached = 16384
        
        self.use_numeric_token_loss = False
        self.numeric_head = None
        
        self.numeric_feat_loss_weight = 0
        self.numeric_token_loss_weight = 0

    def init_constant_tokens(self, tokenizer: AutoTokenizerForTabLM):
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        self.numerical_token_ids = tokenizer.num_value_token_ids
        self.task_pred_token_ids = tokenizer.num_pred_token_ids + [tokenizer.eos_token_id]

    def init_custom_modules(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.init_constant_tokens(tokenizer)
        
        # resize the token embeddings to include additional tokens
        # self.causal_lm.model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

        # whether to calculate the next-token prediction loss for numeric tokens
        self.use_numeric_token_loss = self.config.use_numeric_token_loss
        
        # whether use the numerical cross-entropy loss for numerical tokens
        self.use_weighted_numeric_loss = self.config.use_weighted_numeric_loss
        
        if self.config.use_numeric_head:
            # An example of adding extra parameters to the model
            # Here we define a linear layer 
            self.numeric_head = nn.Linear(self.causal_lm.config.hidden_size, 1, bias=False)
            self.numeric_head.to(device)
        
        self.numeric_feat_loss_weight = self.config.numeric_feat_loss_weight
        self.numeric_token_loss_weight = self.config.numeric_token_loss_weight

    def prepare_inputs_embeddings(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        num_feats: Optional[torch.FloatTensor] = None,
        norm_feats: Optional[torch.FloatTensor] = None,
        numeric_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None
    ):
        if inputs_embeds is None:
            inputs_embeds = self.causal_lm.model.embed_tokens(input_ids)
        # may add some customizations here to enrich the input embeddings with more numerical information

        return inputs_embeds

    def compute_loss(
        self,
        logits: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        num_feats: Optional[torch.FloatTensor] = None,
        norm_feats: Optional[torch.FloatTensor] = None,
        numeric_mask: Optional[torch.Tensor] = None,
        answer_mask: Optional[torch.Tensor] = None,
    ):
        """
        Computes the loss for the tabular causal language model.
        
        Args:
            logits (torch.FloatTensor): The predicted logits from the model.
            labels (Optional[torch.LongTensor]): The target labels.
            hidden_states (Optional[torch.FloatTensor]): The hidden states from the model.
            num_feats (Optional[torch.FloatTensor]): The raw numerical features.
            norm_feats (Optional[torch.FloatTensor]): The normalized numerical features.
            numeric_mask (Optional[torch.Tensor]): The mask for numerical tokens.
            answer_mask (Optional[torch.Tensor]): The mask for answer tokens.
        
        Returns:
            Tuple[torch.Tensor]: A tuple containing the total loss, answer loss, numerical feature loss, and numerical token loss.
        """
        # Assert the length of generated logits is 1 less than the length of labels
        if labels is None:
            return None, None, None, None

        # the following alignment is used when the logits is expanded autoregressively from input_ids but may be shorter than or exceed the length of labels
        valid_len = min(labels.shape[1], logits.shape[1] + 1)
        logits = logits[:, :valid_len-1]
        hidden_states = hidden_states[:, :valid_len-1]
        labels = labels[:, :valid_len]
        numeric_mask = numeric_mask[:, :valid_len] if numeric_mask is not None else None
        answer_mask = answer_mask[:, :valid_len] if answer_mask is not None else None

        ce_loss_fct = nn.CrossEntropyLoss()
        mse_loss_fct = nn.MSELoss()
        
        # Initialize numerical loss function
        weighted_num_loss_fct = WeightedNumericTokenLoss(
            numerical_token_ids=self.numerical_token_ids,
            pad_token_id=self.pad_token_id,
            decay_method='exp',
            decay_rate=0.5
        )
            
        zero_loss_value = torch.tensor(0.0).to(logits.device)

        loss_answer = zero_loss_value
        if labels is not None and answer_mask is not None:
            # TODO: apply numerical ce loss to numerical features
            if self.use_weighted_numeric_loss:
                # before_answer_mask = torch.concat([answer_mask[:, 1:], answer_mask[:, :1]], dim=-1)
                before_answer_mask = answer_mask[:, 1:]
                numerical_token_ids = extract_masked_continuous_values(labels, answer_mask, padding_value=self.pad_token_id)
                numerical_logits = extract_masked_continuous_values(logits, before_answer_mask, padding_value=0)
                loss_answer = weighted_num_loss_fct(numerical_logits, numerical_token_ids)
            else:
                answer_token_pos = (answer_mask == 1)
                # before_answer_token_pos = torch.concat([answer_token_pos[:, 1:], answer_token_pos[:, :1]], dim=-1)
                before_answer_token_pos = answer_token_pos[:, 1:]
                # Shift so that tokens < n predict n
                shift_logits = logits[before_answer_token_pos].contiguous()
                shift_labels = labels[answer_token_pos].contiguous()
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss_answer = ce_loss_fct(shift_logits, shift_labels)
        
        loss_numeric_token = zero_loss_value
        if self.use_numeric_token_loss and numeric_mask is not None and numeric_mask.sum() > 0:
            numeric_token_pos = (numeric_mask == 1)
            # before_numeric_token_pos = torch.concat([numeric_token_pos[:, 1:], numeric_token_pos[:, :1]], dim=-1)
            before_numeric_token_pos = numeric_token_pos[:, 1:]
            shifted_nt_logits = logits[before_numeric_token_pos].contiguous()
            shifted_nt_labels = labels[numeric_token_pos].contiguous()
            loss_numeric_token = ce_loss_fct(shifted_nt_logits, shifted_nt_labels)
        
        loss_numeric_feat = zero_loss_value
        if self.numeric_head is not None and numeric_mask is not None and numeric_mask.sum() > 0:
            # To identify the positions of the last numeric token of each numeric feature
            # before_numeric_feat_pos = (torch.diff(numeric_mask, dim=-1, append=numeric_mask[:, -1:]) == -1)
            before_numeric_feat_pos = (torch.diff(numeric_mask, dim=-1) == -1)
            norm_preds = self.numeric_head(hidden_states[before_numeric_feat_pos])
            if norm_feats is not None:
                loss_numeric_feat = mse_loss_fct(norm_preds, norm_feats.reshape(-1, 1))
        
        loss = loss_answer + \
            self.numeric_token_loss_weight * loss_numeric_token + \
            self.numeric_feat_loss_weight * loss_numeric_feat
        
        return loss, loss_answer, loss_numeric_token, loss_numeric_feat

    def forward(
        self,
        # Sample index
        sample_idx: Optional[torch.LongTensor] = None,
        task_idx: Optional[torch.LongTensor] = None,
        # Tabular data
        num_feats: Optional[torch.FloatTensor] = None,
        norm_feats: Optional[torch.FloatTensor] = None,
        num_feats_mask: Optional[torch.Tensor] = None,
        label_tabular: Optional[Union[torch.LongTensor, torch.FloatTensor]] = None,
        norm_label_tabular: Optional[torch.FloatTensor] = None,
        label_mean: Optional[torch.FloatTensor] = None,
        label_std: Optional[torch.FloatTensor] = None,
        # Text data with only the template
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        numeric_mask: Optional[torch.Tensor] = None,
        answer_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # Other input args (not used at present)
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        # Generate
        is_generate: Optional[bool] = False,
        **kwargs
    ) -> TabCausalLMOutputWithPast:
        if num_feats_mask is not None:
            raw_num_feats = num_feats
            raw_norm_feats = norm_feats
            num_feats = num_feats[num_feats_mask == 1].contiguous()
            norm_feats = norm_feats[num_feats_mask == 1].contiguous()
        
        # We call generate function there to reuse the initialization of the prepared inputs
        if is_generate:
            if self.config.use_legacy_generate:
                return self.legacy_generate(
                    num_feats=num_feats,
                    norm_feats=norm_feats,
                    label_tabular=label_tabular,
                    norm_label_tabular=norm_label_tabular,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numeric_mask=numeric_mask,
                    answer_mask=answer_mask,
                    labels=labels,
                    sample_idx=sample_idx,
                    task_idx=task_idx,
                    label_mean=label_mean,
                    label_std=label_std,
                    return_dict=True,
                    synced_gpus=(dist.get_world_size() > 1),
                    **kwargs
                )
            else:
                max_new_tokens = kwargs.pop('max_generate_length', 10)
                kwargs['max_new_tokens'] = max_new_tokens
                return self.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    answer_mask=answer_mask,
                    sample_idx=sample_idx,
                    task_idx=task_idx,
                    return_dict=True,
                    pad_to_max_length=True,
                    synced_gpus=(dist.get_world_size() > 1),
                    **kwargs,
                )
        
        inputs_embeds = self.prepare_inputs_embeddings(
            input_ids=input_ids,
            num_feats=num_feats,
            norm_feats=norm_feats,
            numeric_mask=numeric_mask,
            inputs_embeds=inputs_embeds,
        )

        # For example, if causal_lm.model is an object of LlamaModel, 
        # (See transformers.models.llama.modeling_llama.LlamaModel)
        # outputs = BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
        outputs = self.causal_lm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs[0]
        logits = self.causal_lm.lm_head(hidden_states)
        
        loss, loss_answer, loss_numeric_token, loss_numeric_feat = self.compute_loss(
            logits=logits,
            labels=labels,
            hidden_states=hidden_states,
            num_feats=num_feats,
            norm_feats=norm_feats,
            numeric_mask=numeric_mask,
            answer_mask=answer_mask
        )

        return TabCausalLMOutputWithPast(
            loss=loss,
            loss_answer=loss_answer,
            loss_numeric_token=loss_numeric_token,
            loss_numeric_feat=loss_numeric_feat,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sample_idx=sample_idx,
            task_idx=task_idx,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        num_beams=1,
        do_sample=False,
        use_cache=True,
        return_dict=False,
        pad_to_max_length=False,
        max_new_tokens=100,
        **kwargs,
    ) -> Union[torch.LongTensor, TabCausalLMOutputWithPast]:
        """
        This function is a wrapper of transformers.generation.utils.GenerationMixin.generate()
        Here we expose some critical arguments.
        Please refer to transformers.generation.configuration_utils.GenerationConfig() for other arguments
        """
        # pop useless model arguments, or they will raise errors in self.causal_lm.generate()
        labels = kwargs.pop('labels', None)
        numeric_mask = kwargs.pop('numeric_mask', None)
        answer_mask = kwargs.pop('answer_mask', None)
        sample_idx = kwargs.pop('sample_idx', None)
        task_idx = kwargs.pop('task_idx', None)
        num_feats = kwargs.pop('num_feats', None)
        norm_feats = kwargs.pop('norm_feats', None)
        num_feats_mask = kwargs.pop('num_feats_mask', None)
        label_tabular = kwargs.pop('label_tabular', None)
        norm_label_tabular = kwargs.pop('norm_label_tabular', None)

        if answer_mask is not None:
            # if 'answer_mask' is provided, we assume that 'input_ids' contains answer tokens to be generated
            # so we shift the answer tokens to the right to remove them from the input
            input_ids = shift_unmasked_values(input_ids, answer_mask, padding_value=self.pad_token_id)
            attention_mask = shift_unmasked_values(attention_mask, answer_mask, padding_value=0)

            answer_mask = shift_unmasked_values(answer_mask, answer_mask, padding_value=0)
            assert answer_mask.sum().int() == 0, "Answer tokens should be removed from the input_ids"
            
        # the causal_lm must inherit from GenerationMixin
        assert isinstance(self.causal_lm, GenerationMixin)

        gen_out = self.causal_lm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            use_cache=use_cache,
            output_logits=True,
            return_dict_in_generate=True,
            **kwargs,
        )

        # we always return dict for the generation output
        # gen_out should be an object of transformers.generation.utils.GenerateOutput
        output_ids = gen_out.sequences[:, input_ids.shape[1]:]

        if not return_dict:
            return output_ids

        # GenerateOutput.logits is a tuple of stepwise logits ([batch_size, vocab_size]) with the length of output_ids.shape[1]
        output_logits = torch.stack(gen_out.logits, dim=1)
        assert output_logits.shape[1] == output_ids.shape[1], "The length of output logits should be equal to the length of output ids"

        answer_ids = output_ids
        answer_logits = output_logits[:, :, self.task_pred_token_ids]  # only contains task-specific tokens to save memory

        if pad_to_max_length:
            pad_answer_length = max_new_tokens - output_ids.shape[1]
            if pad_answer_length > 0:
                pad_answer_logits = torch.zeros(answer_logits.shape[0], pad_answer_length, answer_logits.shape[2], device=answer_logits.device)
                answer_logits = torch.cat([answer_logits, pad_answer_logits], dim=1)
                pad_answer_ids = self.pad_token_id * torch.ones(answer_ids.shape[0], pad_answer_length, device=answer_ids.device, dtype=torch.long)
                answer_ids = torch.cat([answer_ids, pad_answer_ids], dim=1)

        tab_out = TabCausalLMOutputWithPast(
            logits=output_logits,
            answer_ids=answer_ids,
            answer_logits=answer_logits,
            sample_idx=sample_idx,
            task_idx=task_idx,
        )

        return tab_out

    
    def legacy_generate(
        self,
        max_generate_length: int = 20,
        # Tabular data
        num_feats: Optional[torch.FloatTensor] = None,
        norm_feats: Optional[torch.FloatTensor] = None,
        label_tabular: Optional[Union[torch.LongTensor, torch.FloatTensor]] = None,
        norm_label_tabular: Optional[torch.FloatTensor] = None,
        # Text data with only the template
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        numeric_mask: Optional[torch.Tensor] = None,
        answer_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # Token
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        # Index
        sample_idx: Optional[torch.LongTensor] = None,
        task_idx: Optional[torch.LongTensor] = None,
        label_mean: Optional[torch.FloatTensor] = None,
        label_std: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        synced_gpus: bool = False,
        **kwargs
    ) -> Union[Tuple, TabCausalLMOutputWithPast]:
        # Copied from transformers.modeling_llama.LlamaForCausalLM.generate
        # Only support greedy search for now
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        
        # Remove answers and pad to the left
        if answer_mask is not None:
            input_ids = shift_unmasked_values(input_ids, answer_mask, pad_token_id)
            attention_mask = shift_unmasked_values(attention_mask, answer_mask, 0)
            numeric_mask = shift_unmasked_values(numeric_mask, answer_mask, 0)
            labels = shift_unmasked_values(labels, answer_mask, pad_token_id, maintain_masked_values=True)
            answer_mask = shift_unmasked_values(answer_mask, answer_mask, 0, maintain_masked_values=True)
        
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        this_peer_finished = False
        generate_length = 0
        
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            
            inputs_embeds = self.prepare_inputs_embeddings(input_ids, num_feats, norm_feats, numeric_mask)
            outputs = self.causal_lm.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=return_dict
            )
            
            # At present, we generate until the all gpu's sequence is finished for some syncronization issues
            # if synced_gpus and this_peer_finished:
            #     continue  # don't waste resources running the code we don't need
            
            hidden_states = outputs[0]
            logits = self.causal_lm.lm_head(hidden_states)
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)], dim=-1)
            if numeric_mask is not None:
                numeric_mask = torch.cat([numeric_mask, torch.zeros((numeric_mask.shape[0], 1), device=numeric_mask.device)], dim=-1)
            
            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True
            
            generate_length += 1
            if generate_length >= max_generate_length:
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break
        
        # Calculate loss for the generated answer
        loss, loss_answer, loss_numeric_token, loss_numeric_feat = self.compute_loss(
            logits=logits,
            labels=labels,
            hidden_states=hidden_states,
            num_feats=num_feats,
            norm_feats=norm_feats,
            numeric_mask=numeric_mask,
            answer_mask=answer_mask
        )
        
        answer_logits = logits[:, -generate_length:, self.task_pred_token_ids]
        answer_ids = input_ids[:, -generate_length:]
        
        # Pad the answer to the maximum generate length for multiple gpus
        pad_answer_length = max_generate_length - generate_length
        if pad_answer_length > 0:
            pad_answer_logits = torch.zeros(answer_logits.shape[0], pad_answer_length, answer_logits.shape[2]).to(answer_logits.device)
            answer_logits = torch.cat([answer_logits, pad_answer_logits], dim=1)
            pad_answer_ids = pad_token_id * torch.ones(answer_ids.shape[0], pad_answer_length).to(answer_ids.device)
            answer_ids = torch.cat([answer_ids, pad_answer_ids], dim=1)
        
        return TabCausalLMOutputWithPast(
            loss=loss,
            loss_answer=loss_answer,
            loss_numeric_feat=loss_numeric_feat,
            loss_numeric_token=loss_numeric_token,
            answer_logits=answer_logits,
            answer_ids=answer_ids,
            sample_idx=sample_idx,
            task_idx=task_idx,
            label_mean=label_mean,
            label_std=label_std,
        )