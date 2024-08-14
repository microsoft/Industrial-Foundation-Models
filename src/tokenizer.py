from transformers import AutoTokenizer
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding
from transformers.utils import PaddingStrategy
from typing import Any, Union, Dict, Optional


EXTRA_KEYS_TO_BE_PADDED = ['numeric_mask', 'answer_mask']


def _pad(
    self,
    encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
    max_length: Optional[int] = None,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    pad_to_multiple_of: Optional[int] = None,
    return_attention_mask: Optional[bool] = None,
) -> dict:
    """
    This function is mimicking transformers.PreTrainedTokenizerBase._pad(...).
    We only add the same padding options to keys in extra_keys_to_be_padded.
    """
    # Load from model defaults
    if return_attention_mask is None:
        return_attention_mask = "attention_mask" in self.model_input_names

    required_input = encoded_inputs[self.model_input_names[0]]

    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = len(required_input)

    if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

    # Initialize attention mask if not present.
    if return_attention_mask and "attention_mask" not in encoded_inputs:
        encoded_inputs["attention_mask"] = [1] * len(required_input)

    if needs_to_be_padded:
        difference = max_length - len(required_input)

        if self.padding_side == "right":
            if return_attention_mask:
                encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                )
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
            encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            # <Our Code\> right padding for extra keys
            for key in EXTRA_KEYS_TO_BE_PADDED:
                if key in encoded_inputs:
                    encoded_inputs[key] = encoded_inputs[key] + [self.pad_token_id] * difference
            # </Our Code>
        elif self.padding_side == "left":
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                    "token_type_ids"
                ]
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            # <Our Code\> left padding for extra keys
            for key in EXTRA_KEYS_TO_BE_PADDED:
                if key in encoded_inputs:
                    encoded_inputs[key] = [self.pad_token_id] * difference + encoded_inputs[key]
            # </Our Code>
        else:
            raise ValueError("Invalid padding strategy:" + str(self.padding_side))

    return encoded_inputs


class AutoTokenizerForTabLM(object):
    """This is a wrapper class for AutoTokenizer, which is used for tokenizing tabular data."""
    """If you need more features transformers.PreTrainedTokenizerBase, please export corresponding functions here."""

    def __init__(self, model_path_or_str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_path_or_str)

        # Below are some fixed customization for the loaded tokenizer
        self._tokenizer.pad_token_id = (
            0
        )
        self._tokenizer.padding_side = "left"
        self._tokenizer._pad = _pad  # a monkey patch for padding extra keys
    
    def from_pretrained(model_path_or_str):
        return AutoTokenizerForTabLM(model_path_or_str)
    
    def define_numerical_tokens(self):
        # The following tokens are used to represent numbers in the input sequence. For examples: 
        # 'Answer: -1.23e-04' will be tokenized as ['<s>', '▁Answer', ':', '▁-', '1', '.', '2', '3', 'e', '-', '0', '4']
        # 'Answer: 1.23e+04' will be tokenized as ['<s>', '▁Answer', ':', '▁', '1', '.', '2', '3', 'e', '+', '0', '4']
        if '▁-' in self._tokenizer.vocab:
            # This seems to be a special bug when using LLaMA tokenizer with special tokens
            self.sign_tokens = ['▁', '▁-', '+', '-']
        else:
            # This is the normal case (Phi-2 tokenizer)
            self.sign_tokens = ['+', '-']
        self.num_value_tokens = [str(i) for i in range(10)]
        self.num_scale_tokens = ['.', 'e']
        self.num_value_token_ids = [self._tokenizer.convert_tokens_to_ids(t) for t in self.num_value_tokens]
        self.num_pred_token_ids = self.num_value_token_ids + \
            [self._tokenizer.convert_tokens_to_ids(t) for t in self.sign_tokens + self.num_scale_tokens]
    
    def define_additional_special_tokens(self):
        self.num_begin_token = '<NUM_BEGIN>'
        self.num_end_token = '<NUM_END>'
        self.answer_begin_token = '<ANS_BEGIN>'
        self.answer_end_token = '<ANS_END>'
        self.additional_special_tokens = [
            self.num_begin_token,
            self.num_end_token,
            self.answer_begin_token,
            self.answer_end_token
        ]
    
    def add_additional_special_tokens(self):
        self.define_additional_special_tokens()
        self._tokenizer.add_special_tokens({
            'additional_special_tokens': self.additional_special_tokens
        })
        self.special_token_ids = {
            t: self._tokenizer.convert_tokens_to_ids(t) for t in self.additional_special_tokens
        }

    def __call__(self, *args, **kwargs):
        return self._tokenizer.__call__(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_ids(*args, **kwargs)
    
    def convert_ids_to_tokens(self, *args, **kwargs):
        return self._tokenizer.convert_ids_to_tokens(*args, **kwargs)
    
    def tokenize(self, *args, **kwargs):
        return self._tokenizer.tokenize(*args, **kwargs)
    
    def convert_tokens_to_string(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_string(*args, **kwargs)
    
    def encode(self, *args, **kwargs):
        return self._tokenizer.encode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)
    
    def batch_decode(self, *args, **kwargs):
        return self._tokenizer.batch_decode(*args, **kwargs)
    
    @property
    def bos_token_id(self):
        return self._tokenizer.bos_token_id
    
    @property
    def eos_token_id(self):
        return self._tokenizer.eos_token_id
    
    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id
    
    @property
    def vocab(self):
        return self._tokenizer.vocab
    
    def __len__(self):
        return len(self._tokenizer)