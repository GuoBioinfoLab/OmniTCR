"""Amino-acid tokenizer used by the pretrained OmniTCR backbone."""

import json
import os
import re
from typing import Iterable, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer


class AminoAcidTokenizer(PreTrainedTokenizer):
    """Character tokenizer with fixed OmniTCR component tokens.

    The token order is intentionally identical to the original training code.
    Changing the order changes token IDs and invalidates the trained weights.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        max_length=128,
        max_len=None,
        type_tokens: Optional[Iterable[str]] = None,
        amino_acids: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        # ``max_len`` is retained for compatibility with the evaluation code.
        if max_len is not None:
            max_length = max_len

        self.amino_acids = list(
            amino_acids
            if amino_acids is not None
            else [
                "A", "F", "C", "V", "G", "I", "L", "K", "M", "P",
                "S", "T", "W", "Y", "H", "R", "N", "Q", "D", "E",
                "X", "B", "U", "Z", "O",
            ]
        )
        self.type_tokens = list(
            type_tokens
            if type_tokens is not None
            else ["[EPI]", "[HLA]", "[TRA]", "[TRB]"]
        )

        functional_tokens = [
            pad_token,
            bos_token,
            eos_token,
            unk_token,
            mask_token,
        ]
        all_tokens = functional_tokens + self.type_tokens + self.amino_acids

        self.vocab = {token: index for index, token in enumerate(all_tokens)}
        self.id_to_token = {index: token for token, index in self.vocab.items()}
        self.max_length = int(max_length)

        self.special_token_patterns = self.type_tokens + functional_tokens
        patterns = sorted(self.special_token_patterns, key=len, reverse=True)
        pattern = "|".join(re.escape(token) for token in patterns)
        self.special_token_regex = re.compile(f"({pattern})")

        kwargs.pop("model_max_length", None)
        kwargs.pop("padding_side", None)
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            model_max_length=self.max_length,
            padding_side="right",
            clean_up_tokenization_spaces=False,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab.copy()

    def _tokenize(self, text):
        tokens = []
        for segment in self.special_token_regex.split(text):
            if not segment:
                continue
            if segment in self.vocab:
                tokens.append(segment)
            else:
                tokens.extend(segment)
        return tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        return self.id_to_token.get(int(index), self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def encode(
        self,
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        **kwargs,
    ):
        del kwargs
        max_length = self.max_length if max_length is None else int(max_length)

        tokens = self._tokenize(text)
        input_ids = [self._convert_token_to_id(token) for token in tokens]

        if add_special_tokens:
            input_ids = [self.bos_token_id] + input_ids + [self.eos_token_id]

        if truncation and len(input_ids) > max_length:
            if add_special_tokens:
                input_ids = input_ids[: max_length - 1] + [self.eos_token_id]
            else:
                input_ids = input_ids[:max_length]

        attention_mask = [1] * len(input_ids)

        if padding:
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids.extend([self.pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)

        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        elif return_tensors == "np":
            input_ids = np.asarray(input_ids, dtype=np.int64)
            attention_mask = np.asarray(attention_mask, dtype=np.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        del kwargs
        tokens = [self._convert_id_to_token(index) for index in token_ids]
        if skip_special_tokens:
            special = set(self.special_token_patterns)
            tokens = [token for token in tokens if token not in special]
        return self.convert_tokens_to_string(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        filename = (
            f"{filename_prefix}-vocab.json" if filename_prefix else "vocab.json"
        )
        vocab_file = os.path.join(save_directory, filename)
        with open(vocab_file, "w", encoding="utf-8") as handle:
            json.dump(self.vocab, handle, ensure_ascii=False, indent=2)
        return (vocab_file,)

