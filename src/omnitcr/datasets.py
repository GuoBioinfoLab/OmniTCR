"""Small inference datasets and dynamic-padding utilities."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class EncodedSequenceDataset(Dataset):
    """A thin dataset over tokenizer output dictionaries."""

    def __init__(self, encodings: Sequence[dict]):
        self.encodings = list(encodings)

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, index: int) -> dict:
        return self.encodings[index]


class DynamicPaddingCollator:
    """Right-pad tokenized sequences and construct attention masks."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, encodings: Sequence[dict]) -> dict[str, torch.Tensor]:
        if not encodings:
            raise ValueError("Cannot collate an empty batch.")

        maximum = max(len(item["input_ids"]) for item in encodings)
        input_ids = []
        attention_masks = []

        for item in encodings:
            ids = list(item["input_ids"])
            mask = list(item["attention_mask"])
            if len(ids) != len(mask):
                raise ValueError(
                    "input_ids and attention_mask must have equal lengths."
                )
            pad_length = maximum - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_length)
            attention_masks.append(mask + [0] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),
        }
