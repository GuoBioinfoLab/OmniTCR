import torch

from omnitcr.datasets import DynamicPaddingCollator, EncodedSequenceDataset


def test_encoded_sequence_dataset_and_dynamic_padding():
    encodings = [
        {"input_ids": [1, 5, 2], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 8, 9, 2], "attention_mask": [1, 1, 1, 1]},
    ]
    dataset = EncodedSequenceDataset(encodings)
    assert len(dataset) == 2
    assert dataset[0] == encodings[0]

    batch = DynamicPaddingCollator(pad_token_id=0)(dataset.encodings)
    assert torch.equal(
        batch["input_ids"],
        torch.tensor([[1, 5, 2, 0], [1, 8, 9, 2]]),
    )
    assert torch.equal(
        batch["attention_mask"],
        torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
    )
