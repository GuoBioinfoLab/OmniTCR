import torch
from transformers import LlamaConfig

from omnitcr.modeling import (
    TCRLlamaForBinaryClassification,
    TCRLlamaForRepertoireClassification,
)


def tiny_config():
    return LlamaConfig(
        vocab_size=34,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=3,
        num_key_value_heads=3,
        max_position_embeddings=128,
    )


def test_binding_head_matches_fft_architecture():
    model = TCRLlamaForBinaryClassification(tiny_config())
    assert model.classifier[0].in_features == 12
    assert model.classifier[0].out_features == 6
    assert isinstance(model.classifier[1], torch.nn.GELU)
    assert model.classifier[2].p == 0.1
    assert model.classifier[3].out_features == 1


def test_repertoire_head_matches_cancer_architecture():
    model = TCRLlamaForRepertoireClassification(tiny_config())
    assert model.classifier[0].in_features == 12
    assert model.classifier[0].out_features == 1024
    assert isinstance(model.classifier[1], torch.nn.ReLU)
    assert model.classifier[2].out_features == 2
