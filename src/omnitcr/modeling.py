"""Checkpoint-compatible OmniTCR binding architecture."""

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM


class TCRLlamaForCausalLM(LlamaForCausalLM):
    """The unchanged Llama causal-language-model backbone used by OmniTCR."""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)


class TCRLlamaForBinaryClassification(nn.Module):
    """Full-fine-tuning architecture used by the binding checkpoints."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.llama_model = TCRLlamaForCausalLM(config)
        hidden_size = self.llama_model.config.hidden_size

        # Preserve the exact Sequential indices and dimensions used for FFT.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.llama_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

        last_hidden_state = outputs.last_hidden_state
        sequence_lengths = attention_mask.long().sum(dim=1)

        if torch.any(sequence_lengths < 2):
            raise ValueError("Every encoded sequence must contain at least two tokens.")

        # The penultimate non-padding token is the final closing component token.
        token_indices = sequence_lengths - 2
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        representations = last_hidden_state[batch_indices, token_indices]

        return self.classifier(representations).squeeze(-1)


class TCRLlamaForRepertoireClassification(nn.Module):
    """Exact two-logit architecture used for repertoire screening."""

    def __init__(self, config: LlamaConfig, mlp_hidden_size=1024):
        super().__init__()
        self.llama_model = TCRLlamaForCausalLM(config)
        hidden_size = self.llama_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 2),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.llama_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

        last_hidden_state = outputs.last_hidden_state
        sequence_lengths = attention_mask.long().sum(dim=1)
        if torch.any(sequence_lengths < 2):
            raise ValueError("Every encoded sequence must contain at least two tokens.")

        token_indices = sequence_lengths - 2
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        representations = last_hidden_state[batch_indices, token_indices]
        return self.classifier(representations)
