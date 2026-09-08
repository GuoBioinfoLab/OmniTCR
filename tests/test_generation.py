from types import SimpleNamespace

import torch

from omnitcr.config import GenerationConfig, ProjectConfig
from omnitcr.generation import (
    TCRGenerator,
    is_valid_pmi_cdr3,
    is_valid_sft_cdr3,
)
from omnitcr.tokenizer import AminoAcidTokenizer


def test_sft_and_pmi_validity_reproduce_evaluation_code():
    assert is_valid_sft_cdr3("CASSLGRASNQPQHF")
    assert is_valid_pmi_cdr3("CASSLGRASNQPQHF")

    # SFT evaluation checked length and terminal residues but not alphabet.
    assert is_valid_sft_cdr3("CASSXAF")
    assert not is_valid_pmi_cdr3("CASSXAF")

    assert not is_valid_sft_cdr3("ASSLGRASNQPQHF")
    assert not is_valid_sft_cdr3("CASSLGRASNQPQHA")


class FakeBuilder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_prompt(self, peptide, mhc):
        del peptide, mhc
        prompt = "[EPI]AAA[EPI][HLA]AAA[HLA][TRB]"
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        return prompt, [self.tokenizer.bos_token_id] + encoded["input_ids"]


class FakeGenerationModel(torch.nn.Module):
    def __init__(self, tokenizer, generated):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_position_embeddings=128)
        self.tokenizer = tokenizer
        self.generated = generated
        self.generation_kwargs = None
        self.forward_attention_masks = []

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        prefix = kwargs["input_ids"][0].tolist()
        sequences = []
        for sequence in self.generated[: kwargs["num_return_sequences"]]:
            suffix = self.tokenizer.encode(
                sequence, add_special_tokens=False
            )["input_ids"]
            sequences.append(prefix + suffix)
        return torch.tensor(sequences, dtype=torch.long)

    def forward(self, input_ids, attention_mask, return_dict=True):
        assert return_dict
        self.forward_attention_masks.append(attention_mask.detach().cpu())
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.tokenizer.vocab_size,
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits)


def make_generator(mode="sft", generated=None):
    config = ProjectConfig()
    tokenizer = AminoAcidTokenizer(
        **config.tokenizer.special_tokens,
        type_tokens=config.tokenizer.type_tokens,
        amino_acids=config.tokenizer.amino_acids,
        max_len=config.tokenizer.max_length,
    )
    model = FakeGenerationModel(
        tokenizer,
        generated or ["CASSF", "CASSF", "AASSF"],
    )
    generator = TCRGenerator(
        model=model,
        tokenizer=tokenizer,
        input_builder=FakeBuilder(tokenizer),
        mode=mode,
    )
    return generator, model


def test_sft_beam_arguments_attention_mask_and_duplicate_behavior():
    generator, model = make_generator()
    sequences = generator.generate("AAA", "HLA-A*01:01", num_sequences=3)

    # CASSF is too short and therefore filtered; duplicates are not deduplicated.
    assert sequences == []
    kwargs = model.generation_kwargs
    assert torch.equal(kwargs["attention_mask"], torch.ones_like(kwargs["input_ids"]))
    assert kwargs["max_new_tokens"] == 40
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 400
    assert kwargs["num_return_sequences"] == 3
    assert kwargs["early_stopping"] is True
    assert kwargs["repetition_penalty"] == 0.7
    assert kwargs["length_penalty"] == 0.6


def test_sft_preserves_valid_duplicates_and_does_not_refill():
    generator, _ = make_generator(
        generated=["CASSAAF", "CASSAAF", "AASSAAF"]
    )
    assert generator.generate("AAA", "HLA-A*01:01", 3) == [
        "CASSAAF",
        "CASSAAF",
    ]


def test_pmi_scoring_uses_padded_attention_masks():
    generator, model = make_generator(mode="pmi")
    scores = generator._score_tcrs_avg_logp(
        "[EPI]AAA[EPI][HLA]AAA[HLA][TRB]",
        ["CASSAAF", "CASSLONGERF"],
        batch_size=2,
    )
    assert len(scores) == 2
    mask = model.forward_attention_masks[-1]
    assert mask.shape[0] == 2
    assert (mask[0] == 0).any()
    assert not (mask[1] == 0).any()


class StubPMIGenerator(TCRGenerator):
    def _pmi_candidates(self, input_ids):
        del input_ids
        return [
            {"tcr": "CASSAAAF", "beam_rank": 1},
            {"tcr": "CASSDDDF", "beam_rank": 2},
            {"tcr": "CASSCCCF", "beam_rank": 3},
        ]

    def _score_tcrs_avg_logp(self, prompt, tcrs, batch_size=32):
        del tcrs, batch_size
        if prompt == self.settings.null_pmhc_prompt:
            return [-2.0, -1.0, -2.75]
        return [-1.0, -0.9, -1.8]


def test_pmi_formula_and_ranking_order():
    base, _ = make_generator(mode="pmi")
    generator = StubPMIGenerator(
        model=base.model,
        tokenizer=base.tokenizer,
        input_builder=base.input_builder,
        mode="pmi",
        settings=GenerationConfig(),
    )
    # PMI scores are 0.6, -0.1 and 0.4.
    assert generator._pmi_generate("target", [1], 3) == [
        "CASSAAAF",
        "CASSCCCF",
        "CASSDDDF",
    ]
