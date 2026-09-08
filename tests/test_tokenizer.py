from omnitcr.config import ProjectConfig
from omnitcr.tokenizer import AminoAcidTokenizer


def make_tokenizer():
    config = ProjectConfig()
    return AminoAcidTokenizer(
        **config.tokenizer.special_tokens,
        type_tokens=config.tokenizer.type_tokens,
        amino_acids=config.tokenizer.amino_acids,
        max_len=config.tokenizer.max_length,
    )


def test_vocabulary_ids_are_checkpoint_compatible():
    tokenizer = make_tokenizer()
    assert tokenizer.vocab_size == 34
    assert tokenizer.get_vocab()["[PAD]"] == 0
    assert tokenizer.get_vocab()["[BOS]"] == 1
    assert tokenizer.get_vocab()["[EOS]"] == 2
    assert tokenizer.get_vocab()["[EPI]"] == 5
    assert tokenizer.get_vocab()["[HLA]"] == 6
    assert tokenizer.get_vocab()["[TRA]"] == 7
    assert tokenizer.get_vocab()["[TRB]"] == 8
    assert tokenizer.get_vocab()["A"] == 9
    assert tokenizer.get_vocab()["O"] == 33


def test_tokenizer_preserves_component_tokens():
    tokenizer = make_tokenizer()
    encoding = tokenizer.encode("[EPI]AAA[EPI][TRB]CASSF[TRB]")
    assert encoding["input_ids"][0] == tokenizer.bos_token_id
    assert encoding["input_ids"][-2] == tokenizer.get_vocab()["[TRB]"]
    assert encoding["input_ids"][-1] == tokenizer.eos_token_id

