import pytest

from omnitcr.config import ProjectConfig
import pandas as pd

from omnitcr.preprocessing import (
    BindingInputBuilder,
    GenerationInputBuilder,
    MHCPseudosequenceMapper,
    RepertoireInputBuilder,
    prepare_repertoire_dataframe,
)
from omnitcr.tokenizer import AminoAcidTokenizer


PSEUDO = "YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY"


def make_builder(task):
    config = ProjectConfig()
    tokenizer = AminoAcidTokenizer(
        **config.tokenizer.special_tokens,
        type_tokens=config.tokenizer.type_tokens,
        amino_acids=config.tokenizer.amino_acids,
        max_len=config.tokenizer.max_length,
    )
    return BindingInputBuilder(
        task=task,
        tokenizer=tokenizer,
        mhc_mapper=MHCPseudosequenceMapper(),
        max_length=128,
    )


def test_pm_format():
    builder = make_builder("pm")
    assert builder.format({"peptide": "ILGSLGLRK", "mhc": "HLA-A*01:01"}) == (
        f"[EPI]ILGSLGLRK[EPI][HLA]{PSEUDO}[HLA]"
    )


def test_pt_format():
    builder = make_builder("pt")
    assert builder.format(
        {"peptide": "ALSKGVHFV", "trb": "CASSLWGSEAFF"}
    ) == "[EPI]ALSKGVHFV[EPI][TRB]CASSLWGSEAFF[TRB]"


def test_pmt_format():
    builder = make_builder("pmt")
    assert builder.format(
        {
            "peptide": "LLQCTQQAV",
            "mhc": "HLA-A*01:01",
            "trb": "CASSQDRGIGYGYTF",
        }
    ) == (
        f"[EPI]LLQCTQQAV[EPI][HLA]{PSEUDO}[HLA]"
        "[TRB]CASSQDRGIGYGYTF[TRB]"
    )


def test_pmab_format():
    builder = make_builder("pmab")
    assert builder.format(
        {
            "peptide": "YLLAIFSGL",
            "mhc": "HLA-A*01:01",
            "tra": "CAPVSGGGADGLTF",
            "trb": "CASSLPDRGGTKNIQYF",
        }
    ) == (
        f"[EPI]YLLAIFSGL[EPI][HLA]{PSEUDO}[HLA]"
        "[TRA]CAPVSGGGADGLTF[TRA][TRB]CASSLPDRGGTKNIQYF[TRB]"
    )


def test_unknown_mhc_raises_instead_of_dropping_row():
    builder = make_builder("pm")
    with pytest.raises(ValueError, match="Unsupported MHC allele"):
        builder.format({"peptide": "ILGSLGLRK", "mhc": "HLA-Z*99:99"})


def test_original_mhc_label_schema_is_supported(tmp_path):
    table = tmp_path / "mhc.csv"
    pd.DataFrame(
        {"MHC": ["HLA-A*01:01"], "label": [PSEUDO]}
    ).to_csv(table, index=False)
    mapper = MHCPseudosequenceMapper(table)
    assert mapper.convert("hla-a*01:01") == PSEUDO


def test_overlength_input_is_not_silently_truncated():
    builder = make_builder("pt")
    with pytest.raises(ValueError, match="was not truncated"):
        builder.encode({"peptide": "A" * 110, "trb": "CASSLWGSEAFF"})


def test_repertoire_builder_adds_paired_trb_tokens():
    tokenizer = make_builder("pt").tokenizer
    builder = RepertoireInputBuilder(tokenizer=tokenizer)
    assert builder.format("CARSVGGNGGNTEAFF") == (
        "[TRB]CARSVGGNGGNTEAFF[TRB]"
    )
    encoding = builder.encode("CARSVGGNGGNTEAFF")
    assert encoding["input_ids"][-2] == tokenizer.get_vocab()["[TRB]"]
    assert encoding["input_ids"][-1] == tokenizer.eos_token_id


def test_repertoire_preparation_retains_duplicates_and_selects_top_k():
    dataframe = pd.DataFrame(
        {
            "trb": ["CASSA", "CASSA", "CASSB", "CASSC"],
            "sample_id": ["sample_1"] * 4,
            "weight": [9, 4, 2, 1],
        }
    )
    result = prepare_repertoire_dataframe(
        dataframe,
        trb_column="trb",
        sample_id_column="sample_id",
        weight_column="weight",
        top_k=3,
    )
    assert result["trb"].tolist() == ["CASSA", "CASSA", "CASSB"]
    assert len(result) == 3
    assert result["weight"].sum() == pytest.approx(1.0)


def test_repertoire_without_weights_retains_all_rows():
    dataframe = pd.DataFrame(
        {
            "trb": ["CASSA", "CASSB", "CASSC"],
            "sample_id": ["sample_1"] * 3,
        }
    )
    result = prepare_repertoire_dataframe(
        dataframe,
        trb_column="trb",
        sample_id_column="sample_id",
        weight_column=None,
        top_k=1,
    )
    assert len(result) == 3
    assert result["weight"].tolist() == [1.0, 1.0, 1.0]


def test_generation_prompt_uses_allele_mapping_and_open_trb_token():
    binding_builder = make_builder("pm")
    builder = GenerationInputBuilder(
        tokenizer=binding_builder.tokenizer,
        mhc_mapper=MHCPseudosequenceMapper(),
    )
    assert builder.format("GADGVGKSA", "HLA-A*01:01") == (
        f"[EPI]GADGVGKSA[EPI][HLA]{PSEUDO}[HLA][TRB]"
    )
    prompt, input_ids = builder.encode_prompt("GADGVGKSA", "HLA-A*01:01")
    assert prompt.endswith("[TRB]")
    assert input_ids[0] == binding_builder.tokenizer.bos_token_id
    assert input_ids[-1] == binding_builder.tokenizer.get_vocab()["[TRB]"]
