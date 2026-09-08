"""Input validation, MHC lookup and component serialization."""

from importlib.resources import as_file, files
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from .config import BINDING_TASKS, TASK_REQUIRED_COLUMNS


class MHCPseudosequenceMapper:
    """Map MHC allele names to the pseudosequences used during training."""

    def __init__(self, mapping_path: Optional[str | Path] = None):
        self.mapping_path = mapping_path
        self.mapping = self._load_mapping()

    def _load_mapping(self) -> dict[str, str]:
        if self.mapping_path is None:
            resource = files("omnitcr.resources").joinpath(
                "mhc_pseudosequences.csv"
            )
            with as_file(resource) as path:
                dataframe = pd.read_csv(path)
        else:
            path = Path(self.mapping_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"MHC pseudosequence table does not exist: {path}"
                )
            dataframe = pd.read_csv(path)

        if {"mhc", "pseudosequence"}.issubset(dataframe.columns):
            allele_column = "mhc"
            sequence_column = "pseudosequence"
        elif {"MHC", "label"}.issubset(dataframe.columns):
            allele_column = "MHC"
            sequence_column = "label"
        else:
            raise ValueError(
                "The MHC table must contain either 'mhc,pseudosequence' "
                "or the original 'MHC,label' columns."
            )

        selected = dataframe[[allele_column, sequence_column]].dropna().copy()
        selected[allele_column] = (
            selected[allele_column].astype(str).str.strip().str.upper()
        )
        selected[sequence_column] = (
            selected[sequence_column].astype(str).str.strip().str.upper()
        )

        duplicated = selected[allele_column].duplicated(keep=False)
        if duplicated.any():
            duplicates = sorted(selected.loc[duplicated, allele_column].unique())
            raise ValueError(
                "Duplicate MHC alleles in the pseudosequence table: "
                + ", ".join(duplicates[:10])
            )

        if selected.empty:
            raise ValueError("The MHC pseudosequence table is empty.")

        return selected.set_index(allele_column)[sequence_column].to_dict()

    def convert(self, mhc: str) -> str:
        if not isinstance(mhc, str) or not mhc.strip():
            raise ValueError("mhc must be a non-empty allele name.")

        normalized = mhc.strip().upper()
        try:
            return self.mapping[normalized]
        except KeyError as error:
            raise ValueError(
                f"Unsupported MHC allele '{mhc}'. The allele is absent from "
                "the configured pseudosequence table."
            ) from error


class BindingInputBuilder:
    """Validate component sequences and construct exact OmniTCR inputs."""

    def __init__(self, task, tokenizer, mhc_mapper, max_length=128):
        task = str(task).strip().lower()
        if task not in BINDING_TASKS:
            raise ValueError(
                f"Unsupported binding task '{task}'. Choose from {BINDING_TASKS}."
            )

        self.task = task
        self.tokenizer = tokenizer
        self.mhc_mapper = mhc_mapper
        self.max_length = int(max_length)
        self.allowed_amino_acids = set(tokenizer.amino_acids)

    @property
    def required_columns(self):
        return TASK_REQUIRED_COLUMNS[self.task]

    def _sequence(self, value, name):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty sequence.")

        sequence = value.strip().upper()
        invalid = sorted(set(sequence) - self.allowed_amino_acids)
        if invalid:
            raise ValueError(
                f"{name} contains unsupported characters: " + ", ".join(invalid)
            )
        return sequence

    @staticmethod
    def _wrap(token, sequence):
        return f"[{token}]{sequence}[{token}]"

    def format(self, record: Mapping[str, object]) -> str:
        missing = [name for name in self.required_columns if name not in record]
        if missing:
            raise ValueError("Missing inputs: " + ", ".join(missing))

        peptide = self._sequence(record["peptide"], "peptide")
        formatted = self._wrap("EPI", peptide)

        if self.task in {"pm", "pmt", "pmab"}:
            mhc_pseudosequence = self.mhc_mapper.convert(record["mhc"])
            mhc_pseudosequence = self._sequence(
                mhc_pseudosequence, "mhc pseudosequence"
            )
            formatted += self._wrap("HLA", mhc_pseudosequence)

        if self.task == "pmab":
            tra = self._sequence(record["tra"], "tra")
            formatted += self._wrap("TRA", tra)

        if self.task in {"pt", "pmt", "pmab"}:
            trb = self._sequence(record["trb"], "trb")
            formatted += self._wrap("TRB", trb)

        return formatted

    def encode(self, record: Mapping[str, object]):
        formatted = self.format(record)
        encoding = self.tokenizer.encode(
            formatted,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            max_length=self.max_length,
        )

        token_count = len(encoding["input_ids"])
        if token_count > self.max_length:
            raise ValueError(
                f"Encoded input contains {token_count} tokens, exceeding the "
                f"maximum of {self.max_length}. The input was not truncated."
            )

        return encoding


class RepertoireInputBuilder:
    """Construct the exact single-TRB inputs used by the cancer model."""

    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.allowed_amino_acids = set(tokenizer.amino_acids)

    def format(self, trb) -> str:
        if not isinstance(trb, str) or not trb.strip():
            raise ValueError("trb must be a non-empty sequence.")

        sequence = trb.strip().upper()
        invalid = sorted(set(sequence) - self.allowed_amino_acids)
        if invalid:
            raise ValueError(
                "trb contains unsupported characters: " + ", ".join(invalid)
            )
        return f"[TRB]{sequence}[TRB]"

    def encode(self, trb):
        formatted = self.format(trb)
        encoding = self.tokenizer.encode(
            formatted,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            max_length=self.max_length,
        )

        token_count = len(encoding["input_ids"])
        if token_count > self.max_length:
            raise ValueError(
                f"Encoded input contains {token_count} tokens, exceeding the "
                f"maximum of {self.max_length}. The input was not truncated."
            )
        return encoding


def prepare_repertoire_dataframe(
    dataframe: pd.DataFrame,
    trb_column: str,
    sample_id_column: str,
    weight_column: Optional[str] = None,
    top_k: int = 1000,
) -> pd.DataFrame:
    """Reproduce the evaluation script's row selection and weight handling.

    Duplicate TRB rows are deliberately retained. If weights are present,
    rows are sorted by sample and descending weight, the top ``k`` rows per
    sample are retained, and weights are normalized within each sample.
    Without weights, all rows are retained, matching the evaluation script.
    """

    if dataframe.empty:
        raise ValueError("The repertoire input is empty.")

    working = pd.DataFrame(
        {
            "trb": dataframe[trb_column],
            "sample_id": dataframe[sample_id_column],
        }
    )

    if working["sample_id"].isna().any():
        raise ValueError("sample_id contains missing values.")
    working["sample_id"] = working["sample_id"].astype(str)
    if working["sample_id"].str.strip().eq("").any():
        raise ValueError("sample_id contains empty values.")

    if weight_column is not None:
        weights = pd.to_numeric(dataframe[weight_column], errors="coerce")
        if weights.isna().any():
            raise ValueError("weight contains missing or non-numeric values.")
        if (weights < 0).any():
            raise ValueError("Weights must be non-negative.")

        working["weight"] = weights.astype(float)
        if top_k > 0:
            working = working.sort_values(
                by=["sample_id", "weight"],
                ascending=[True, False],
            )
            working = working.groupby("sample_id").head(int(top_k))
            working = working.reset_index(drop=True)

        working["weight"] = working.groupby("sample_id")["weight"].transform(
            lambda values: (
                values / values.sum() if values.sum() > 0 else values
            )
        )
    else:
        working["weight"] = 1.0

    return working


class GenerationInputBuilder:
    """Construct pMHC prompts for conditional CDR3beta generation."""

    def __init__(self, tokenizer, mhc_mapper, max_length=128):
        self.tokenizer = tokenizer
        self.mhc_mapper = mhc_mapper
        self.max_length = int(max_length)
        self.allowed_amino_acids = set(tokenizer.amino_acids)

    def _sequence(self, value, name):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty sequence.")
        sequence = value.strip().upper()
        invalid = sorted(set(sequence) - self.allowed_amino_acids)
        if invalid:
            raise ValueError(
                f"{name} contains unsupported characters: " + ", ".join(invalid)
            )
        return sequence

    def format(self, peptide, mhc) -> str:
        peptide = self._sequence(peptide, "peptide")
        pseudosequence = self.mhc_mapper.convert(mhc)
        pseudosequence = self._sequence(pseudosequence, "mhc pseudosequence")
        return (
            f"[EPI]{peptide}[EPI]"
            f"[HLA]{pseudosequence}[HLA]"
            "[TRB]"
        )

    def encode_prompt(self, peptide, mhc):
        prompt = self.format(peptide, mhc)
        encoding = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            max_length=self.max_length,
        )
        input_ids = [self.tokenizer.bos_token_id] + encoding["input_ids"]
        if len(input_ids) > self.max_length:
            raise ValueError(
                f"Generation prompt contains {len(input_ids)} tokens, "
                f"exceeding the maximum of {self.max_length}."
            )
        return prompt, input_ids
