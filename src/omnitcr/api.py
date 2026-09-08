"""Public sequence and CSV inference API for OmniTCR models."""

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .checkpoints import (
    load_binding_model,
    load_generation_model,
    load_repertoire_model,
)
from .config import BINDING_TASKS, DEFAULT_REPO_ID, ProjectConfig, SUPPORTED_TASKS
from .datasets import DynamicPaddingCollator
from .generation import TCRGenerator
from .preprocessing import (
    BindingInputBuilder,
    GenerationInputBuilder,
    MHCPseudosequenceMapper,
    RepertoireInputBuilder,
    prepare_repertoire_dataframe,
)
from .tokenizer import AminoAcidTokenizer


class OmniTCR:
    """Run OmniTCR binding, repertoire or generation inference."""

    def __init__(
        self,
        task: str,
        device: Optional[str] = None,
        checkpoint_path: Optional[str | Path] = None,
        pretrained_model_path: Optional[str | Path] = None,
        mhc_pseudosequences_path: Optional[str | Path] = None,
        repo_id: str = DEFAULT_REPO_ID,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        max_length: int = 128,
        mode: str = "sft",
    ):
        task = str(task).strip().lower()
        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{task}'. Choose from {SUPPORTED_TASKS}."
            )

        self.task = task
        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer.")
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.repo_id = repo_id
        self.revision = revision

        # Preserve the tokenizer initialization pattern used in evaluation.
        project_config = ProjectConfig()
        project_config.tokenizer.max_length = int(max_length)
        self.tokenizer = AminoAcidTokenizer(
            **project_config.tokenizer.special_tokens,
            type_tokens=project_config.tokenizer.type_tokens,
            amino_acids=project_config.tokenizer.amino_acids,
            max_len=project_config.tokenizer.max_length,
        )
        self.collator = DynamicPaddingCollator(self.tokenizer.pad_token_id)

        self.mode = str(mode).strip().lower()

        if self.task in BINDING_TASKS:
            mhc_mapper = (
                MHCPseudosequenceMapper(mhc_pseudosequences_path)
                if self.task in {"pm", "pmt", "pmab"}
                else None
            )
            self.input_builder = BindingInputBuilder(
                task=self.task,
                tokenizer=self.tokenizer,
                mhc_mapper=mhc_mapper,
                max_length=max_length,
            )
            self.model = load_binding_model(
                task=self.task,
                device=self.device,
                checkpoint_path=checkpoint_path,
                pretrained_model_path=pretrained_model_path,
                repo_id=repo_id,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
            )
        elif self.task == "repertoire":
            self.input_builder = RepertoireInputBuilder(
                tokenizer=self.tokenizer,
                max_length=max_length,
            )
            self.model = load_repertoire_model(
                device=self.device,
                checkpoint_path=checkpoint_path,
                pretrained_model_path=pretrained_model_path,
                repo_id=repo_id,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
            )
        else:
            if self.mode not in {"sft", "pmi"}:
                raise ValueError("Generation mode must be 'sft' or 'pmi'.")
            mhc_mapper = MHCPseudosequenceMapper(mhc_pseudosequences_path)
            self.input_builder = GenerationInputBuilder(
                tokenizer=self.tokenizer,
                mhc_mapper=mhc_mapper,
                max_length=max_length,
            )
            self.model = load_generation_model(
                device=self.device,
                checkpoint_path=checkpoint_path,
                pretrained_model_path=pretrained_model_path,
                repo_id=repo_id,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
            )
            self.generator = TCRGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                input_builder=self.input_builder,
                mode=self.mode,
            )

    def _score_binding_records(self, records, batch_size, csv_row_offset=None):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        scores = []
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                batch_records = records[start : start + batch_size]
                encodings = []
                for offset, record in enumerate(batch_records):
                    try:
                        encodings.append(self.input_builder.encode(record))
                    except Exception as error:
                        if csv_row_offset is None:
                            raise ValueError(f"Invalid input: {error}") from error
                        row_number = start + offset + csv_row_offset
                        raise ValueError(
                            f"Invalid input at CSV row {row_number}: {error}"
                        ) from error

                batch = self.collator(encodings)
                logits = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                probabilities = torch.sigmoid(logits)
                scores.extend(
                    probabilities.detach().float().cpu().numpy().tolist()
                )

        return [float(score) for score in scores]

    def _score_repertoire_trbs(self, trb_sequences, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        scores = []
        with torch.inference_mode():
            for start in range(0, len(trb_sequences), batch_size):
                batch_sequences = trb_sequences[start : start + batch_size]
                encodings = []
                for offset, trb in enumerate(batch_sequences):
                    try:
                        encodings.append(self.input_builder.encode(trb))
                    except Exception as error:
                        sequence_number = start + offset + 1
                        raise ValueError(
                            f"Invalid TRB sequence {sequence_number}: {error}"
                        ) from error

                batch = self.collator(encodings)
                logits = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                probabilities = torch.softmax(logits, dim=1)[:, 1]
                scores.extend(
                    probabilities.detach().float().cpu().numpy().tolist()
                )

        return [float(score) for score in scores]

    def predict(
        self,
        peptide: str,
        *,
        mhc: Optional[str] = None,
        tra: Optional[str] = None,
        trb: Optional[str] = None,
    ) -> float:
        """Return one sigmoid binding score for one biological input."""

        if self.task not in BINDING_TASKS:
            raise RuntimeError(
                "predict() is available only for binding tasks."
            )

        record = {
            "peptide": peptide,
            "mhc": mhc,
            "tra": tra,
            "trb": trb,
        }
        return self._score_binding_records([record], batch_size=1)[0]

    def predict_batch(
        self,
        records: Sequence[Mapping[str, object]],
        batch_size: int = 128,
    ) -> list[float]:
        """Return sigmoid binding scores for structured input records.

        Each record must contain the columns required by the selected binding
        task. Record order is preserved.
        """

        if self.task not in BINDING_TASKS:
            raise RuntimeError(
                "predict_batch() is available only for binding tasks."
            )
        if isinstance(records, Mapping):
            raise TypeError("records must be a sequence of mappings.")
        records = list(records)
        if not records:
            raise ValueError("records cannot be empty.")
        if not all(isinstance(record, Mapping) for record in records):
            raise TypeError("Every item in records must be a mapping.")
        return self._score_binding_records(records, batch_size=batch_size)

    def predict_repertoire(
        self,
        trb_sequences: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        batch_size: int = 256,
        top_k: int = 1000,
    ) -> float:
        """Return the evaluation-compatible median score for one sample."""

        if self.task != "repertoire":
            raise RuntimeError(
                "predict_repertoire() requires task='repertoire'."
            )
        if isinstance(trb_sequences, str):
            raise TypeError("trb_sequences must be a sequence of strings.")

        trb_sequences = list(trb_sequences)
        if not trb_sequences:
            raise ValueError("trb_sequences cannot be empty.")

        dataframe = pd.DataFrame(
            {
                "trb": trb_sequences,
                "sample_id": ["sample"] * len(trb_sequences),
            }
        )
        weight_column = None
        if weights is not None:
            weights = list(weights)
            if len(weights) != len(trb_sequences):
                raise ValueError(
                    "weights and trb_sequences must have the same length."
                )
            dataframe["weight"] = weights
            weight_column = "weight"

        working = prepare_repertoire_dataframe(
            dataframe=dataframe,
            trb_column="trb",
            sample_id_column="sample_id",
            weight_column=weight_column,
            top_k=top_k,
        )
        sequence_scores = self._score_repertoire_trbs(
            working["trb"].tolist(),
            batch_size=batch_size,
        )
        return float(np.median(sequence_scores))

    def generate(
        self,
        peptide: str,
        mhc: str,
        num_sequences: int = 100,
    ) -> list[str]:
        """Generate CDR3beta sequences with SFT or PMI inference."""

        if self.task != "generation":
            raise RuntimeError("generate() requires task='generation'.")
        return self.generator.generate(
            peptide=peptide,
            mhc=mhc,
            num_sequences=num_sequences,
        )

    def predict_csv(
        self,
        input_path: str | Path,
        output_path: str | Path,
        batch_size: Optional[int] = None,
    ) -> Path:
        """Run task-appropriate CSV inference and write prediction scores."""

        if self.task == "generation":
            return self.generate_csv(
                input_path=input_path,
                output_path=output_path,
                num_sequences=100,
            )

        if self.task == "repertoire":
            return self.predict_repertoire_csv(
                input_path=input_path,
                output_path=output_path,
                batch_size=256 if batch_size is None else batch_size,
            )

        if batch_size is None:
            batch_size = 128

        input_path = Path(input_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()

        if not input_path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

        dataframe = pd.read_csv(input_path)
        if dataframe.empty:
            raise ValueError(f"Input CSV is empty: {input_path}")

        normalized_columns = {
            str(column).strip().lower(): column for column in dataframe.columns
        }
        missing = [
            column
            for column in self.input_builder.required_columns
            if column not in normalized_columns
        ]
        if missing:
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(missing)
            )

        records = []
        for _, row in dataframe.iterrows():
            records.append(
                {
                    column: row[normalized_columns[column]]
                    for column in self.input_builder.required_columns
                }
            )

        scores = self._score_binding_records(
            records,
            batch_size=batch_size,
            csv_row_offset=2,
        )
        result = dataframe.copy()
        result["score"] = scores

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        return output_path

    def generate_csv(
        self,
        input_path: str | Path,
        output_path: str | Path,
        num_sequences: int = 100,
    ) -> Path:
        """Generate TCR rows from peptide and MHC-allele CSV inputs."""

        if self.task != "generation":
            raise RuntimeError("generate_csv() requires task='generation'.")

        input_path = Path(input_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

        dataframe = pd.read_csv(input_path)
        if dataframe.empty:
            raise ValueError(f"Input CSV is empty: {input_path}")

        normalized_columns = {
            str(column).strip().lower(): column for column in dataframe.columns
        }
        peptide_column = normalized_columns.get(
            "peptide", normalized_columns.get("epitope")
        )
        mhc_column = normalized_columns.get("mhc")
        if peptide_column is None or mhc_column is None:
            raise ValueError(
                "Generation CSV requires 'peptide,mhc' columns. "
                "'epitope' is accepted as an alias for 'peptide'."
            )

        output_rows = []
        for row_index, row in dataframe.iterrows():
            peptide = row[peptide_column]
            mhc = row[mhc_column]
            try:
                generated = self.generate(
                    peptide=peptide,
                    mhc=mhc,
                    num_sequences=num_sequences,
                )
            except Exception as error:
                raise ValueError(
                    f"Invalid generation input at CSV row {row_index + 2}: {error}"
                ) from error

            for rank, trb in enumerate(generated, start=1):
                output_rows.append(
                    {
                        "peptide": str(peptide).strip().upper(),
                        "mhc": str(mhc).strip().upper(),
                        "rank": rank,
                        "generated_trb": trb,
                    }
                )

        result = pd.DataFrame(
            output_rows,
            columns=["peptide", "mhc", "rank", "generated_trb"],
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        return output_path

    def predict_repertoire_csv(
        self,
        input_path: str | Path,
        output_path: str | Path,
        batch_size: int = 256,
        top_k: int = 1000,
    ) -> Path:
        """Return one median class-1 score per repertoire sample."""

        if self.task != "repertoire":
            raise RuntimeError(
                "predict_repertoire_csv() requires task='repertoire'."
            )

        input_path = Path(input_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

        dataframe = pd.read_csv(input_path)
        if dataframe.empty:
            raise ValueError(f"Input CSV is empty: {input_path}")

        normalized_columns = {
            str(column).strip().lower(): column for column in dataframe.columns
        }
        missing = [
            column for column in ("trb", "sample_id")
            if column not in normalized_columns
        ]
        if missing:
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(missing)
            )

        weight_column = normalized_columns.get("weight")
        working = prepare_repertoire_dataframe(
            dataframe=dataframe,
            trb_column=normalized_columns["trb"],
            sample_id_column=normalized_columns["sample_id"],
            weight_column=weight_column,
            top_k=top_k,
        )

        sequence_scores = self._score_repertoire_trbs(
            working["trb"].tolist(),
            batch_size=batch_size,
        )
        working = working.copy()
        working["_score"] = sequence_scores

        result = (
            working.groupby("sample_id", sort=False)["_score"]
            .median()
            .reset_index(name="score")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        return output_path
