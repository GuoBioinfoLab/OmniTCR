"""Checkpoint-compatible constants used by OmniTCR inference."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TokenizerConfig:
    special_tokens: Dict[str, str] = field(
        default_factory=lambda: {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "mask_token": "[MASK]",
        }
    )
    type_tokens: List[str] = field(
        default_factory=lambda: ["[EPI]", "[HLA]", "[TRA]", "[TRB]"]
    )
    amino_acids: List[str] = field(
        default_factory=lambda: [
            "A", "F", "C", "V", "G", "I", "L", "K", "M", "P",
            "S", "T", "W", "Y", "H", "R", "N", "Q", "D", "E",
            "X", "B", "U", "Z", "O",
        ]
    )
    max_length: int = 128


@dataclass
class ProjectConfig:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)


BINDING_TASKS = ("pm", "pt", "pmt", "pmab")
SUPPORTED_TASKS = BINDING_TASKS + ("repertoire", "generation")

TASK_REQUIRED_COLUMNS = {
    "pm": ("peptide", "mhc"),
    "pt": ("peptide", "trb"),
    "pmt": ("peptide", "mhc", "trb"),
    "pmab": ("peptide", "mhc", "tra", "trb"),
}

TASK_CHECKPOINTS = {
    "pm": "OmniTCR(FFT)_PM/model.safetensors",
    "pt": "OmniTCR(FFT)_PT/model.safetensors",
    "pmt": "OmniTCR(FFT)_PMT/model.safetensors",
    "pmab": "OmniTCR(FFT)_PMAB/model.safetensors",
    "repertoire": "OmniTCR(FFT)_CA/model.safetensors",
}

DEFAULT_REPO_ID = "loveCloud/OmniTCR"
BASE_CONFIG_FILE = "OmniTCR(Base)/config.json"
GENERATION_MODEL_SUBFOLDER = "OmniTCR(SFT)"


@dataclass(frozen=True)
class GenerationConfig:
    """Fixed generation settings used in the manuscript evaluation code."""

    max_new_tokens: int = 40
    num_beams: int = 400
    repetition_penalty: float = 0.7
    length_penalty: float = 0.6
    pmi_candidates: int = 200
    pmi_alpha: float = 0.8
    pmi_scoring_batch_size: int = 32
    null_pmhc_prompt: str = "[EPI][EPI][HLA][HLA][TRB]"
