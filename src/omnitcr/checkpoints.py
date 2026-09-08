"""Resolve and load local or Hugging Face OmniTCR checkpoints."""

from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, LlamaConfig

from .config import (
    BASE_CONFIG_FILE,
    BINDING_TASKS,
    DEFAULT_REPO_ID,
    GENERATION_MODEL_SUBFOLDER,
    TASK_CHECKPOINTS,
)
from .modeling import (
    TCRLlamaForBinaryClassification,
    TCRLlamaForRepertoireClassification,
)


def _resolve_config_path(
    pretrained_model_path,
    repo_id,
    revision,
    token,
    cache_dir,
):
    if pretrained_model_path is not None:
        base_path = Path(pretrained_model_path).expanduser().resolve()
        if base_path.is_dir() and (base_path / "config.json").is_file():
            config_path = base_path / "config.json"
        elif base_path.is_dir():
            config_path = base_path / BASE_CONFIG_FILE
        else:
            config_path = base_path
        if not config_path.is_file():
            raise FileNotFoundError(f"Base config does not exist: {config_path}")
        return config_path

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=BASE_CONFIG_FILE,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
    )


def _resolve_checkpoint_path(
    task,
    checkpoint_path,
    repo_id,
    revision,
    token,
    cache_dir,
):
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser().resolve()
        if path.is_dir() and (path / "model.safetensors").is_file():
            path = path / "model.safetensors"
        elif path.is_dir():
            path = path / TASK_CHECKPOINTS[task]
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        if path.suffix != ".safetensors":
            raise ValueError("Public inference requires a .safetensors checkpoint.")
        return path

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=TASK_CHECKPOINTS[task],
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
    )


def _remove_ddp_prefix(state_dict):
    """Remove the prefix added by DistributedDataParallel, when present."""

    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _infer_local_base_path(checkpoint_path, pretrained_model_path):
    """Reuse a local repository root for the base config when possible."""

    if pretrained_model_path is not None or checkpoint_path is None:
        return pretrained_model_path
    candidate = Path(checkpoint_path).expanduser().resolve()
    if candidate.is_dir() and (candidate / BASE_CONFIG_FILE).is_file():
        return candidate
    return pretrained_model_path


def load_binding_model(
    task: str,
    device: torch.device,
    checkpoint_path: Optional[str | Path] = None,
    pretrained_model_path: Optional[str | Path] = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
):
    task = str(task).strip().lower()
    if task not in BINDING_TASKS:
        raise ValueError(f"Unsupported binding task '{task}'.")

    pretrained_model_path = _infer_local_base_path(
        checkpoint_path, pretrained_model_path
    )
    config_path = _resolve_config_path(
        pretrained_model_path, repo_id, revision, token, cache_dir
    )
    weights_path = _resolve_checkpoint_path(
        task, checkpoint_path, repo_id, revision, token, cache_dir
    )

    config = LlamaConfig.from_json_file(str(config_path))
    model = TCRLlamaForBinaryClassification(config=config)

    state_dict = load_file(str(weights_path), device="cpu")
    state_dict = _remove_ddp_prefix(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The binding checkpoint is incompatible with the published "
            "OmniTCR FFT architecture. Verify that the safetensors file was "
            "created from checkpoint['model_state_dict'] without renaming keys."
        ) from error

    model.to(device)
    model.eval()
    return model


def load_repertoire_model(
    device: torch.device,
    checkpoint_path: Optional[str | Path] = None,
    pretrained_model_path: Optional[str | Path] = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
):
    pretrained_model_path = _infer_local_base_path(
        checkpoint_path, pretrained_model_path
    )
    config_path = _resolve_config_path(
        pretrained_model_path, repo_id, revision, token, cache_dir
    )
    weights_path = _resolve_checkpoint_path(
        "repertoire", checkpoint_path, repo_id, revision, token, cache_dir
    )

    config = LlamaConfig.from_json_file(str(config_path))
    model = TCRLlamaForRepertoireClassification(config=config)

    state_dict = load_file(str(weights_path), device="cpu")
    state_dict = _remove_ddp_prefix(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The repertoire checkpoint is incompatible with the published "
            "OmniTCR repertoire architecture. Verify that the safetensors "
            "file was created from checkpoint['model_state_dict'] without "
            "renaming keys."
        ) from error

    model.to(device)
    model.eval()
    return model


def load_generation_model(
    device: torch.device,
    checkpoint_path: Optional[str | Path] = None,
    pretrained_model_path: Optional[str | Path] = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
):
    """Load the one causal-LM checkpoint shared by SFT and PMI modes."""

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if checkpoint_path is None:
        snapshot_root = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            allow_patterns=[f"{GENERATION_MODEL_SUBFOLDER}/*"],
        )
        model_path = Path(snapshot_root) / GENERATION_MODEL_SUBFOLDER
    else:
        model_path = Path(checkpoint_path).expanduser().resolve()
        if model_path.is_dir() and not (model_path / "config.json").is_file():
            nested_model_path = model_path / GENERATION_MODEL_SUBFOLDER
            if nested_model_path.is_dir():
                model_path = nested_model_path

    if model_path.is_dir():
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
    elif model_path.is_file() and model_path.suffix == ".safetensors":
        config_path = _resolve_config_path(
            pretrained_model_path, repo_id, revision, token, cache_dir
        )
        config = LlamaConfig.from_json_file(str(config_path))
        model = AutoModelForCausalLM.from_config(config)
        state_dict = load_file(str(model_path), device="cpu")
        state_dict = _remove_ddp_prefix(state_dict)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "The generation checkpoint is incompatible with the "
                "OmniTCR causal-LM architecture."
            ) from error
        model.to(dtype=dtype)
    else:
        raise FileNotFoundError(
            "Generation checkpoint must be an HF model directory or a "
            f".safetensors file: {model_path}"
        )

    model.to(device)
    model.eval()
    return model
