"""Convert an OmniTCR training checkpoint from PyTorch to safetensors."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import save_file


def extract_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The checkpoint does not contain a state dictionary.")

    state_dict = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Non-tensor state-dict value found at key '{key}'.")
        normalized_key = key[7:] if key.startswith("module.") else key
        state_dict[normalized_key] = value.detach().cpu().contiguous()
    if not state_dict:
        raise ValueError("The checkpoint state dictionary is empty.")
    return state_dict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input .pt checkpoint.")
    parser.add_argument("output", type=Path, help="Output .safetensors file.")
    args = parser.parse_args()

    if args.input.suffix != ".pt" or not args.input.is_file():
        raise FileNotFoundError(f"Input .pt file not found: {args.input}")
    if args.output.suffix != ".safetensors":
        raise ValueError("Output path must end with '.safetensors'.")

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(args.output))
    print(f"Saved {len(state_dict):,} tensors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
