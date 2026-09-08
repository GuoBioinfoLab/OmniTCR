"""Check that every file required by the public API exists on Hugging Face."""

from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download

from omnitcr.config import (
    BASE_CONFIG_FILE,
    DEFAULT_REPO_ID,
    GENERATION_MODEL_SUBFOLDER,
    TASK_CHECKPOINTS,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    required = [BASE_CONFIG_FILE, *TASK_CHECKPOINTS.values()]
    required.extend(
        [
            f"{GENERATION_MODEL_SUBFOLDER}/config.json",
            f"{GENERATION_MODEL_SUBFOLDER}/model.safetensors",
        ]
    )

    for filename in required:
        hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
        print(f"OK  {filename}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

