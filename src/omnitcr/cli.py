"""Command-line interface for CSV-based OmniTCR inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .api import OmniTCR
from .config import BINDING_TASKS, DEFAULT_REPO_ID


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repository containing the OmniTCR checkpoints.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face branch, tag or commit for reproducibility.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 'cuda', 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional local task checkpoint, task directory or repository root.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        default=None,
        help="Optional local OmniTCR(Base) directory, config file or repository root.",
    )
    parser.add_argument(
        "--mhc-pseudosequences-path",
        default=None,
        help="Optional MHC allele-to-pseudosequence CSV file.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face download cache directory.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnitcr",
        description=(
            "OmniTCR inference for binding prediction, repertoire-level "
            "cancer screening and pMHC-conditioned TCR generation."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable informational logging.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    binding = subparsers.add_parser(
        "binding",
        help="Score peptide-MHC-TCR component combinations from a CSV file.",
    )
    binding.add_argument("--task", required=True, choices=BINDING_TASKS)
    binding.add_argument("--input", required=True, help="Input CSV path.")
    binding.add_argument("--output", required=True, help="Output CSV path.")
    binding.add_argument("--batch-size", type=int, default=128)
    _add_model_arguments(binding)

    repertoire = subparsers.add_parser(
        "repertoire",
        help="Return one cancer-associated score per repertoire sample.",
    )
    repertoire.add_argument("--input", required=True, help="Input CSV path.")
    repertoire.add_argument("--output", required=True, help="Output CSV path.")
    repertoire.add_argument("--batch-size", type=int, default=256)
    repertoire.add_argument("--top-k", type=int, default=1000)
    _add_model_arguments(repertoire)

    generation = subparsers.add_parser(
        "generate",
        help="Generate CDR3beta sequences from peptide-MHC allele pairs.",
    )
    generation.add_argument("--mode", choices=("sft", "pmi"), default="sft")
    generation.add_argument("--input", required=True, help="Input CSV path.")
    generation.add_argument("--output", required=True, help="Output CSV path.")
    generation.add_argument("--num-sequences", type=int, default=100)
    _add_model_arguments(generation)

    return parser


def _model_kwargs(args: argparse.Namespace) -> dict:
    return {
        "device": args.device,
        "checkpoint_path": args.checkpoint_path,
        "pretrained_model_path": args.pretrained_model_path,
        "mhc_pseudosequences_path": args.mhc_pseudosequences_path,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "cache_dir": args.cache_dir,
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command == "binding":
        model = OmniTCR(task=args.task, **_model_kwargs(args))
        output_path = model.predict_csv(
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
        )
    elif args.command == "repertoire":
        model = OmniTCR(task="repertoire", **_model_kwargs(args))
        output_path = model.predict_repertoire_csv(
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            top_k=args.top_k,
        )
    else:
        model = OmniTCR(
            task="generation",
            mode=args.mode,
            **_model_kwargs(args),
        )
        output_path = model.generate_csv(
            input_path=args.input,
            output_path=args.output,
            num_sequences=args.num_sequences,
        )

    print(f"Wrote {Path(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
