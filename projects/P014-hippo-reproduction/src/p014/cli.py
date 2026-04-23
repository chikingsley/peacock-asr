from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from p014.config import ScenarioName, TrainingRunSettings, load_named_config
from p014.features import (
    extract_gop_for_split,
    extract_ssl_utterance_for_split,
    merge_gop_shards,
)
from p014.training import train_hippo


def _add_training_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="One or more integer seeds (space separated).",
    )
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-test-examples", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--modernbert-model")
    parser.add_argument("--use-cono", dest="use_cono", action="store_true", default=None)
    parser.add_argument("--no-use-cono", dest="use_cono", action="store_false", default=None)
    parser.add_argument("--lambda-phone", type=float)
    parser.add_argument("--lambda-word", type=float)
    parser.add_argument("--lambda-utterance", type=float)
    parser.add_argument("--lambda-cono", type=float)
    parser.add_argument("--lambda-diversity", type=float)
    parser.add_argument("--lambda-tightness", type=float)
    parser.add_argument("--trackio-project")
    parser.add_argument("--trackio-space")
    parser.add_argument(
        "--scenario",
        choices=[scenario.value for scenario in ScenarioName],
        default=None,
        help=(
            "Which scenario to train. ``read_aloud`` uses SpeechOcean's reference "
            "phones; ``free_speaking`` uses Whisper+g2p phones. When "
            "``free_speaking`` is selected, ``--use-curriculum`` and "
            "``--use-cono`` default to ON unless explicitly overridden."
        ),
    )
    parser.add_argument(
        "--use-curriculum",
        dest="use_curriculum",
        action="store_true",
        default=None,
        help="Enable paper eq.18 Bernoulli(p=τ/T) read-aloud↔free-speaking mixing.",
    )
    parser.add_argument(
        "--no-use-curriculum",
        dest="use_curriculum",
        action="store_false",
        default=None,
        help="Disable curriculum learning even under ``--scenario free_speaking``.",
    )
    parser.add_argument(
        "--curriculum-total-iters",
        type=int,
        default=None,
        help=(
            "Override ``T`` in ``P(τ) = τ/T``. Default: ``epochs * steps_per_epoch``."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")


def _settings_from_args(args: argparse.Namespace) -> TrainingRunSettings:
    overrides: dict[str, Any] = {}
    flag_to_field = {
        "output_dir": "output_dir",
        "cache_dir": "cache_dir",
        "run_name": "run_name",
        "device": "device",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "seeds": "seeds",
        "max_train_examples": "max_train_examples",
        "max_test_examples": "max_test_examples",
        "num_workers": "num_workers",
        "modernbert_model": "modernbert_model",
        "use_cono": "use_cono",
        "lambda_phone": "lambda_phone",
        "lambda_word": "lambda_word",
        "lambda_utterance": "lambda_utterance",
        "lambda_cono": "lambda_cono",
        "lambda_diversity": "lambda_diversity",
        "lambda_tightness": "lambda_tightness",
        "trackio_project": "trackio_project",
        "trackio_space": "trackio_space",
        "scenario": "scenario",
        "use_curriculum": "use_curriculum",
        "curriculum_total_iters": "curriculum_total_iters",
    }
    for arg_name, field_name in flag_to_field.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[field_name] = value

    # Paper eq.18 ties curriculum + CONO to the free-speaking scenario. If the
    # user explicitly flips either flag, we respect that; otherwise default ON.
    if overrides.get("scenario") == ScenarioName.FREE_SPEAKING.value:
        if getattr(args, "use_curriculum", None) is None:
            overrides["use_curriculum"] = True
        if getattr(args, "use_cono", None) is None:
            overrides["use_cono"] = True

    return TrainingRunSettings(**overrides)


def _resolve_device_arg(device_name: str | None) -> torch.device | None:
    if device_name is None:
        return None
    return torch.device(device_name)


def _add_gop_extract_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/p014"))
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--device")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")


def _add_gop_merge_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/p014"))
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")


def _add_ssl_extract_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/p014"))
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--device")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P014 HiPPO utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("show-config", help="Render a validated scenario config")
    config_parser.add_argument(
        "--scenario",
        choices=[scenario.value for scenario in ScenarioName],
        default=ScenarioName.FREE_SPEAKING.value,
        help="Scenario config to render",
    )
    config_parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")

    train_parser = subparsers.add_parser("train", help="Train the HiPPO read-aloud reproduction")
    _add_training_flags(train_parser)

    extract_gop_parser = subparsers.add_parser(
        "extract-gop", help="Extract CTC-GOP cache for a split or a single shard"
    )
    _add_gop_extract_flags(extract_gop_parser)

    merge_gop_parser = subparsers.add_parser(
        "merge-gop-shards", help="Merge shard-specific GOP caches into the standard split cache"
    )
    _add_gop_merge_flags(merge_gop_parser)

    extract_ssl_parser = subparsers.add_parser(
        "extract-ssl", help="Extract utterance-level SSL cache for a split"
    )
    _add_ssl_extract_flags(extract_ssl_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "show-config":
        config = load_named_config(args.scenario)
        # ``--json`` is accepted for scripting even though the default output
        # is already JSON; this keeps the flag's contract consistent.
        _ = bool(args.json)
        print(config.model_dump_json(indent=2))
        return 0

    if args.command == "train":
        settings = _settings_from_args(args)
        trial_summaries = train_hippo(settings)
        if args.json:
            print(
                json.dumps(
                    [asdict(trial) for trial in trial_summaries],
                    indent=2,
                    default=str,
                )
            )
        else:
            for trial in trial_summaries:
                print(trial)
        return 0

    if args.command == "extract-gop":
        cache_path = extract_gop_for_split(
            split=args.split,
            cache_dir=args.cache_dir,
            dataset_id=args.dataset_id or "mispeech/speechocean762",
            device=_resolve_device_arg(args.device),
            max_examples=args.max_examples,
            suffix=args.suffix,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "split": args.split,
                        "cache_path": str(cache_path),
                        "num_shards": args.num_shards,
                        "shard_index": args.shard_index,
                    },
                    indent=2,
                )
            )
        else:
            print(cache_path)
        return 0

    if args.command == "merge-gop-shards":
        cache_path = merge_gop_shards(
            split=args.split,
            cache_dir=args.cache_dir,
            num_shards=args.num_shards,
            dataset_id=args.dataset_id or "mispeech/speechocean762",
            max_examples=args.max_examples,
            suffix=args.suffix,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "split": args.split,
                        "cache_path": str(cache_path),
                        "num_shards": args.num_shards,
                    },
                    indent=2,
                )
            )
        else:
            print(cache_path)
        return 0

    if args.command == "extract-ssl":
        cache_path = extract_ssl_utterance_for_split(
            split=args.split,
            cache_dir=args.cache_dir,
            dataset_id=args.dataset_id or "mispeech/speechocean762",
            device=_resolve_device_arg(args.device),
            max_examples=args.max_examples,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "split": args.split,
                        "cache_path": str(cache_path),
                    },
                    indent=2,
                )
            )
        else:
            print(cache_path)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def show_config_entrypoint() -> int:
    return main(["show-config", *sys.argv[1:]])


# Re-export for callers that want programmatic typing.
__all__ = ["main", "show_config_entrypoint"]
