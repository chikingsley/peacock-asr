#!/usr/bin/env python3
"""Build a manifest-backed OmniASR phoneme dataset from the P004 phone cuts."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "projects" / "P003-compact-backbones"
P004_ROOT = REPO_ROOT / "projects" / "P004-training-from-scratch"
P004_MANIFEST_ROOT = P004_ROOT / "experiments" / "data" / "manifests_phone_raw"
P004_AUDIO_ROOT = P004_MANIFEST_ROOT / "audio"
DEFAULT_MANIFEST_DIR = (
    PROJECT_ROOT / "code" / "omni" / "datasets" / "librispeech_phone_manifest_v1"
)
DEFAULT_CARD_PATH = (
    PROJECT_ROOT
    / "code"
    / "omni"
    / "cards"
    / "dataset_librispeech_phone_manifest_v1.yaml"
)
DATASET_ASSET_NAME = "peacock_librispeech_phone_manifest_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Output directory for train/dev manifest files.",
    )
    parser.add_argument(
        "--dataset-card",
        type=Path,
        default=DEFAULT_CARD_PATH,
        help="Output dataset asset YAML path.",
    )
    return parser.parse_args()


def _iter_cuts(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_split(
    *,
    split_name: str,
    records: list[dict[str, Any]],
    manifest_dir: Path,
) -> int:
    tsv_path = manifest_dir / f"{split_name}.tsv"
    wrd_path = manifest_dir / f"{split_name}.wrd"
    count = 0
    with tsv_path.open("w", encoding="utf-8") as tsv, wrd_path.open(
        "w", encoding="utf-8"
    ) as wrd:
        tsv.write(f"{P004_AUDIO_ROOT.resolve()}\n")
        for record in records:
            supervision = record["supervisions"][0]
            source = Path(record["recording"]["sources"][0]["source"]).resolve()
            rel_source = source.relative_to(P004_AUDIO_ROOT.resolve())
            num_samples = int(record["recording"]["num_samples"])
            text = str(supervision["text"]).strip()
            if not text:
                continue
            tsv.write(f"{rel_source.as_posix()}\t{num_samples}\n")
            wrd.write(f"{text}\n")
            count += 1
    return count


def _write_dataset_card(dataset_card: Path, manifest_dir: Path) -> None:
    dataset_card.parent.mkdir(parents=True, exist_ok=True)
    dataset_card.write_text(
        (
            f"name: {DATASET_ASSET_NAME}\n"
            "dataset_family: manifest_asr_dataset\n"
            "dataset_config:\n"
            f"  data: {manifest_dir.resolve()}\n"
            "tokenizer_ref: omniASR_tokenizer_arpabet_41_v1\n"
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    manifest_dir = args.manifest_dir.resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_records = []
    for name in ("train-clean-100", "train-clean-360"):
        train_records.extend(
            _iter_cuts(P004_MANIFEST_ROOT / f"librispeech_cuts_{name}.jsonl.gz")
        )
    dev_records = _iter_cuts(P004_MANIFEST_ROOT / "librispeech_cuts_dev-clean.jsonl.gz")

    train_count = _write_split(
        split_name="train",
        records=train_records,
        manifest_dir=manifest_dir,
    )
    dev_count = _write_split(
        split_name="dev",
        records=dev_records,
        manifest_dir=manifest_dir,
    )
    _write_dataset_card(args.dataset_card.resolve(), manifest_dir)

    summary = {
        "dataset_asset_name": DATASET_ASSET_NAME,
        "manifest_dir": str(manifest_dir),
        "dataset_card": str(args.dataset_card.resolve()),
        "audio_root": str(P004_AUDIO_ROOT.resolve()),
        "train_count": train_count,
        "dev_count": dev_count,
    }
    sys.stdout.write(f"{json.dumps(summary, indent=2)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
