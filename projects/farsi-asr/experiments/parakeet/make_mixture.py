"""Compose an hour-budgeted training mixture from per-corpus NeMo manifests.

Each --take is `<manifest-stem>:<hours>`; rows are sampled without replacement in a
seeded shuffle until the hour budget fills, so mixtures are reproducible and the
realized sampled hours per source are printed for the experiment record.

Usage:
    uv run --no-sync python experiments/parakeet/make_mixture.py \
        --out data/parakeet/manifests/read20h_train.jsonl \
        --take fleurs_train:7 --take neyshekar_train:9 --take worldspeech_train:4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from farsi_asr import ROOT

MANIFEST_ROOT = ROOT / "data/parakeet/manifests"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--take",
        action="append",
        required=True,
        metavar="STEM:HOURS",
        help="Manifest stem under data/parakeet/manifests plus an hour budget.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)  # noqa: S311 - deterministic sampling, not crypto
    combined: list[dict] = []
    for spec in args.take:
        stem, _, hours_text = spec.partition(":")
        budget_seconds = float(hours_text) * 3600.0
        manifest = MANIFEST_ROOT / f"{stem}.jsonl"
        rows = [json.loads(line) for line in manifest.open(encoding="utf-8")]
        rng.shuffle(rows)
        taken: list[dict] = []
        total = 0.0
        for row in rows:
            if total >= budget_seconds:
                break
            taken.append(row)
            total += row["duration"]
        combined.extend(taken)
        print(
            f"{stem}: requested {float(hours_text):.2f} h -> realized {total / 3600:.2f} h "
            f"({len(taken)}/{len(rows)} rows)",
            flush=True,
        )

    rng.shuffle(combined)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    total_hours = sum(row["duration"] for row in combined) / 3600
    print(f"{args.out}: {len(combined)} rows / {total_hours:.2f} h total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
