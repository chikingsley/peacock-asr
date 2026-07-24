# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "kaldialign @ git+https://github.com/pzelasko/kaldialign@06ac40f03c3d368932adf8536965a088d54189b1",
#   "regex>=2026.6.28",
# ]
# ///
"""Score saved ASR predictions with one pinned official Open ASR English scorer."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

KALDIALIGN_REVISION = "06ac40f03c3d368932adf8536965a088d54189b1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repository: Path) -> str:
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required to verify the pinned normalizer revision")
    result = subprocess.run(  # noqa: S603
        [git, "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_predictions(path: Path) -> tuple[list[str], list[str]]:
    references: list[str] = []
    hypotheses: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row: Any = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            reference = row.get("text")
            hypothesis = row.get("hypothesis")
            if not isinstance(reference, str) or not isinstance(hypothesis, str):
                raise TypeError(f"{path}:{line_number} lacks string text/hypothesis fields")
            references.append(reference)
            hypotheses.append(hypothesis)
    if not references:
        raise ValueError(f"no predictions in {path}")
    return references, hypotheses


def _score_pair(pair: tuple[tuple[str, ...], tuple[str, ...]]) -> dict[str, float | int]:
    from kaldialign import batch_error_rate  # noqa: PLC0415

    reference, hypothesis = pair
    return batch_error_rate([reference], [hypothesis], merge_compounds=True)


def exact_batch_error_rate(
    references: list[tuple[str, ...]],
    hypotheses: list[tuple[str, ...]],
    *,
    workers: int,
) -> dict[str, float | int]:
    """Run the pinned row-wise metric concurrently and sum its exact integer counts."""
    if workers < 1:
        raise ValueError("workers must be positive")
    pairs = list(zip(references, hypotheses, strict=True))
    if workers == 1 or len(pairs) == 1:
        from kaldialign import batch_error_rate  # noqa: PLC0415

        return batch_error_rate(references, hypotheses, merge_compounds=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(pairs))) as pool:
        partials = list(pool.map(_score_pair, pairs))
    result = {
        key: sum(int(partial[key]) for partial in partials)
        for key in ("ins", "del", "sub", "total", "ref_len")
    }
    result["err_rate"] = result["total"] / result["ref_len"]
    return result


def score(
    predictions: Path,
    normalizer_root: Path,
    *,
    normalizer_revision: str,
    workers: int,
) -> dict[str, Any]:
    actual_revision = git_revision(normalizer_root)
    if actual_revision != normalizer_revision:
        raise ValueError(
            f"normalizer revision mismatch: expected {normalizer_revision}, got {actual_revision}"
        )
    sys.path.insert(0, str(normalizer_root))
    from normalizer import EnglishTextNormalizer  # noqa: PLC0415

    references, hypotheses = read_predictions(predictions)
    normalizer = EnglishTextNormalizer()
    normalized_references = [normalizer(text) for text in references]
    normalized_hypotheses = [normalizer(text) for text in hypotheses]
    result = exact_batch_error_rate(
        [tuple(text.split()) for text in normalized_references],
        [tuple(text.split()) for text in normalized_hypotheses],
        workers=workers,
    )
    return {
        "schema_version": 1,
        "metric": "open-asr-english-kaldialign-merge-compounds",
        "rows": len(references),
        "wer_percent": float(result["err_rate"] * 100),
        "insertions": int(result["ins"]),
        "deletions": int(result["del"]),
        "substitutions": int(result["sub"]),
        "reference_words": int(result["ref_len"]),
        "normalized_empty_references": sum(not text for text in normalized_references),
        "normalized_empty_hypotheses": sum(not text for text in normalized_hypotheses),
        "predictions": str(predictions.resolve()),
        "predictions_sha256": sha256(predictions),
        "normalizer_root": str(normalizer_root.resolve()),
        "normalizer_revision": actual_revision,
        "kaldialign_revision": KALDIALIGN_REVISION,
        "scorer_workers": workers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalizer-root", type=Path, required=True)
    parser.add_argument("--normalizer-revision", required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable score output already exists: {args.output}")
    result = score(
        args.predictions.expanduser().resolve(),
        args.normalizer_root.expanduser().resolve(),
        normalizer_revision=args.normalizer_revision,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
