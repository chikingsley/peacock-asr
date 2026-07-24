"""Build immutable, style-harmonized English training mixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_APOSTROPHES = frozenset({"'", "\u2018", "\u2019", "\u02bc", "`"})
_SOURCE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_WHITESPACE_RE = re.compile(r"\s+")
LABEL_PROFILE = "lexical-lower-v1"
_MINIMUM_SOURCE_COUNT = 2


@dataclass(frozen=True)
class Source:
    """One source dataset with separate train and development manifests."""

    name: str
    directory: Path

    @property
    def train_manifest(self) -> Path:
        return self.directory / "train.jsonl"

    @property
    def dev_manifest(self) -> Path:
        return self.directory / "dev.jsonl"


def normalize_training_text(text: str) -> str:
    """Normalize case and punctuation while preserving lexical apostrophes."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    characters: list[str] = []
    for character in normalized:
        if character in _APOSTROPHES:
            characters.append("'")
        elif unicodedata.category(character).startswith("P"):
            characters.append(" ")
        else:
            characters.append(character)
    return _WHITESPACE_RE.sub(" ", "".join(characters)).strip()


def parse_source(value: str) -> Source:
    """Parse a ``NAME=DIRECTORY`` source argument."""
    name, separator, directory = value.partition("=")
    if not separator or not name or not directory:
        raise ValueError(f"invalid --source {value!r}; expected NAME=DIRECTORY")
    if not _SOURCE_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid source name {name!r}; use lowercase letters, digits, and dashes")
    return Source(name=name, directory=Path(directory).resolve())


def parse_source_weights(values: list[str]) -> dict[str, float] | None:
    """Parse optional ``NAME=WEIGHT`` arguments and reject ambiguous mixtures."""
    if not values:
        return None
    weights: dict[str, float] = {}
    for value in values:
        name, separator, raw_weight = value.partition("=")
        if not separator or not name or not raw_weight:
            raise ValueError(f"invalid --source-weight {value!r}; expected NAME=WEIGHT")
        if not _SOURCE_NAME_RE.fullmatch(name):
            raise ValueError(
                f"invalid source name {name!r}; use lowercase letters, digits, and dashes"
            )
        if name in weights:
            raise ValueError(f"duplicate source weight for {name!r}")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"source weight for {name!r} must be finite and positive")
        weights[name] = weight
    return weights


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


def _transform_manifest(source: Path, destination: Path) -> tuple[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    count = 0
    with (
        source.open(encoding="utf-8") as input_handle,
        destination.open("x", encoding="utf-8") as output_handle,
    ):
        for line_number, line in enumerate(input_handle, start=1):
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str):
                raise TypeError(f"{source}:{line_number} has no string text field")
            normalized = normalize_training_text(text)
            if not normalized:
                raise ValueError(f"{source}:{line_number} normalized to empty text")
            row["text"] = normalized
            output_handle.write(_json_line(row))
            rows.append(row)
            count += 1
    return count, rows


def _selection_key(row: dict[str, Any], seed: int) -> str:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("balanced validation requires a nonempty sample_id on every row")
    return hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest()


def _validate_sources(sources: list[Source], output_dir: Path) -> Path:
    if len(sources) < _MINIMUM_SOURCE_COUNT:
        raise ValueError("a balanced mixture requires at least two sources")
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("source names must be unique")
    if output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    for source in sources:
        for manifest in (source.train_manifest, source.dev_manifest):
            if not manifest.is_file():
                raise FileNotFoundError(manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp"
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    return temporary


def _write_source_views(
    sources: list[Source], temporary: Path, sampling_weights: dict[str, float]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    summaries: list[dict[str, Any]] = []
    dev_rows: dict[str, list[dict[str, Any]]] = {}
    for source in sorted(sources, key=lambda item: item.name):
        source_output = temporary / source.name
        source_output.mkdir()
        train_output = source_output / "train.jsonl"
        dev_output = source_output / "dev.jsonl"
        train_count, _ = _transform_manifest(source.train_manifest, train_output)
        dev_count, transformed_dev = _transform_manifest(source.dev_manifest, dev_output)
        dev_rows[source.name] = transformed_dev
        summaries.append(
            {
                "name": source.name,
                "sampling_weight": sampling_weights[source.name],
                "input": {
                    "directory": str(source.directory),
                    "train_sha256": _sha256(source.train_manifest),
                    "dev_sha256": _sha256(source.dev_manifest),
                },
                "output": {
                    "train": f"{source.name}/train.jsonl",
                    "train_rows": train_count,
                    "train_sha256": _sha256(train_output),
                    "dev": f"{source.name}/dev.jsonl",
                    "dev_rows": dev_count,
                    "dev_sha256": _sha256(dev_output),
                },
            }
        )
    return summaries, dev_rows


def _sampling_weights(
    sources: list[Source], requested: dict[str, float] | None
) -> dict[str, float]:
    names = {source.name for source in sources}
    if requested is None:
        return {name: 1.0 / len(sources) for name in names}
    if set(requested) != names:
        missing = sorted(names - set(requested))
        extra = sorted(set(requested) - names)
        raise ValueError(
            f"source weights must match sources exactly; missing={missing}, extra={extra}"
        )
    total = sum(requested.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"source weights must sum to 1.0, got {total:.12g}")
    return requested


def _validation_count(dev_rows: dict[str, list[dict[str, Any]]], requested: int | None) -> int:
    available = min(len(rows) for rows in dev_rows.values())
    selected = requested or available
    if selected <= 0:
        raise ValueError("validation_per_source must be positive")
    if selected > available:
        raise ValueError(
            f"validation_per_source={selected} exceeds the smallest source dev={available}"
        )
    return selected


def build_mixture(
    sources: list[Source],
    output_dir: Path,
    *,
    validation_per_source: int | None = None,
    seed: int = 0,
    sampling_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build per-source lexical manifests and a balanced validation manifest."""
    output_dir = output_dir.resolve()
    temporary = _validate_sources(sources, output_dir)
    resolved_weights = _sampling_weights(sources, sampling_weights)
    temporary.mkdir()

    try:
        summaries, dev_rows = _write_source_views(sources, temporary, resolved_weights)
        selected_count = _validation_count(dev_rows, validation_per_source)
        selected = {
            name: sorted(rows, key=lambda row: _selection_key(row, seed))[:selected_count]
            for name, rows in dev_rows.items()
        }
        balanced_dev = temporary / "balanced-dev.jsonl"
        with balanced_dev.open("x", encoding="utf-8") as handle:
            for index in range(selected_count):
                for name in sorted(selected):
                    handle.write(_json_line(selected[name][index]))

        summary = {
            "schema_version": 1,
            "label_profile": LABEL_PROFILE,
            "seed": seed,
            "sources": summaries,
            "balanced_validation": {
                "path": "balanced-dev.jsonl",
                "rows_per_source": selected_count,
                "rows": selected_count * len(sources),
                "sha256": _sha256(balanced_dev),
            },
        }
        summary_path = temporary / "mixture_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    temporary.replace(output_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=DIRECTORY")
    parser.add_argument("--source-weight", action="append", default=[], metavar="NAME=WEIGHT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-per-source", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_mixture(
        [parse_source(value) for value in args.source],
        args.output_dir,
        validation_per_source=args.validation_per_source,
        seed=args.seed,
        sampling_weights=parse_source_weights(args.source_weight),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
