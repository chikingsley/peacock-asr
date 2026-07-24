"""Prepare a leakage-safe Common Voice 26 English training selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

_HASH = re.compile(r"^[0-9a-f]{64}$")
_BATCH_ROWS = 512


@dataclass(frozen=True, slots=True)
class IdentityIndex:
    """Stable clip/audio identities loaded from one frozen JSONL ledger."""

    clips: frozenset[str]
    audio_sha256: frozenset[str]

    def matches(self, clip_id: str, audio_sha256: str) -> bool:
        return clip_id in self.clips or audio_sha256 in self.audio_sha256


@dataclass(slots=True)
class PrepStats:
    """Counts accumulated while classifying CV26 upstream-train rows."""

    rows: int = 0
    hours: float = 0.0
    train_candidate: int = 0
    post_cv7_candidate: int = 0
    cv7_replay: int = 0
    base_replay_unknown: int = 0
    excluded_benchmark: int = 0


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """One source row plus its stable identities."""

    source: Path
    row_index: int
    row: Mapping[str, object]
    clip_id: str
    audio_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip_id(row: Mapping[str, object]) -> str:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        audio = row.get("audio")
        value = audio.get("path") if isinstance(audio, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("CV26 row has no stable clip path")
    return Path(value).name


def _audio_bytes(row: Mapping[str, object]) -> bytes:
    audio = row.get("audio")
    raw = audio.get("bytes") if isinstance(audio, Mapping) else None
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("CV26 row has no encoded audio bytes")
    return raw


def _ledger_record(line: str, *, path: Path, line_number: int) -> tuple[str | None, str | None]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
    if not isinstance(row, dict):
        raise TypeError(f"identity ledger row must be an object: {path}:{line_number}")
    clip = row.get("clip_id") or row.get("path")
    clip_id = Path(clip).name if isinstance(clip, str) and clip.strip() else None
    audio_hash = row.get("audio_sha256") or row.get("encoded_audio_sha256")
    if audio_hash is not None:
        audio_hash = str(audio_hash).lower()
        if not _HASH.fullmatch(audio_hash):
            raise ValueError(f"invalid audio SHA-256 in {path}:{line_number}")
    if clip_id is None and audio_hash is None:
        raise ValueError(f"identity ledger row has no clip or audio hash: {path}:{line_number}")
    return clip_id, audio_hash


def load_identity_ledger(path: Path) -> IdentityIndex:
    """Load a non-empty frozen identity ledger, failing closed on malformed rows."""
    if not path.is_file():
        raise FileNotFoundError(f"identity ledger not found: {path}")
    clips: set[str] = set()
    hashes: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            clip, audio_hash = _ledger_record(raw, path=path, line_number=line_number)
            if clip is not None:
                clips.add(clip)
            if audio_hash is not None:
                hashes.add(audio_hash)
    if not clips and not hashes:
        raise ValueError(f"identity ledger is empty: {path}")
    return IdentityIndex(frozenset(clips), frozenset(hashes))


def _named_ledgers(values: Sequence[str]) -> dict[str, Path]:
    ledgers: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"--benchmark-ledger expects NAME=PATH, got {value!r}")
        if name in ledgers:
            raise ValueError(f"duplicate benchmark ledger name: {name}")
        ledgers[name] = Path(raw_path)
    if not ledgers:
        raise ValueError("at least one --benchmark-ledger is required")
    return ledgers


def _source_rows(paths: Iterable[Path]) -> Iterator[tuple[Path, int, dict[str, object]]]:
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"source Parquet not found: {path}")
        if not path.name.endswith(".parquet"):
            raise ValueError(f"CV26 prep refuses partial/non-Parquet source: {path}")
        row_index = 0
        parquet = pq.ParquetFile(path)
        required = {"audio", "path", "upstream_split"}
        missing = required.difference(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"CV26 source {path} is missing columns: {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=_BATCH_ROWS):
            for row in batch.to_pylist():
                if row.get("upstream_split") != "train":
                    raise ValueError(
                        f"CV26 prep accepts upstream train only: {path} row {row_index}"
                    )
                yield path, row_index, row
                row_index += 1


def _identity_rows(paths: Iterable[Path]) -> Iterator[tuple[Path, int, dict[str, object]]]:
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"source Parquet not found: {path}")
        if not path.name.endswith(".parquet"):
            raise ValueError(f"identity build refuses partial/non-Parquet source: {path}")
        row_index = 0
        parquet = pq.ParquetFile(path)
        if "audio" not in parquet.schema_arrow.names:
            raise ValueError(f"identity source {path} has no audio column")
        for batch in parquet.iter_batches(batch_size=_BATCH_ROWS):
            for row in batch.to_pylist():
                yield path, row_index, row
                row_index += 1


def _selection_row(
    identity: SourceIdentity,
    data_class: str,
    benchmark_matches: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "audio_sha256": identity.audio_sha256,
        "benchmark_matches": list(benchmark_matches) or None,
        "client_id": identity.row.get("client_id"),
        "clip_id": identity.clip_id,
        "data_class": data_class,
        "duration_ms": identity.row.get("duration_ms"),
        "row_index": identity.row_index,
        "sentence_id": identity.row.get("sentence_id"),
        "source_dataset_id": identity.row.get("source_dataset_id"),
        "source_parquet": identity.source.name,
    }


def _write_jsonl(handle: object, row: Mapping[str, object]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _data_class(
    identity: SourceIdentity,
    cv7: IdentityIndex | None,
    benchmarks: Mapping[str, IdentityIndex],
) -> tuple[str, list[str]]:
    matches = sorted(
        name
        for name, index in benchmarks.items()
        if index.matches(identity.clip_id, identity.audio_sha256)
    )
    if matches:
        return "excluded_benchmark", matches
    if cv7 is None:
        return "base_replay_unknown", []
    if cv7.matches(identity.clip_id, identity.audio_sha256):
        return "cv7_replay", []
    return "post_cv7_candidate", []


def _validate_inputs(
    source_parquets: Sequence[Path],
    source_revision: str,
    benchmark_ledgers: Mapping[str, Path],
    output_dir: Path,
) -> None:
    if not source_revision.strip():
        raise ValueError("a pinned --source-revision is required")
    if not source_parquets:
        raise ValueError("at least one --source-parquet is required")
    source_names = [path.name for path in source_parquets]
    if len(set(source_names)) != len(source_names):
        raise ValueError("source Parquet basenames must be unique")
    if not benchmark_ledgers:
        raise ValueError("at least one benchmark ledger is required")
    if output_dir.exists():
        raise FileExistsError(f"immutable CV26 prep output already exists: {output_dir}")


def prepare(
    *,
    source_parquets: Sequence[Path],
    source_revision: str,
    cv7_ledger: Path | None,
    benchmark_ledgers: Mapping[str, Path],
    output_dir: Path,
) -> PrepStats:
    """Build a benchmark-clean CV26 train selection and optional base-replay classes."""
    _validate_inputs(source_parquets, source_revision, benchmark_ledgers, output_dir)
    cv7 = load_identity_ledger(cv7_ledger) if cv7_ledger is not None else None
    benchmark_paths = dict(sorted(benchmark_ledgers.items()))
    benchmarks = {name: load_identity_ledger(path) for name, path in benchmark_paths.items()}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    stats = PrepStats()
    try:
        with (
            (temp / "train_candidate.jsonl").open("w", encoding="utf-8") as candidate,
            (temp / "post_cv7_candidate.jsonl").open("w", encoding="utf-8") as novel,
            (temp / "cv7_replay.jsonl").open("w", encoding="utf-8") as replay,
            (temp / "base_replay_unknown.jsonl").open("w", encoding="utf-8") as unknown,
            (temp / "excluded_benchmark.jsonl").open("w", encoding="utf-8") as excluded,
        ):
            for source, row_index, row in _source_rows(source_parquets):
                clip_id = _clip_id(row)
                audio_sha256 = hashlib.sha256(_audio_bytes(row)).hexdigest()
                identity = SourceIdentity(source, row_index, row, clip_id, audio_sha256)
                duration_ms = row.get("duration_ms")
                if isinstance(duration_ms, int) and duration_ms > 0:
                    stats.hours += duration_ms / 3_600_000
                data_class, matches = _data_class(identity, cv7, benchmarks)
                stats.rows += 1
                if data_class == "excluded_benchmark":
                    stats.excluded_benchmark += 1
                    _write_jsonl(
                        excluded,
                        _selection_row(
                            identity,
                            data_class="excluded_benchmark",
                            benchmark_matches=matches,
                        ),
                    )
                else:
                    stats.train_candidate += 1
                    _write_jsonl(
                        candidate,
                        _selection_row(
                            identity,
                            data_class=data_class,
                        ),
                    )
                    if data_class == "cv7_replay":
                        stats.cv7_replay += 1
                        _write_jsonl(
                            replay,
                            _selection_row(identity, data_class="cv7_replay"),
                        )
                    elif data_class == "post_cv7_candidate":
                        stats.post_cv7_candidate += 1
                        _write_jsonl(
                            novel,
                            _selection_row(identity, data_class="post_cv7_candidate"),
                        )
                    else:
                        stats.base_replay_unknown += 1
                        _write_jsonl(
                            unknown,
                            _selection_row(identity, data_class="base_replay_unknown"),
                        )
        summary = {
            **asdict(stats),
            "benchmark_ledgers": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in benchmark_paths.items()
            },
            "cv7_ledger": (
                {"path": str(cv7_ledger), "sha256": _sha256(cv7_ledger)}
                if cv7_ledger is not None
                else None
            ),
            "base_replay_classification": "exact-cv7" if cv7 is not None else "unknown",
            "source_parquets": [
                {"path": str(path), "sha256": _sha256(path)} for path in sorted(source_parquets)
            ],
            "source_revision": source_revision,
        }
        (temp / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.replace(output_dir)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return stats


def build_identity_ledger(
    *,
    source_parquets: Sequence[Path],
    source_revision: str,
    source_name: str,
    output_ledger: Path,
) -> int:
    """Build a pinned clip/audio identity ledger from benchmark or base-training Parquets."""
    if not source_revision.strip():
        raise ValueError("a pinned --source-revision is required")
    if not source_name.strip():
        raise ValueError("--source-name is required")
    if not source_parquets:
        raise ValueError("at least one --source-parquet is required")
    summary_path = output_ledger.with_suffix(output_ledger.suffix + ".summary.json")
    if output_ledger.exists() or summary_path.exists():
        raise FileExistsError(f"immutable identity ledger already exists: {output_ledger}")
    output_ledger.parent.mkdir(parents=True, exist_ok=True)
    temp = output_ledger.with_name(f".{output_ledger.name}.tmp")
    rows = 0
    try:
        with temp.open("w", encoding="utf-8") as handle:
            for source, row_index, row in _identity_rows(source_parquets):
                _write_jsonl(
                    handle,
                    {
                        "audio_sha256": hashlib.sha256(_audio_bytes(row)).hexdigest(),
                        "clip_id": _clip_id(row),
                        "row_index": row_index,
                        "source_parquet": source.name,
                    },
                )
                rows += 1
        _require_identity_rows(rows)
        temp.replace(output_ledger)
        summary = {
            "ledger": str(output_ledger),
            "ledger_sha256": _sha256(output_ledger),
            "rows": rows,
            "source_name": source_name,
            "source_parquets": [
                {"path": str(path), "sha256": _sha256(path)} for path in sorted(source_parquets)
            ],
            "source_revision": source_revision,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        temp.unlink(missing_ok=True)
        output_ledger.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        raise
    return rows


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-parquet", action="append", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)


def _require_identity_rows(rows: int) -> None:
    if rows == 0:
        raise ValueError("identity source produced no rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe Common Voice 26 English training data."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ledger = commands.add_parser("build-ledger", help="build a frozen identity ledger")
    _add_source_args(ledger)
    ledger.add_argument("--source-name", required=True)
    ledger.add_argument("--output-ledger", type=Path, required=True)

    prep = commands.add_parser("prepare", help="classify CV26 upstream-train identities")
    _add_source_args(prep)
    prep.add_argument(
        "--cv7-ledger",
        type=Path,
        help="optional exact CV7 identity ledger; omission records base replay as unknown",
    )
    prep.add_argument(
        "--benchmark-ledger",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    prep.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-ledger":
            rows = build_identity_ledger(
                source_parquets=args.source_parquet,
                source_revision=args.source_revision,
                source_name=args.source_name,
                output_ledger=args.output_ledger,
            )
            print(f"built identity ledger rows={rows} -> {args.output_ledger}", flush=True)
            return 0
        benchmark_ledgers = _named_ledgers(args.benchmark_ledger)
        stats = prepare(
            source_parquets=args.source_parquet,
            source_revision=args.source_revision,
            cv7_ledger=args.cv7_ledger,
            benchmark_ledgers=benchmark_ledgers,
            output_dir=args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"prepared rows={stats.rows} hours={stats.hours:.2f} "
        f"train_candidate={stats.train_candidate} "
        f"post_cv7_candidate={stats.post_cv7_candidate} cv7_replay={stats.cv7_replay} "
        f"base_replay_unknown={stats.base_replay_unknown} "
        f"excluded_benchmark={stats.excluded_benchmark} -> {args.output_dir}",
        flush=True,
    )
    return 0
