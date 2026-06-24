"""Convert one Common Voice MDC archive into appendable Hugging Face parquet shards.

:func:`convert_archive` is the importable unit the pipeline calls directly and gets the written
shard paths back — no subprocess, no diffing the output dir to discover what was produced. One
shard is written per upstream split (``data/<split>/<collection>__<locale>__<dataset_id>.parquet``)
so adding a language adds files instead of rewriting a monolithic dataset.

Conversion is a SINGLE streaming pass over the archive (``r|*``): members are read once in order,
the small split/duration TSVs are buffered, and each clip's bytes are written straight to its
split shard(s) as it is reached. This avoids ``getmembers()`` and per-clip random ``extractfile``
seeks, which on a multi-GB gzip tar re-decompress from the start every time (O(n^2), effectively
hours per large archive). Audio is held one clip at a time, so memory stays bounded.
"""

from __future__ import annotations

import csv
import io
import logging
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import IO

    from cv26.manifest import ManifestRow

LOGGER = logging.getLogger("cv26.convert")

# Some Common Voice TSV fields exceed Python's 128 KB default csv field limit; raise it so the
# DictReader doesn't crash mid-archive on a long field.
csv.field_size_limit(64 * 1024 * 1024)

ASR_SPLITS = ("train", "dev", "test", "validated", "invalidated", "other")
_SPLIT_TSV_NAMES: dict[str, str] = {f"{split}.tsv": split for split in ASR_SPLITS}
_ROOT_DEPTH = 2  # archive members live under "<collection>/<locale>/..."
_WRITE_BATCH = 256  # rows (with embedded audio) held in memory per split before a parquet flush
COMMON_VOICE_COLUMNS = (
    "client_id",
    "path",
    "sentence_id",
    "sentence",
    "sentence_domain",
    "up_votes",
    "down_votes",
    "age",
    "gender",
    "accents",
    "variant",
    "locale",
    "segment",
)


def slug(value: str) -> str:
    """Convert a human name into a single safe path component (no separators can survive)."""
    # "/" and "\\" are mapped too so an unvalidated component (e.g. collection) can never inject a
    # path separator and escape the output root.
    subs = ((" ", "_"), (".", "_"), ("-", "_"), ("/", "_"), ("\\", "_"), ("'", ""), ("’", ""))  # noqa: RUF001
    for old, new in subs:
        value = value.replace(old, new)
    return value.lower()


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field(
                "audio",
                pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
            ),
            pa.field("source_audio_path", pa.string()),
            pa.field("upstream_split", pa.string()),
            pa.field("source_dataset_id", pa.string()),
            pa.field("source_archive", pa.string()),
            pa.field("collection", pa.string()),
            pa.field("locale", pa.string()),
            pa.field("language", pa.string()),
            pa.field("license", pa.string()),
            pa.field("license_url", pa.string()),
            pa.field("duration_ms", pa.int64()),
            pa.field("client_id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("sentence_id", pa.string()),
            pa.field("sentence", pa.string()),
            pa.field("sentence_domain", pa.string()),
            pa.field("up_votes", pa.int64()),
            pa.field("down_votes", pa.int64()),
            pa.field("age", pa.string()),
            pa.field("gender", pa.string()),
            pa.field("accents", pa.string()),
            pa.field("variant", pa.string()),
            pa.field("segment", pa.string()),
        ],
    )


def _int_or_none(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def shard_path(out_dir: Path, row: ManifestRow, upstream_split: str) -> Path:
    """Return the parquet shard path for one archive split."""
    return (
        out_dir
        / "data"
        / upstream_split
        / f"{slug(row.collection)}__{row.locale.replace('/', '_')}__{row.dataset_id}.parquet"
    )


def _is_readable_parquet(path: Path) -> bool:
    """Return whether the file is a complete, readable parquet shard (not a partial write)."""
    try:
        pq.read_metadata(path)
    except (OSError, pa.ArrowException):
        LOGGER.warning("corrupt shard, rewriting: %s", path.name)
        return False
    return True


def _read_tsv(member_file: IO[bytes]) -> list[dict[str, str]]:
    # Read bytes and decode — a streaming-mode (`r|*`) member is not seekable, so TextIOWrapper
    # (which probes .seekable()) can't wrap it. TSVs are small, so reading fully is fine.
    text = member_file.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def _record(
    audio_bytes: bytes,
    clip_name: str,
    root: str,
    duration_ms: int | None,
    row: ManifestRow,
    upstream_split: str,
    tsv_row: Mapping[str, str],
) -> dict[str, object]:
    record: dict[str, object] = {
        "audio": {"bytes": audio_bytes, "path": clip_name},
        "source_audio_path": f"{root}/clips/{clip_name}",
        "upstream_split": upstream_split,
        "source_dataset_id": row.dataset_id,
        "source_archive": row.filename,
        "collection": row.collection,
        "locale": row.locale,
        "language": row.language,
        "license": row.license,
        "license_url": row.license_url,
        "duration_ms": duration_ms,
    }
    for column in COMMON_VOICE_COLUMNS:
        record[column] = tsv_row.get(column) or None
    record["up_votes"] = _int_or_none(tsv_row.get("up_votes"))
    record["down_votes"] = _int_or_none(tsv_row.get("down_votes"))
    return record


class _ShardWriters:
    """Lazily-opened, bounded-memory parquet writers — one per split, written atomically."""

    def __init__(self, out_dir: Path, row: ManifestRow) -> None:
        self.out_dir = out_dir
        self.row = row
        self.schema = _schema()
        self.batches: dict[str, list[dict[str, object]]] = {}
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.counts: dict[str, int] = {}

    def _dest(self, split: str) -> Path:
        dest = shard_path(self.out_dir, self.row, split)
        if not dest.resolve().is_relative_to(self.out_dir.resolve()):
            msg = f"shard path escapes out_dir {self.out_dir}: {dest}"
            raise ValueError(msg)
        return dest

    def _tmp(self, split: str) -> Path:
        dest = self._dest(split)
        return dest.with_name(dest.name + ".tmp")

    def add(self, split: str, record: dict[str, object]) -> None:
        batch = self.batches.setdefault(split, [])
        batch.append(record)
        if len(batch) >= _WRITE_BATCH:
            self.flush(split)

    def flush(self, split: str) -> None:
        batch = self.batches.get(split)
        if not batch:
            return
        writer = self.writers.get(split)
        if writer is None:
            tmp = self._tmp(split)
            tmp.parent.mkdir(parents=True, exist_ok=True)
            writer = self.writers[split] = pq.ParquetWriter(tmp, self.schema, compression="zstd")
        writer.write_table(pa.Table.from_pylist(batch, schema=self.schema))
        self.counts[split] = self.counts.get(split, 0) + len(batch)
        batch.clear()

    def finalize(self, present_splits: set[str]) -> list[Path]:
        """Flush/close, atomically rename non-empty shards, drop empty/stale ones; return shards."""
        for split in list(self.batches):
            self.flush(split)
        for writer in self.writers.values():
            writer.close()
        shards: list[Path] = []
        for split in present_splits:
            dest = self._dest(split)
            if self.counts.get(split, 0) > 0:
                self._tmp(split).replace(dest)
                LOGGER.info("wrote %s rows=%s", dest.relative_to(self.out_dir), self.counts[split])
                shards.append(dest)
            else:  # split had no usable rows: clear any partial temp and any stale prior shard
                self._tmp(split).unlink(missing_ok=True)
                dest.unlink(missing_ok=True)
        return shards


class _Converter:
    """Single-pass archive reader: buffers split/duration TSVs, writes clip rows as encountered."""

    def __init__(self, row: ManifestRow, out_dir: Path) -> None:
        self.row = row
        self.writers = _ShardWriters(out_dir, row)
        self.split_rows: dict[str, dict[str, dict[str, str]]] = {}  # split -> {clip_name: tsv_row}
        self.durations: dict[str, int] = {}
        self.root = ""

    def ingest(self, tar: tarfile.TarFile, member: tarfile.TarInfo) -> None:
        if not member.name:
            return
        if not self.root:
            parts = Path(member.name).parts
            if len(parts) >= _ROOT_DEPTH:
                self.root = "/".join(parts[:_ROOT_DEPTH])
        if not member.isfile():
            return
        name = Path(member.name).name
        if name in _SPLIT_TSV_NAMES:
            handle = tar.extractfile(member)
            if handle is not None:
                self.split_rows[_SPLIT_TSV_NAMES[name]] = {
                    r.get("path", ""): r for r in _read_tsv(handle)
                }
        elif name == "clip_durations.tsv":
            self._read_durations(tar, member)
        elif "/clips/" in member.name and name.endswith(".mp3"):
            self._write_clip(tar, member, name)

    def _read_durations(self, tar: tarfile.TarFile, member: tarfile.TarInfo) -> None:
        handle = tar.extractfile(member)
        if handle is None:
            return
        for r in _read_tsv(handle):
            clip, ms = r.get("clip", ""), r.get("duration[ms]", "")
            if clip and ms.isdigit():
                self.durations[clip] = int(ms)

    def _write_clip(self, tar: tarfile.TarFile, member: tarfile.TarInfo, name: str) -> None:
        referencing = [split for split, rows in self.split_rows.items() if name in rows]
        if not referencing:
            return
        handle = tar.extractfile(member)
        if handle is None:
            LOGGER.warning("unreadable clip %s in %s", name, self.row.filename)
            return
        audio = handle.read()
        for split in referencing:
            self.writers.add(
                split,
                _record(
                    audio,
                    name,
                    self.root,
                    self.durations.get(name),
                    self.row,
                    split,
                    self.split_rows[split][name],
                ),
            )

    def finalize(self) -> list[Path]:
        return self.writers.finalize(set(self.split_rows))


def _stream_convert(row: ManifestRow, archive_path: Path, out_dir: Path) -> list[Path]:
    converter = _Converter(row, out_dir)
    with tarfile.open(archive_path, "r|*") as tar:  # streaming mode: one forward pass, no seeks
        for member in tar:
            converter.ingest(tar, member)
    shards = converter.finalize()
    if not shards:
        LOGGER.info("converted no ASR rows: %s", row.dataset_id)
    return shards


def convert_archive(
    row: ManifestRow,
    archive_dir: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Convert one downloaded archive into parquet shards; return the shard paths for it.

    A re-run reuses existing readable shards without re-reading the archive (unless ``overwrite``);
    a corrupt existing shard triggers a full reconvert. Splits with no ASR rows produce no shard.
    """
    archive_path = archive_dir / row.filename
    if not archive_path.exists():
        LOGGER.info("not_downloaded: %s", archive_path)
        return []

    if not overwrite:
        existing = [p for split in ASR_SPLITS if (p := shard_path(out_dir, row, split)).exists()]
        if existing and all(_is_readable_parquet(p) for p in existing):
            for path in existing:
                LOGGER.info("exists: %s", path.relative_to(out_dir))
            return existing

    return _stream_convert(row, archive_path, out_dir)
