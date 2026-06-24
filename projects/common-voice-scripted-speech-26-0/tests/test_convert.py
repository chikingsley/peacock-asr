"""Real end-to-end convert test: build a Common-Voice-shaped tar, convert, read the parquet."""

from __future__ import annotations

import io
import tarfile
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from cv26.convert import convert_archive, shard_path
from cv26.manifest import ManifestRow

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

ROOT = "common-voice/ab"
MP3_BYTES = b"ID3\x04fake-audio-bytes"
TRAIN_TSV = (
    "client_id\tpath\tsentence_id\tsentence\tup_votes\tdown_votes\tlocale\n"
    "c1\tone.mp3\ts1\thello world\t2\t0\tab\n"
    "c2\ttwo.mp3\ts2\tgoodbye\t1\t1\tab\n"
)
DURATIONS_TSV = "clip\tduration[ms]\none.mp3\t1234\ntwo.mp3\t5678\n"


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _make_archive(path: Path) -> None:
    with tarfile.open(path, "w:gz") as tar:
        _add(tar, f"{ROOT}/train.tsv", TRAIN_TSV.encode())
        _add(tar, f"{ROOT}/clip_durations.tsv", DURATIONS_TSV.encode())
        _add(tar, f"{ROOT}/clips/one.mp3", MP3_BYTES)
        _add(tar, f"{ROOT}/clips/two.mp3", MP3_BYTES)


def _row(filename: str) -> ManifestRow:
    return ManifestRow(
        dataset_id="ds123",
        name="Common Voice Scripted Speech 26.0 - Abkhaz",
        collection="Common Voice Scripted Speech 26.0",
        locale="ab",
        language="Abkhaz",
        filename=filename,
        license="CC0-1.0",
        license_url="https://spdx.org/licenses/CC0-1.0.html",
        source="Mozilla Data Collective",
        download_api="https://mozilladatacollective.com/api/datasets/ds123/download",
    )


def test_convert_archive_writes_expected_parquet(tmp_path: Path) -> None:
    """A built archive converts to one train shard with embedded audio, durations, and metadata."""
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    row = _row("cv.tar.gz")
    _make_archive(archive_dir / row.filename)
    out_dir = tmp_path / "out"

    shards = convert_archive(row, archive_dir, out_dir)

    assert shards == [shard_path(out_dir, row, "train")]
    table = pq.read_table(shards[0])
    assert table.num_rows == 2
    rows = table.to_pylist()
    by_sentence = {r["sentence"]: r for r in rows}
    assert set(by_sentence) == {"hello world", "goodbye"}
    hello = by_sentence["hello world"]
    assert hello["audio"]["bytes"] == MP3_BYTES
    assert hello["audio"]["path"] == "one.mp3"
    assert hello["duration_ms"] == 1234
    assert hello["up_votes"] == 2
    assert hello["language"] == "Abkhaz"
    assert hello["locale"] == "ab"
    assert hello["upstream_split"] == "train"


def test_convert_archive_skips_existing_shard(tmp_path: Path) -> None:
    """A second run returns the existing shard path without rewriting it."""
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    row = _row("cv.tar.gz")
    _make_archive(archive_dir / row.filename)
    out_dir = tmp_path / "out"

    first = convert_archive(row, archive_dir, out_dir)
    mtime = first[0].stat().st_mtime_ns
    second = convert_archive(row, archive_dir, out_dir)

    assert second == first
    assert second[0].stat().st_mtime_ns == mtime  # not rewritten


def test_convert_archive_missing_returns_empty(tmp_path: Path) -> None:
    """A row whose archive is absent yields no shards."""
    assert convert_archive(_row("absent.tar.gz"), tmp_path, tmp_path / "out") == []


def test_convert_archive_rewrites_corrupt_shard(tmp_path: Path) -> None:
    """A partial/corrupt existing shard is detected and rewritten instead of trusted."""
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    row = _row("cv.tar.gz")
    _make_archive(archive_dir / row.filename)
    out_dir = tmp_path / "out"

    shards = convert_archive(row, archive_dir, out_dir)
    shards[0].write_bytes(b"not a parquet file")  # simulate a crash mid-write

    again = convert_archive(row, archive_dir, out_dir)

    assert again == shards
    assert pq.read_table(again[0]).num_rows == 2


def _evil_collection_row() -> ManifestRow:
    # `collection` is not allowlisted (it is human text); slug() must neutralize separators in it.
    return ManifestRow(
        dataset_id="ds123",
        name="evil",
        collection="/tmp/cv26_escape",  # noqa: S108 — adversarial input, asserting it cannot escape
        locale="ab",
        language="Y",
        filename="cv.tar.gz",
        license="CC0-1.0",
        license_url="https://spdx.org/licenses/CC0-1.0.html",
        source="s",
        download_api="https://example/api",
    )


def test_malicious_collection_cannot_escape_out_dir(
    tmp_path: Path,
    make_archive: Callable[..., None],
) -> None:
    """A `collection` with path separators is slugged and stays under out_dir, never at /tmp."""
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    row = _evil_collection_row()
    make_archive(archive_dir / row.filename)
    out_dir = tmp_path / "out"

    shards = convert_archive(row, archive_dir, out_dir)

    assert shards
    assert all(s.resolve().is_relative_to(out_dir.resolve()) for s in shards)


def test_corrupt_shard_rewriting_to_zero_rows_is_removed(
    tmp_path: Path,
    make_archive: Callable[..., None],
    cv_row: Callable[..., ManifestRow],
) -> None:
    """A corrupt shard whose split now yields zero rows is deleted, not left lingering."""
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    row = cv_row()
    make_archive(archive_dir / row.filename, clips=["one.mp3"], tsv_paths=["ghost.mp3"])
    out_dir = tmp_path / "out"
    dest = shard_path(out_dir, row, "train")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"corrupt not parquet")

    shards = convert_archive(row, archive_dir, out_dir)

    assert shards == []
    assert not dest.exists()
