"""Shared fixtures: build real Common-Voice-shaped tar archives and manifest rows for tests."""

from __future__ import annotations

import io
import tarfile
from typing import TYPE_CHECKING

import pytest

from cv26.manifest import ManifestRow

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

ROOT = "common-voice/ab"


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def make_archive() -> Callable[..., None]:
    """Return a factory writing a CV-shaped tar; ``tsv_paths`` may reference missing clips."""

    def _make(
        path: Path,
        *,
        clips: Sequence[str] = ("one.mp3", "two.mp3"),
        tsv_paths: Sequence[str] | None = None,
    ) -> None:
        tsv_paths = list(tsv_paths if tsv_paths is not None else clips)
        body = "\n".join(
            f"c{i}\t{p}\ts{i}\tsentence {i}\t2\t0\tab" for i, p in enumerate(tsv_paths)
        )
        train = (
            "client_id\tpath\tsentence_id\tsentence\tup_votes\tdown_votes\tlocale\n" + body + "\n"
        )
        durations = "clip\tduration[ms]\n" + "\n".join(f"{c}\t1234" for c in clips) + "\n"
        with tarfile.open(path, "w:gz") as tar:
            _add(tar, f"{ROOT}/train.tsv", train.encode())
            _add(tar, f"{ROOT}/clip_durations.tsv", durations.encode())
            for clip in clips:
                _add(tar, f"{ROOT}/clips/{clip}", b"ID3" + clip.encode())

    return _make


@pytest.fixture
def cv_row() -> Callable[..., ManifestRow]:
    """Return a factory building a valid manifest row with a given archive filename."""

    def _row(filename: str = "cv.tar.gz") -> ManifestRow:
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

    return _row
