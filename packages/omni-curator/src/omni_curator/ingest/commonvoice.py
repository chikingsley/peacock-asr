"""Mozilla Common Voice -> ``Sample``. Pulled from the Mozilla Data Collective.

Common Voice now lives on the Mozilla Data Collective (mozilladatacollective.com): POST to a
dataset's download endpoint with a Bearer API key to get a short-lived presigned URL, then download
+ extract the tarball (:func:`download_commonvoice`). After extraction the language folder holds
split ``.tsv`` files (``train``/``dev``/``test``/``validated``; columns include ``client_id``,
``path``, ``sentence``) and a ``clips/`` dir of mp3s (typically 48 kHz — ``process`` resamples to
16 kHz). ``clip_durations.tsv`` gives per-clip duration. The API key goes in the env; dataset ids
are per language+corpus (recorded by the consuming project).
"""

from __future__ import annotations

import csv
import json
import shutil
import tarfile
import urllib.request
from typing import TYPE_CHECKING

from omni_curator.sample import Sample

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_MDC_BASE = "https://mozilladatacollective.com/api/datasets"


def download_commonvoice(
    dataset_id: str, *, dest: Path, api_key: str, archive_name: str | None = None
) -> Path:
    """Download + extract a Common Voice corpus from the Mozilla Data Collective; return its dir.

    POSTs ``{_MDC_BASE}/{dataset_id}/download`` with the Bearer key for a presigned URL, streams the
    tarball into ``dest``, extracts it, and returns the dir holding the split ``.tsv`` files.
    """
    dest.mkdir(parents=True, exist_ok=True)
    existing = _cv_data_dir(dest)
    if (existing / "validated.tsv").exists() or (existing / "train.tsv").exists():
        return existing  # already downloaded + extracted; resume from here
    request = urllib.request.Request(  # noqa: S310 — fixed https MDC endpoint
        f"{_MDC_BASE}/{dataset_id}/download",
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        download_url = str(json.load(response)["downloadUrl"])
    archive = dest / (archive_name or f"{dataset_id}.tar.gz")
    with (
        urllib.request.urlopen(download_url, timeout=1800) as response,  # noqa: S310
        archive.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    return _cv_data_dir(dest)


def _cv_data_dir(root: Path) -> Path:
    """Find the dir holding the audio (clips/ or audios/) + tsv (the tarball nests the corpus)."""
    if (root / "clips").is_dir() or (root / "audios").is_dir():
        return root
    for tsv in sorted(root.rglob("*.tsv")):
        if (tsv.parent / "clips").is_dir() or (tsv.parent / "audios").is_dir():
            return tsv.parent
    return root


# The scripted corpus ships disjoint train/dev/test split files.
_CV_SPLITS = ("train", "dev", "test")


def _scripted_durations(cv_dir: Path) -> dict[str, float]:
    """Map clip filename -> seconds from clip_durations.tsv (ms), for the scripted layout."""
    path = cv_dir / "clip_durations.tsv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        field = "duration[ms]" if "duration[ms]" in (reader.fieldnames or []) else "duration"
        return {row["clip"]: float(row[field]) / 1000.0 for row in reader if row.get(field)}


def _read_tsv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _row_sample(
    row: dict[str, str],
    *,
    split: str,
    audio_dir: Path,
    language: str,
    source: str,
    durations: dict[str, float],
) -> Sample | None:
    # scripted uses path/sentence; spontaneous uses audio_file/transcription + duration_ms.
    clip = row.get("path") or row.get("audio_file")
    text = str(row.get("sentence") or row.get("transcription") or "").strip()
    if not clip or not text:
        return None
    duration = durations.get(clip, 0.0)
    if not duration and row.get("duration_ms"):
        duration = float(row["duration_ms"]) / 1000.0
    return Sample(
        id=f"{source}_{language}_{split}_{clip}",
        source=source,
        language=language,
        text=text,
        audio_path=str(audio_dir / clip),
        duration=duration,
        sample_rate=0,  # set when process resamples the mp3
        split=split,
        speaker_id=row.get("client_id"),
        citation="commonvoice.mozilla.org",
        meta={"age": row.get("age"), "gender": row.get("gender")},
    )


def load_commonvoice(
    cv_dir: Path,
    *,
    language: str,
    source: str = "commonvoice",
    splits: tuple[str, ...] = _CV_SPLITS,
) -> Iterator[Sample]:
    """Read an extracted Common Voice dir -> ``Sample``s. Handles the scripted layout
    (``clips/`` + ``train``/``dev``/``test.tsv`` with path/sentence) and the spontaneous layout
    (``audios/`` + one tsv with audio_file/transcription and a ``split`` column). Clips stay mp3
    until process resamples them.
    """
    cv_dir = _cv_data_dir(cv_dir)
    audio_dir = cv_dir / ("clips" if (cv_dir / "clips").is_dir() else "audios")
    common = {"audio_dir": audio_dir, "language": language, "source": source}
    split_tsvs = [(s, cv_dir / f"{s}.tsv") for s in splits if (cv_dir / f"{s}.tsv").exists()]

    if split_tsvs:  # scripted: one tsv per split
        durations = _scripted_durations(cv_dir)
        for split, tsv in split_tsvs:
            for row in _read_tsv(tsv):
                sample = _row_sample(row, split=split, durations=durations, **common)
                if sample is not None:
                    yield sample
        return

    single = next(iter(sorted(cv_dir.glob("*.tsv"))), None)  # spontaneous: single tsv, split column
    if single is None:
        return
    for row in _read_tsv(single):
        sample = _row_sample(row, split=row.get("split") or "train", durations={}, **common)
        if sample is not None:
            yield sample
