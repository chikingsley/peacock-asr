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
    """Find the dir holding the split .tsv files (the tarball may nest the language folder)."""
    if (root / "validated.tsv").exists() or (root / "train.tsv").exists():
        return root
    for marker in ("validated.tsv", "train.tsv"):
        found = next(iter(sorted(root.rglob(marker))), None)
        if found is not None:
            return found.parent
    return root

# Common Voice ships these; we keep train/dev/test (the disjoint official splits).
_CV_SPLITS = ("train", "dev", "test")


def _durations(cv_dir: Path) -> dict[str, float]:
    """Map clip filename -> seconds from clip_durations.tsv (duration is in ms), if present."""
    path = cv_dir / "clip_durations.tsv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        field = "duration[ms]" if "duration[ms]" in (reader.fieldnames or []) else "duration"
        return {row["clip"]: float(row[field]) / 1000.0 for row in reader if row.get(field)}


def load_commonvoice(
    cv_dir: Path,
    *,
    language: str,
    source: str = "commonvoice",
    splits: tuple[str, ...] = _CV_SPLITS,
) -> Iterator[Sample]:
    """Read an extracted Common Voice language dir -> ``Sample``s (clips stay mp3 until process)."""
    cv_dir = _cv_data_dir(cv_dir)  # robust if handed the extraction root
    clips = cv_dir / "clips"
    durations = _durations(cv_dir)
    for split in splits:
        tsv = cv_dir / f"{split}.tsv"
        if not tsv.exists():
            continue
        with tsv.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                clip = row.get("path")
                text = str(row.get("sentence") or "").strip()
                if not clip or not text:
                    continue
                yield Sample(
                    id=f"{source}_{language}_{split}_{clip}",
                    source=source,
                    language=language,
                    text=text,
                    audio_path=str(clips / clip),
                    duration=durations.get(clip, 0.0),
                    sample_rate=0,  # unknown until process probes/resamples the mp3
                    split=split,
                    speaker_id=row.get("client_id"),
                    citation="commonvoice.mozilla.org",
                    meta={"age": row.get("age"), "gender": row.get("gender")},
                )
