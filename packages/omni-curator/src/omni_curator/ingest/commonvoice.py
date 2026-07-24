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
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
from typing import TYPE_CHECKING

from omni_curator.data.sample import Sample
from omni_curator.ingest._util import slug as _slug
from omni_curator.ingest.huggingface import _derived_split

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_MDC_BASE = "https://mozilladatacollective.com/api/datasets"
_SAMPLE_RATE = 16_000


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
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        field = "duration[ms]" if "duration[ms]" in (reader.fieldnames or []) else "duration"
        return {row["clip"]: float(row[field]) / 1000.0 for row in reader if row.get(field)}


def _read_tsv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


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


def _read_tar_tsv(tar: tarfile.TarFile, member: tarfile.TarInfo) -> list[dict[str, str]]:
    handle = tar.extractfile(member)
    if handle is None:
        return []
    # Common Voice TSV values are not CSV-quoted. Prompt text can contain unmatched literal
    # double quotes; treating those as quote delimiters merges many physical records into one
    # enormous sentence and can make the RNNT joint tensor exhaust GPU memory.
    text = io.StringIO(handle.read().decode("utf-8"), newline="")
    return list(csv.DictReader(text, delimiter="\t", quoting=csv.QUOTE_NONE))


def _valid_audio_info(path: Path, soundfile: object) -> object | None:
    if not path.exists():
        return None
    try:
        return soundfile.info(str(path))
    except soundfile.LibsndfileError:
        path.unlink(missing_ok=True)
        return None


def _write_archive_clip(
    encoded: bytes,
    destination: Path,
    *,
    soundfile: object,
    soxr: object,
    numpy: object,
) -> object:
    """Decode one archive member and atomically write canonical 16 kHz mono FLAC."""
    info = _valid_audio_info(destination, soundfile)
    if info is not None:
        return info
    samples, sample_rate = soundfile.read(io.BytesIO(encoded), dtype="float32")
    samples = numpy.asarray(samples)
    if samples.ndim > 1:
        samples = numpy.mean(samples, axis=1)
    if sample_rate != _SAMPLE_RATE:
        samples = soxr.resample(samples, sample_rate, _SAMPLE_RATE)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        soundfile.write(str(temporary), samples, _SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return soundfile.info(str(destination))


def load_commonvoice_archive(  # noqa: C901, PLR0912, PLR0915
    archive: Path,
    *,
    language: str,
    source: str,
    audio_dir: Path,
    upstream_split: str = "train",
    validation_fraction: float = 0.05,
    split_seed: int = 17,
    excluded_clip_ids: frozenset[str] = frozenset(),
    excluded_audio_sha256: frozenset[str] = frozenset(),
    max_hours: float | None = None,
) -> Iterator[Sample]:
    """Stream one Common Voice archive directly into canonical audio samples.

    The gzip tar is consumed once in forward-only mode. Only the requested upstream split is
    decoded, so callers do not need to extract the archive or create embedded-audio Parquet
    intermediates. Existing valid FLACs are reused on rerun. Exact benchmark clip IDs and encoded
    audio hashes can be excluded before samples enter the curator store.
    """
    import numpy as np
    import soundfile as sf
    import soxr

    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    target_tsv = f"{upstream_split}.tsv"
    source_slug = _slug(source)
    rows: dict[str, dict[str, str]] | None = None
    layout: str | None = None
    yielded_seconds = 0.0

    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.rsplit("/", 1)[-1]
            if name == target_tsv:
                rows = {
                    str(row.get("path") or ""): row
                    for row in _read_tar_tsv(tar, member)
                    if row.get("path") and row.get("sentence")
                }
                layout = "scripted"
                continue
            if name.endswith(".tsv") and rows is None:
                candidate_rows = _read_tar_tsv(tar, member)
                if candidate_rows and {"audio_file", "transcription", "split"}.issubset(
                    candidate_rows[0]
                ):
                    rows = {
                        str(row.get("audio_file") or ""): row
                        for row in candidate_rows
                        if row.get("audio_file")
                        and row.get("transcription")
                        and row.get("split") == upstream_split
                    }
                    layout = "spontaneous"
                continue
            audio_marker = "/audios/" if layout == "spontaneous" else "/clips/"
            if audio_marker not in member.name or not name.lower().endswith(".mp3"):
                continue
            if rows is None:
                raise ValueError(
                    f"archive stores audio before {target_tsv} or spontaneous metadata; "
                    "forward-only ingest cannot classify them"
                )
            row = rows.pop(name, None)
            if row is None or name in excluded_clip_ids:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            encoded = handle.read()
            encoded_sha256 = hashlib.sha256(encoded).hexdigest()
            if encoded_sha256 in excluded_audio_sha256:
                continue
            identity = f"{source}\0{upstream_split}\0{name}"
            identity_hash = hashlib.sha256(identity.encode()).hexdigest()
            uid = f"{source_slug}_{identity_hash[:24]}"
            destination = audio_dir / identity_hash[:2] / f"{uid}.flac"
            info = _write_archive_clip(
                encoded,
                destination,
                soundfile=sf,
                soxr=soxr,
                numpy=np,
            )
            duration = float(info.frames) / float(info.samplerate)
            group = str(row.get("client_id") or name)
            output_split = _derived_split(
                upstream_split,
                group=group,
                validation_fraction=validation_fraction,
                split_seed=split_seed,
            )
            metadata = {
                "upstream_split": upstream_split,
                "clip_id": name,
                "encoded_audio_sha256": encoded_sha256,
                "sentence_id": row.get("sentence_id"),
                "up_votes": row.get("up_votes"),
                "down_votes": row.get("down_votes"),
                "age": row.get("age"),
                "gender": row.get("gender"),
                "accents": row.get("accents"),
                "variant": row.get("variant"),
                "split_group": group,
            }
            if layout == "spontaneous":
                metadata.update(
                    {
                        "audio_id": row.get("audio_id"),
                        "duration_ms": row.get("duration_ms"),
                        "prompt_id": row.get("prompt_id"),
                        "prompt": row.get("prompt"),
                        "votes": row.get("votes"),
                        "is_edited": row.get("is_edited"),
                        "quality_tags": row.get("quality_tags"),
                    }
                )
            yield Sample(
                id=uid,
                source=source,
                language=language,
                text=str(row.get("sentence") or row.get("transcription") or "").strip(),
                audio_path=str(destination),
                duration=duration,
                sample_rate=_SAMPLE_RATE,
                split=output_split,
                speaker_id=row.get("client_id") or None,
                citation="Mozilla Common Voice",
                meta=metadata,
            )
            yielded_seconds += duration
            if max_hours is not None and yielded_seconds >= max_hours * 3600:
                return

    if rows is None:
        raise ValueError(f"archive has no {target_tsv} or spontaneous metadata TSV: {archive}")
