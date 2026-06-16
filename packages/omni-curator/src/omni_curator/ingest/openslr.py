"""OpenSLR Russian corpora -> ``Sample``, pulled from the canonical OpenSLR releases.

Two layouts, selected by SLR id:

**SLR114 — Golos** (https://openslr.org/114/, https://github.com/salute-developers/golos).
Manually-annotated crowd + farfield Russian speech, ~1240 h, shipped as WAV in a dozen tarballs
(``train_farfield.tar``, ``train_crowd0..9.tar``, ``test.tar``). The training transcript manifests
live *only* in ``train_crowd9.tar``. Extracting the set yields::

    train/crowd/{0..9}/...wav      test/crowd/...wav
    train/farfield/...wav          test/farfield/...wav
    train/crowd/manifest.jsonl     test/crowd/manifest.jsonl
    train/farfield/manifest.jsonl  test/farfield/manifest.jsonl
    train/farfield/{10min,1hour,10hours,100hours}.jsonl   (subset manifests; ignored)

Each ``manifest.jsonl`` is NeMo-style: one JSON object per line with ``audio_filepath`` (relative to
the manifest's own directory) and ``text``. We read every ``manifest.jsonl`` (skipping the size-cut
subset manifests) and resolve audio relative to each manifest.

**SLR96 — RuLS (Russian LibriSpeech)** (https://openslr.org/96/). ~98 h of LibriVox audiobook read
speech. Extracts to ``{train,dev,test}/`` each holding a NeMo ``manifest.json`` (jsonl:
``audio_filepath`` relative to the split dir, ``duration``, ``text`` + ``text_no_preprocessing``)
beside an ``audio/<reader>/<book>/*.wav`` tree (16 kHz wav). We read each split's ``manifest.json``
and take the split from the dir name.

Both paths download to ``project.raw_dir`` and resample to 16 kHz mono FLAC under
``project.canonical_dir`` via :func:`omni_curator.process.resample_sample`.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from omni_curator.sample import Sample

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from omni_curator.project import CuratorProject, IngestFn

# Canonical CDN the OpenSLR page + Golos GitHub README point at (the SberDevices mirror).
_GOLOS_CDN = "https://cdn.chatwm.opensmodel.sberdevices.ru/golos"
# train_crowd9.tar carries the train manifests, so it must come last (its extract completes train/).
_GOLOS_TARS = (
    "test.tar",
    "train_farfield.tar",
    *(f"train_crowd{i}.tar" for i in range(10)),
)
_RULS_URL = "https://www.openslr.org/resources/96/ruls_data.tar.gz"

# Golos farfield ships size-cut subsets next to the full manifest; only the full one is the corpus.
_GOLOS_MANIFEST = "manifest.jsonl"


def _download(url: str, dest: Path, *, timeout: int = 3600) -> Path:
    """Stream ``url`` into ``dest`` (skips if present); return the local path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with (
        urllib.request.urlopen(url, timeout=timeout) as response,  # noqa: S310 — fixed https mirrors
        tmp.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    tmp.rename(dest)
    return dest


def _extract(archive: Path, dest: Path, *, sentinel: Path) -> None:
    """Extract ``archive`` under ``dest`` once (``sentinel`` marks a completed prior extract)."""
    if sentinel.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")


def download_golos(dest: Path, *, tars: tuple[str, ...] = _GOLOS_TARS) -> Path:
    """Download + extract the Golos WAV tarballs into ``dest/golos``; return the extracted root."""
    root = dest / "golos"
    for name in tars:
        archive = _download(f"{_GOLOS_CDN}/{name}", dest / name)
        # Each tar unpacks into the shared train/ + test/ tree; key extraction off its own marker.
        _extract(archive, root, sentinel=dest / f".{name}.done")
        (dest / f".{name}.done").touch()
    return root


def download_ruls(dest: Path) -> Path:
    """Download + extract ruls_data.tar.gz into ``dest``; return the dir holding the split tree."""
    archive = _download(_RULS_URL, dest / "ruls_data.tar.gz")
    _extract(archive, dest, sentinel=dest / ".ruls_data.done")
    (dest / ".ruls_data.done").touch()
    return _ruls_root(dest)


def _ruls_root(dest: Path) -> Path:
    """Find the dir whose children are the split folders (the tarball nests under ruls_data/)."""
    if (dest / "ruls_data").is_dir():
        return dest / "ruls_data"
    return dest


def _golos_split(manifest: Path) -> str:
    """train/ vs test/ from the manifest path (crowd/farfield kept separate via ``source``)."""
    parts = {p.lower() for p in manifest.parts}
    return "test" if "test" in parts else "train"


def load_golos(
    root: Path, *, language: str, source: str = "golos"
) -> Iterator[Sample]:
    """Read an extracted Golos tree -> ``Sample``s (audio stays WAV; process resamples it).

    Reads every full ``manifest.jsonl`` (NeMo jsonl: ``audio_filepath`` relative to the manifest
    dir, ``text``) and skips the farfield size-cut subset manifests.
    """
    for manifest in sorted(root.rglob(_GOLOS_MANIFEST)):
        split = _golos_split(manifest)
        with manifest.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                rel = row.get("audio_filepath")
                text = str(row.get("text") or "").strip()
                if not rel or not text:
                    continue
                clip = (manifest.parent / rel).resolve()
                # uid keeps the domain (crowd/farfield) + relative path unique across manifests.
                uid = f"{source}_{split}_{manifest.parent.name}_{Path(rel).stem}"
                yield Sample(
                    id=uid,
                    source=source,
                    language=language,
                    text=text,
                    audio_path=str(clip),
                    duration=float(row.get("duration") or 0.0),
                    sample_rate=0,  # set when process resamples the wav
                    split=split,
                    citation="openslr.org/114",
                    meta={"domain": manifest.parent.name},
                )


def load_ruls(
    root: Path, *, language: str, source: str = "ruls"
) -> Iterator[Sample]:
    """Read an extracted RuLS (Russian LibriSpeech) tree -> ``Sample``s (wav; process resamples).

    RuLS ships ``{train,dev,test}/manifest.json`` (NeMo jsonl: ``audio_filepath`` relative to the
    split dir, ``duration``, normalized ``text`` + original ``text_no_preprocessing``). We prefer
    the un-preprocessed text (the curator normalizes uniformly); split comes from the dir name.
    """
    for manifest in sorted(root.glob("*/manifest.json")):
        split = manifest.parent.name  # train | dev | test
        with manifest.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                rel = row.get("audio_filepath")
                text = str(row.get("text_no_preprocessing") or row.get("text") or "").strip()
                if not rel or not text:
                    continue
                clip = (manifest.parent / rel).resolve()
                if not clip.exists():
                    continue
                yield Sample(
                    id=f"{source}_{split}_{Path(rel).stem}",
                    source=source,
                    language=language,
                    text=text,
                    audio_path=str(clip),
                    duration=float(row.get("duration") or 0.0),
                    sample_rate=0,
                    split=split,
                    citation="openslr.org/96",
                )


# slr id -> (downloader, loader, default source name).
_LOADERS = {
    "114": (download_golos, load_golos, "golos"),
    "96": (download_ruls, load_ruls, "ruls"),
}


def load_openslr(
    slr_id: str, *, dest: Path, language: str, source: str | None = None
) -> Iterator[Sample]:
    """Download + parse an OpenSLR Russian corpus (``114``=Golos, ``96``=RuLS) -> ``Sample``s.

    Audio is left in its source format (Golos wav, RuLS flac); the caller resamples to 16 kHz mono
    FLAC. ``dest`` is the per-corpus download cache (under ``project.raw_dir``).
    """
    key = slr_id.removeprefix("SLR").removeprefix("slr")
    if key not in _LOADERS:
        raise ValueError(f"unsupported OpenSLR id {slr_id!r}; supported: {sorted(_LOADERS)}")
    download, load, default_source = _LOADERS[key]
    root = download(dest)
    yield from load(root, language=language, source=source or default_source)


def openslr_source(
    slr_id: str, *, source: str | None = None, force_split: str = "train"
) -> IngestFn:
    """Ready-made ingest: an OpenSLR Russian corpus (``"114"``=Golos, ``"96"``=RuLS).

    Downloads the canonical OpenSLR tarballs to ``project.raw_dir`` and emits 16 kHz mono FLAC clips
    under ``project.canonical_dir``. ``source`` defaults to the corpus name (``golos``/``ruls``).

    ``force_split`` overrides every row's split — defaults to ``"train"`` so these third-party
    corpora's own ``test`` partitions don't leak into the project's benchmark split (benchmark
    splits export ungated, so this is a trust decision; pass ``force_split=None`` to keep them).
    """

    def load(project: CuratorProject) -> Iterable[Sample]:
        import dataclasses

        from omni_curator.process import resample_samples

        key = slr_id.removeprefix("SLR").removeprefix("slr")
        name = source or _LOADERS.get(key, (None, None, f"slr{key}"))[2]
        samples = load_openslr(
            slr_id,
            dest=project.raw_dir / "openslr" / name,
            language=project.language,
            source=name,
        )
        if force_split is not None:
            samples = (dataclasses.replace(s, split=force_split) for s in samples)
        yield from resample_samples(samples, project.canonical_dir / name)

    return load
