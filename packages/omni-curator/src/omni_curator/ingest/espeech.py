"""ESpeech Russian ASR datasets -> ``Sample``.

Each ESpeech repo (webinars2 / podcasts / buldjat / upvote / tuchniyzhab) ships its audio as a
**split tar** (``<name>_archive.tar.aa`` .. ``.ag``) — concatenate the parts and untar to get::

    <root>/<recording-uuid>/<uuid>_<seg>.mp3        # one mp3 per speech segment
    <root>/<recording-uuid>/<uuid>.json             # per-recording metadata (list of segments)

The per-recording ``<uuid>.json`` is a JSON array, one object per segment with ``text`` (the
transcript) and ``index`` (zero-padded, e.g. ``"00035"``) — the clip for that segment is
``<uuid>_<int(index)>.mp3``. Some repos use ``file_name`` instead of ``index``. HF's generic loader
only exposes the ``mp3`` (the json isn't a webdataset sidecar), so we download + extract + parse the
metadata ourselves. All ESpeech sets are pseudo-labelled (WhisperX-style) — Scribe-verify before
trusting.
"""

from __future__ import annotations

import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from omni_curator.data.sample import Sample

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from omni_curator.project import CuratorProject, IngestFn


def download_espeech(repo: str, dest: Path) -> Path:
    """Download a repo's tar archive(s) and extract into ``dest``; return the root.

    Handles all ESpeech layouts: a single ``<name>.tar``, one split tar (``.tar.aa`` .. ``.ag``), or
    several split tars in one repo (podcasts ships ``podcasts_1_…``, ``podcasts_2_…``). Idempotent:
    skips the (slow) download+extract if already done.
    """
    dest.mkdir(parents=True, exist_ok=True)
    sentinel = dest / ".extracted.done"
    if sentinel.exists():
        return dest
    parts_dir = dest / "_parts"
    parts_dir.mkdir(exist_ok=True)
    subprocess.run(  # noqa: S603  # resumable; --include keeps READMEs/images out
        [  # noqa: S607
            "hf", "download", repo, "--repo-type", "dataset",
            "--include", "*.tar*", "--local-dir", str(parts_dir),
        ],
        check=True,
    )
    parts = sorted(  # exclude hf's .cache/*.metadata bookkeeping files (they also match *.tar*)
        p
        for p in parts_dir.rglob("*.tar*")
        if ".cache" not in p.parts and not p.name.endswith(".metadata")
    )
    if not parts:
        msg = f"no tar archive (*.tar*) found for {repo} under {parts_dir}"
        raise FileNotFoundError(msg)
    # group by archive base: a split part ``<base>.tar.aa`` joins ``<base>.tar``; a plain ``.tar``
    # is its own group. Each group is one tar — concat its parts in order and stream into tar.
    groups: dict[str, list[Path]] = {}
    for p in parts:
        base = p.name[:-3] if re.fullmatch(r".*\.tar\.[a-z]{2}", p.name) else p.name
        groups.setdefault(base, []).append(p)
    for _, gparts in sorted(groups.items()):
        with subprocess.Popen(  # noqa: S603
            ["cat", *map(str, sorted(gparts))],  # noqa: S607
            stdout=subprocess.PIPE,
        ) as cat, tarfile.open(fileobj=cat.stdout, mode="r|") as tar:
            tar.extractall(dest, filter="data")
            cat.wait()
    sentinel.touch()
    return dest


def _segment_clip(meta: Path, uuid: str, position: int, seg: dict[str, Any]) -> Path:
    """Resolve a segment's mp3: ``file_name`` if given, else ``<uuid>_<position>.mp3``.

    Clips are numbered by the segment's POSITION in the json array (0..N-1), NOT its ``index`` field
    — ``index`` is an original-recording marker that can repeat within a recording.
    """
    file_name = seg.get("file_name")
    if file_name:
        return meta.parent / Path(str(file_name)).name
    return meta.parent / f"{uuid}_{position}.mp3"


def load_espeech(root: Path, *, language: str, source: str) -> Iterator[Sample]:
    """Read an extracted ESpeech tree -> ``Sample``s by pairing each json segment with its mp3.

    Walks every per-recording ``<uuid>/<uuid>.json`` (lazily — these sets run to millions of clips),
    yielding one sample per segment that has both a transcript and an on-disk clip.
    """
    for meta in root.rglob("*.json"):
        uuid = meta.parent.name
        if meta.stem != uuid:  # only the per-recording metadata, not stray json
            continue
        try:
            segments = json.loads(meta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(segments, list):
            continue
        for position, item in enumerate(segments):
            if not isinstance(item, dict):
                continue
            seg = cast("dict[str, Any]", item)  # json values are dynamic
            text = str(seg.get("text") or "").strip()
            clip = _segment_clip(meta, uuid, position, seg)
            if not text or not clip.exists():
                continue
            yield Sample(
                id=f"{source}_{uuid}_{position}",  # position is unique per recording
                source=source,
                language=language,
                text=text,
                audio_path=str(clip),
                duration=float(seg.get("end") or 0.0) - float(seg.get("start") or 0.0),
                sample_rate=0,  # set when process resamples
                split="train",
                speaker_id=str(seg["speaker"]) if seg.get("speaker") else None,
                citation="huggingface.co/ESpeech",
                meta={"recording": uuid},
            )


def espeech_source(repo: str, *, source: str, force_split: str = "train") -> IngestFn:
    """Ready-made ingest for an ESpeech split-tar dataset (download+extract+parse+resample)."""

    def load(project: CuratorProject) -> Iterable[Sample]:
        import dataclasses

        from omni_curator.process import resample_samples

        root = download_espeech(repo, project.raw_dir / "espeech" / source)
        samples = load_espeech(root, language=project.language, source=source)
        if force_split is not None:
            samples = (dataclasses.replace(s, split=force_split) for s in samples)
        return resample_samples(samples, project.canonical_dir / source)

    return load
