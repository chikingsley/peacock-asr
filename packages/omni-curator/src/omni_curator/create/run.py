"""End-to-end create: source -> label -> store. Lands generated labels in the curator store."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omni_curator.create.pipeline import chunks_path, vad_path
from omni_curator.create.transcribe import DEFAULT_LANGS

if TYPE_CHECKING:
    from pathlib import Path

    from omni_curator.store import CuratorStore

_PATHS = {"chunks": chunks_path, "vad": vad_path}


def label_to_store(
    audio: Path,
    *,
    store: CuratorStore,
    source: str,
    language: str,
    script: str,
    id_prefix: str,
    out_dir: Path,
    path: str = "chunks",
    citation: str | None = None,
    langs: tuple[str, ...] = DEFAULT_LANGS,
    instruction: str | None = None,
    **path_kwargs: Any,
) -> int:
    """Label one audio file with the create pipeline; upsert its clips into the store."""
    transcript = _PATHS[path](
        audio,
        out_dir=out_dir,
        language=language,
        script=script,
        langs=langs,
        instruction=instruction,
        **path_kwargs,
    )
    samples = transcript.to_samples(
        source=source, language=language, id_prefix=id_prefix, citation=citation
    )
    return store.upsert(samples)


def label_youtube(
    url: str,
    *,
    store: CuratorStore,
    language: str,
    script: str,
    work_dir: Path,
    path: str = "chunks",
    **kwargs: Any,
) -> int:
    """Download a YouTube video's audio, label it, and land the clips in the store."""
    from omni_curator.create.sources.youtube import download_audio

    audio = download_audio(url, out_dir=work_dir / "raw")
    return label_to_store(
        audio.audio_path,
        store=store,
        source="youtube",
        language=language,
        script=script,
        id_prefix=audio.video_id,
        out_dir=work_dir / "labeled" / audio.video_id,
        path=path,
        citation=audio.url,
        **kwargs,
    )
