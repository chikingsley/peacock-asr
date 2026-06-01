"""The two finalized curation paths, from one audio file to one transcript.

Both share the per-clip core — cut -> Scribe ensemble -> compile-down — run over a thread pool
(the Scribe and SuperWhisper calls are I/O-bound). They differ only in how the audio is cut and
how the per-clip labels are assembled:

- ``vad_path``    : segment_vad (no overlap)        -> labels joined in order        -> polish
- ``chunks_path`` : segment_chunks (overlap)        -> stitch (reconcile the seams)  -> polish

Use ``vad_path`` for dense continuous speech; use ``chunks_path`` for sparse / drill-style audio
where VAD would drop short utterances. ``polish`` is a cheap final glitch-fix on both and can be
turned off. The output is a transcript, not yet a dataset (chunking -> omni-parquet is a later
stage).
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from omni_curator.create.fuse import compile_down, polish, stitch
from omni_curator.create.segmenters import segment_chunks, segment_vad
from omni_curator.create.transcribe import (
    DEFAULT_LANGS,
    default_key,
    make_scribe_fns,
    transcribe_clip,
)
from omni_curator.process.language import keep_for_language
from omni_curator.sample import Sample

if TYPE_CHECKING:
    from pathlib import Path

    from superwhisper_api.text.client import SuperwhisperClient


# --------------------------------------------------------------------------------------------
# Create-stage audio primitives — cut a span out of a source recording, measure a file's length.
# Used only by the create paths below (and create/align), so they live here rather than as a
# package-level module.
# --------------------------------------------------------------------------------------------


def cut_audio(source: Path, output: Path, start: float, end: float) -> None:
    """Cut ``[start, end)`` from ``source`` to a 16 kHz mono FLAC at ``output`` (via ffmpeg)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(source),
            "-ar", "16000", "-ac", "1", "-c:a", "flac", str(output),
        ],
        check=True,
    )


def audio_duration(audio: Path) -> float:
    """Duration of ``audio`` in seconds."""
    import soundfile as sf

    info = sf.info(str(audio))
    return float(info.frames) / float(info.samplerate)


@dataclass
class Clip:
    index: int
    start: float
    end: float
    audio_path: str
    variants: list[str]
    label: str


@dataclass
class Transcript:
    text: str
    clips: list[Clip]

    def write_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"text": self.text, "clips": [asdict(c) for c in self.clips]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def to_samples(
        self,
        *,
        source: str,
        language: str,
        id_prefix: str,
        citation: str | None = None,
        split: str = "train",
        sample_rate: int = 16_000,
    ) -> list[Sample]:
        """Each labelled clip becomes a store Sample (the create -> store bridge).

        ``id_prefix`` makes ids unique across recordings (e.g. the video id); the clips are the
        16 kHz FLAC cuts produced by the pipeline. Clips with an empty label, or one the language
        gate rejects (e.g. a Russian segment in a Tajik source — see
        :func:`omni_curator.process.language.keep_for_language`), are skipped.
        """
        return [
            Sample(
                id=f"{id_prefix}_{clip.index:04d}",
                source=source,
                language=language,
                text=clip.label,
                audio_path=clip.audio_path,
                duration=round(clip.end - clip.start, 3),
                sample_rate=sample_rate,
                split=split,
                citation=citation,
            )
            for clip in self.clips
            if clip.label.strip() and keep_for_language(clip.label, language)
        ]


def _label_clip(
    item: tuple[int, tuple[float, float]],
    *,
    source: Path,
    out_dir: Path,
    scribe_fns: dict[str, Any],
    runs: int,
    client: SuperwhisperClient,
    language: str,
    script: str,
    instruction: str | None,
) -> Clip:
    index, (start, end) = item
    clip = out_dir / "cuts" / f"seg_{index:04d}.flac"
    cut_audio(source, clip, start, end)
    variants = transcribe_clip(clip, scribe_fns, runs=runs)
    label = (
        compile_down(
            variants, language=language, script=script, instruction=instruction, client=client
        )
        if variants
        else ""
    )
    return Clip(index, round(start, 2), round(end, 2), str(clip), variants, label)


def _label_spans(
    audio: Path,
    spans: list[tuple[float, float]],
    *,
    out_dir: Path,
    langs: tuple[str, ...],
    runs: int,
    workers: int,
    language: str,
    script: str,
    instruction: str | None,
    key: str | None,
) -> list[Clip]:
    from superwhisper_api.text.client import SuperwhisperClient

    (out_dir / "cuts").mkdir(parents=True, exist_ok=True)
    scribe_fns = make_scribe_fns(key or default_key(), langs)
    client = SuperwhisperClient()
    worker = partial(
        _label_clip,
        source=audio,
        out_dir=out_dir,
        scribe_fns=scribe_fns,
        runs=runs,
        client=client,
        language=language,
        script=script,
        instruction=instruction,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        clips = list(pool.map(worker, enumerate(spans)))
    clips.sort(key=lambda c: c.index)
    return clips


def vad_path(
    audio: Path,
    *,
    out_dir: Path,
    language: str,
    script: str,
    langs: tuple[str, ...] = DEFAULT_LANGS,
    instruction: str | None = None,
    runs: int = 1,
    workers: int = 8,
    do_polish: bool = True,
    vad_kwargs: dict[str, Any] | None = None,
) -> Transcript:
    """VAD -> ensemble -> compile-down -> (join) -> polish. For dense continuous speech."""
    spans = segment_vad(audio, **(vad_kwargs or {}))
    clips = _label_spans(
        audio, spans, out_dir=out_dir, langs=langs, runs=runs, workers=workers,
        language=language, script=script, instruction=instruction, key=None,
    )
    text = " ".join(c.label for c in clips if c.label.strip())
    if do_polish and text:
        text = polish(text)
    return Transcript(text, clips)


def chunks_path(
    audio: Path,
    *,
    out_dir: Path,
    language: str,
    script: str,
    langs: tuple[str, ...] = DEFAULT_LANGS,
    instruction: str | None = None,
    chunk: float = 40.0,
    overlap: float = 10.0,
    runs: int = 1,
    workers: int = 8,
    do_polish: bool = True,
) -> Transcript:
    """Overlapping chunks -> ensemble -> compile-down -> stitch -> polish. For sparse audio."""
    spans = segment_chunks(audio_duration(audio), chunk=chunk, overlap=overlap)
    clips = _label_spans(
        audio, spans, out_dir=out_dir, langs=langs, runs=runs, workers=workers,
        language=language, script=script, instruction=instruction, key=None,
    )
    text = stitch(
        [c.label for c in clips], language=language, script=script, overlap=overlap
    )
    if do_polish and text:
        text = polish(text)
    return Transcript(text, clips)
