"""Continuous transcript + audio -> clip-level training Samples, via VAD windows + CTC forced align.

WHY THIS EXISTS
---------------
``chunks_path`` produces a clean CONTINUOUS transcript (great for sparse / drill audio) but its
clips are OVERLAPPING fixed windows — they are *not* clip-aligned, so you cannot drop them into the
store as training pairs (the same speech is transcribed twice, and a clip's audio rarely matches
its label end-to-end). To get clip-level ``(audio, text)`` pairs you must re-cut the audio on real
boundaries and put the right words on each cut.

APPROACH (implemented here)
---------------------------
1. ``segment_vad`` gives non-overlapping speech windows = the clip BOUNDARIES we cut on.
2. A REAL CTC forced aligner — Meta's MMS (``torchaudio.pipelines.MMS_FA``, the same MMS lineage as
   the Omni model) — aligns the clean ``reference`` transcript onto the audio, yielding a PRECISE
   start/end (seconds) for every reference word. MMS_FA is multilingual via romanization: the
   reference is romanized with ``uroman`` into the aligner's 28-symbol latin alphabet, aligned, and
   the per-word frame spans are mapped back onto the ORIGINAL (un-romanized) words.
3. Each original word is assigned to the VAD window its (now accurate) midpoint falls in. Words in
   no window (VAD silence / gaps) are dropped. Per-window words, in time order, become the label.
4. Each non-empty window is cut to a 16 kHz mono FLAC and emitted as a :class:`Clip`; the
   collection is wrapped in a :class:`~omni_curator.create.pipeline.Transcript` whose
   ``.to_samples(...)`` yields clip-level store ``Sample``s.

UROMAN / DIGIT CAVEAT
---------------------
uroman leaves raw digits (and a handful of non-letter glyphs) untouched, and the MMS_FA alphabet is
``a-z`` + ``'`` only. We romanize each word individually, lowercase it, and keep only in-alphabet
characters as the ALIGNMENT token string. A word whose cleaned romanization is EMPTY (a bare number,
stray punctuation) carries no acoustic content the aligner can place, so it is dropped from the
alignment — but its ORIGINAL surface form is kept as a label and given an interpolated time between
its aligned neighbours, so clip labels stay faithful to ``reference``.

LONG AUDIO
----------
MMS_FA emits one frame per ~20 ms and forced-align is O(frames * tokens); a single call over a long
recording is both slow and memory-hungry. We therefore compute model EMISSIONS over overlapping
audio windows (:data:`_EMISSION_WINDOW_S` / :data:`_EMISSION_OVERLAP_S`), trim each window's
overlap, and concatenate the emissions into one emission matrix before a single forced-align pass.
This keeps the alignment global (no per-window token assignment guesswork) while bounding the
emission compute. The ~11.8 min Tajik narration aligns fine this way on CPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from omni_curator.create.pipeline import Clip, Transcript, cut_audio
from omni_curator.create.segmenters import segment_vad

if TYPE_CHECKING:
    from pathlib import Path

#: MMS_FA runs at 16 kHz; the bundle's emission stride is ~20 ms / frame.
_SAMPLE_RATE = 16_000
#: Emission windowing for long audio (seconds). Forced-align then runs once over the merged matrix.
_EMISSION_WINDOW_S = 30.0
_EMISSION_OVERLAP_S = 2.0


@dataclass(frozen=True)
class _AlignedWord:
    """One reference word with its forced-aligned audio timing (seconds from recording start)."""

    text: str
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


def _clean_tokens(romanized: str) -> str:
    """Lowercase a romanized word and keep only MMS_FA in-alphabet chars (``a-z`` and ``'``)."""
    return re.sub(r"[^a-z']", "", romanized.lower())


def _romanize_words(words: list[str]) -> list[str]:
    """Romanize each word to its MMS_FA token string (``""`` for unromanizable / digit words)."""
    import uroman

    roman = uroman.Uroman()
    # romanize_string returns `str` for the default RomFormat.STR; the annotation is a wider union.
    return [_clean_tokens(cast("str", roman.romanize_string(w))) for w in words]


def _emissions(audio: Path, model: Any) -> tuple[Any, float]:
    """Model emissions over the whole recording (windowed for long audio) + seconds-per-frame.

    Emissions are computed over overlapping windows, each window's overlap region trimmed, then
    concatenated so the downstream forced-align sees one continuous emission matrix.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    # soundfile (already a core dep) rather than torchaudio.load: torchaudio 2.9 routes load()
    # through torchcodec, which is not installed.
    data, sample_rate = sf.read(str(audio), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))  # (channels, samples)
    if sample_rate != _SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, _SAMPLE_RATE)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    total = waveform.shape[1]

    window = int(_EMISSION_WINDOW_S * _SAMPLE_RATE)
    overlap = int(_EMISSION_OVERLAP_S * _SAMPLE_RATE)
    chunks: list[Any] = []
    start = 0
    while start < total:
        end = min(start + window, total)
        with torch.inference_mode():
            emission, _ = model(waveform[:, start:end])
        emission = emission[0]  # (frames, tokens)
        frames = emission.shape[0]
        samples_here = end - start
        sec_per_frame = (samples_here / _SAMPLE_RATE) / frames
        # Trim the overlap tail off every window but the last so frames are not double-counted.
        if end < total:
            keep = frames - round(_EMISSION_OVERLAP_S / sec_per_frame)
            emission = emission[:keep]
        chunks.append((emission, sec_per_frame))
        if end >= total:
            break
        start += window - overlap

    sec_per_frame = chunks[0][1]
    merged = torch.cat([c[0] for c in chunks], dim=0)
    return merged, sec_per_frame


def _interpolate_dropped(
    spans: list[tuple[float, float] | None], words: list[str]
) -> list[_AlignedWord]:
    """Attach times to every word; words dropped from alignment get an interpolated slot.

    ``spans[i]`` is the aligned ``(start, end)`` for ``words[i]`` or ``None`` if that word was
    dropped (empty romanization). A run of dropped words is spread evenly across the gap between the
    last and next aligned times so the original label keeps a sensible, monotonic timestamp.
    """
    aligned: list[_AlignedWord] = []
    index = 0
    n = len(words)
    while index < n:
        span = spans[index]
        if span is not None:
            aligned.append(_AlignedWord(words[index], span[0], span[1]))
            index += 1
            continue
        run_start = index
        while index < n and spans[index] is None:
            index += 1
        # Bound the dropped run by its aligned neighbours (fall back to recording edges).
        left = aligned[-1].end if aligned else 0.0
        next_span = spans[index] if index < n else None
        right = next_span[0] if next_span is not None else left
        count = index - run_start
        step = (right - left) / (count + 1)
        for offset in range(count):
            point = left + step * (offset + 1)
            aligned.append(_AlignedWord(words[run_start + offset], point, point))
    return aligned


def forced_align_words(audio: Path, reference: str) -> list[_AlignedWord]:
    """CTC-forced-align ``reference`` onto ``audio`` (MMS_FA); return per-word times in seconds.

    The reference is split on whitespace into surface words, romanized into the MMS_FA alphabet,
    and aligned in one pass over windowed emissions. Word frame spans are converted to seconds and
    mapped back onto the ORIGINAL words; words that romanize to nothing (e.g. bare digits) are kept
    with an interpolated time so the returned list is 1:1 with ``reference``'s words.
    """
    import torchaudio

    words = reference.split()
    if not words:
        return []
    roman = _romanize_words(words)
    alignable = [(i, r) for i, r in enumerate(roman) if r]
    if not alignable:
        return []

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False)
    model.eval()
    emission, sec_per_frame = _emissions(audio, model)

    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    # torchaudio's Tokenizer.__call__ is annotated `-> List[List[str]]` but returns int token ids
    # (see Tokenizer.__call__ in torchaudio/pipelines/_wav2vec2/aligner.py), which the Aligner then
    # consumes as `List[List[int]]`; cast to reconcile the upstream stub.
    token_lists = cast("list[list[int]]", tokenizer([r for _, r in alignable]))
    token_spans = aligner(emission, token_lists)

    spans: list[tuple[float, float] | None] = [None] * len(words)
    for (orig_index, _), word_spans in zip(alignable, token_spans, strict=True):
        start = word_spans[0].start * sec_per_frame
        end = word_spans[-1].end * sec_per_frame
        spans[orig_index] = (start, end)
    return _interpolate_dropped(spans, words)


def _assign_words(
    words: list[_AlignedWord], spans: list[tuple[float, float]]
) -> list[list[_AlignedWord]]:
    """Bucket words into VAD spans by midpoint; spans are non-overlapping so each word lands once.

    A word whose midpoint falls in no span (VAD silence / a gap) is dropped — it has no clip.
    """
    buckets: list[list[_AlignedWord]] = [[] for _ in spans]
    for word in sorted(words, key=lambda w: w.start):
        mid = word.mid
        for index, (start, end) in enumerate(spans):
            if start <= mid < end:
                buckets[index].append(word)
                break
    return buckets


def _label(words: list[_AlignedWord]) -> str:
    """Join a clip's words into its label (single-spaced, in time order)."""
    return " ".join(w.text for w in words).strip()


def align_to_clips(
    audio: Path,
    *,
    out_dir: Path,
    language: str | None = None,  # noqa: ARG001 — kept for signature stability; MMS_FA is language-blind
    key: str | None = None,  # noqa: ARG001 — kept for signature stability; no Scribe call any more
    scribe_fn: Any | None = None,  # noqa: ARG001 — kept for signature stability; alignment is acoustic now
    reference: str | None = None,
    vad_kwargs: dict[str, Any] | None = None,
) -> Transcript:
    """VAD-segment ``audio``, CTC-force-align ``reference`` onto it, emit clip-level clips.

    ``reference`` is the clean continuous transcript (e.g. from ``chunks_path``) and is REQUIRED:
    its words are force-aligned onto the audio with MMS_FA, each word assigned to the VAD window its
    aligned midpoint falls in, and each non-empty window cut to a 16 kHz mono FLAC. ``language`` /
    ``key`` / ``scribe_fn`` are accepted for signature stability but unused — MMS_FA is
    language-blind (it aligns the romanized reference) and no Scribe call is made. Returns a
    :class:`Transcript`; call ``.to_samples(source=..., id_prefix=...)`` for store ``Sample``s.
    """
    if not reference or not reference.strip():
        raise ValueError("align_to_clips requires a non-empty `reference` transcript to align.")

    spans = segment_vad(audio, **(vad_kwargs or {}))
    words = forced_align_words(audio, reference)
    buckets = _assign_words(words, spans)

    (out_dir / "cuts").mkdir(parents=True, exist_ok=True)
    clips: list[Clip] = []
    for index, ((start, end), clip_words) in enumerate(zip(spans, buckets, strict=True)):
        label = _label(clip_words)
        if not label:
            continue
        clip_path = out_dir / "cuts" / f"seg_{index:04d}.flac"
        cut_audio(audio, clip_path, start, end)
        clips.append(
            Clip(
                index=index,
                start=round(start, 2),
                end=round(end, 2),
                audio_path=str(clip_path),
                variants=[label],
                label=label,
            )
        )
    text = " ".join(c.label for c in clips)
    return Transcript(text=text, clips=clips)
