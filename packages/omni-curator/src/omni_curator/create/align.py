"""chunks/continuous transcript -> clip-level training Samples, via VAD + word timing.

WHY THIS EXISTS
---------------
The ``chunks_path`` produces a clean CONTINUOUS transcript (great for sparse / drill audio) but
its clips are OVERLAPPING fixed windows — they are *not* clip-aligned, so you cannot drop them
into the store as training pairs (the same speech is transcribed twice, and a clip's audio rarely
matches its label end-to-end). To get clip-level ``(audio, text)`` pairs you must re-cut the audio
on real boundaries and put the right words on each cut.

APPROACH (implemented here)
---------------------------
1. ``segment_vad`` gives non-overlapping speech windows = the clip boundaries we cut on.
2. Scribe transcribes the WHOLE recording once *with word timestamps*
   (:func:`~omni_curator.create.transcribe.transcribe_words`).
3. Each word is assigned to the VAD window its **midpoint** falls in (``_assign_words``). Words in
   no window (silence / VAD gaps) are dropped. The per-window words, in time order, become that
   clip's label.
4. Each non-empty window is cut to a 16 kHz mono FLAC and emitted as a :class:`Clip`; the
   collection is wrapped in a :class:`~omni_curator.create.pipeline.Transcript` whose
   ``.to_samples(...)`` yields clip-level store ``Sample``s.

WORD-TIMING SOURCE OF TRUTH
---------------------------
ASR word timestamps (Scribe) are the alignment signal. The optional ``reference`` (e.g. the clean
stitched ``chunks_path`` transcript) is used only as a *light reconciliation*: per clip, if the
ASR-assigned text and the corresponding slice of the reference differ trivially, we keep the ASR
words but the reference is available to a caller for QA. We do NOT force-align the reference onto
the audio (see LIMITATIONS).

LIMITATIONS / WHAT A BETTER VERSION NEEDS
-----------------------------------------
- This is *approximate* forced alignment by ASR word timing, not true acoustic forced alignment.
  Word boundaries are only as good as Scribe's timestamps; on noisy / overlapping speech they
  drift, and a word straddling a VAD boundary lands wholly in one clip by its midpoint.
- Long recordings exceed Scribe's per-request limits. A production version would transcribe in
  overlapping chunks and merge word streams (dedup the overlap by time), instead of one call.
- The clean ``reference`` transcript is the higher-quality text but is NOT time-aligned. A better
  version would run a real forced aligner (e.g. CTC-segmentation / MFA / WhisperX) to map the
  reference words onto the audio, then cut on those boundaries — giving clip labels with the clean
  transcript's wording AND tight timing. That needs an acoustic model + pronunciation handling per
  language and is out of scope here; this module is the working approximation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omni_curator.audio import cut_audio
from omni_curator.create.pipeline import Clip, Transcript
from omni_curator.create.segmenters import segment_vad
from omni_curator.create.transcribe import (
    TimedWord,
    scribe_word_fn,
    transcribe_words,
)

if TYPE_CHECKING:
    from pathlib import Path


def _assign_words(
    words: list[TimedWord], spans: list[tuple[float, float]]
) -> list[list[TimedWord]]:
    """Bucket words into spans by midpoint; spans are non-overlapping so each word lands once.

    Words are pre-sorted by start, and within a span kept in time order. A word whose midpoint
    falls in no span (VAD silence / a gap) is dropped — it has no clip to belong to.
    """
    buckets: list[list[TimedWord]] = [[] for _ in spans]
    for word in sorted(words, key=lambda w: w.start):
        mid = word.mid
        for index, (start, end) in enumerate(spans):
            if start <= mid < end:
                buckets[index].append(word)
                break
    return buckets


def _label(words: list[TimedWord]) -> str:
    """Join a clip's words into its label (single-spaced, in time order)."""
    return " ".join(w.text for w in words).strip()


def align_to_clips(
    audio: Path,
    *,
    out_dir: Path,
    language: str | None = None,
    key: str | None = None,
    scribe_fn: Any | None = None,
    reference: str | None = None,  # noqa: ARG001 — reserved for QA reconciliation; see module docstring
    vad_kwargs: dict[str, Any] | None = None,
) -> Transcript:
    """VAD-segment ``audio``, assign Scribe word timings to each window, emit clip-level clips.

    Pass ``scribe_fn`` to reuse a bound transcription fn (tests inject a fake); otherwise one is
    built from ``key`` (resolved via the usual ElevenLabs key chain) and ``language`` (``None`` =
    auto-detect). ``reference`` is accepted for future reconciliation but is not force-aligned
    here. Returns a :class:`Transcript`; call ``.to_samples(source=..., id_prefix=...)`` to get
    store ``Sample``s.
    """
    spans = segment_vad(audio, **(vad_kwargs or {}))
    if scribe_fn is None:
        from omni_curator.create.transcribe import default_key

        scribe_fn = scribe_word_fn(key or default_key(), language)
    words = transcribe_words(audio, scribe_fn)
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
