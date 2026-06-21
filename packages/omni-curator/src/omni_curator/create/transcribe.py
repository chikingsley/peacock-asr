"""Scribe ensemble: transcribe one clip several ways for the consensus fuse.

Each clip is sent to the deployed ASR service (ElevenLabs Scribe behind it) under several
language settings (e.g. ``auto`` to code-switch, plus the target language to force it) and/or
repeated runs, so cross-language differences and run-to-run variance are both visible to
``fuse.consensus_fuse``.

The service owns ASR key rotation, so there is NO ElevenLabs key handling here: a "scribe fn"
is just a bound :func:`omni_curator.swservice.transcribe_file` call (one language setting), and
:class:`ScribeError` is raised when the service reports a failure for one clip.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from omni_curator.swservice import transcribe_file

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

#: A bound single-language transcription function: a clip path -> the service ``result`` dict.
ScribeFn = Callable[["Path"], dict[str, object]]

#: Default ensemble: a single auto-detect / code-switching pass. Callers usually add the
#: target language code, e.g. ``("auto", "tgk")`` or ``("auto", "fr")``.
DEFAULT_LANGS = ("auto",)

_AUTH_ERROR_MARKERS = ("401", "unauthorized", "403", "forbidden")


class ScribeError(RuntimeError):
    """A Scribe call failed for one clip.

    The deployed service owns key rotation, so ``auth`` is informational only (it no longer
    drives an in-process renewal): a job failure surfaces as a per-clip error that the caller
    counts and skips. ``generation`` is retained for call-site compatibility.
    """

    def __init__(self, message: str, *, auth: bool = False, generation: int = 0) -> None:
        super().__init__(message)
        self.auth = auth
        self.generation = generation


def raise_for_scribe_error(result: Mapping[str, object]) -> None:
    """Raise :class:`ScribeError` if a transcription result dict carries an ``error``."""
    error = result.get("error")
    if error:
        msg = str(error)
        lowered = msg.lower()
        raise ScribeError(msg, auth=any(m in lowered for m in _AUTH_ERROR_MARKERS))


def make_scribe_fns(
    langs: tuple[str, ...] = DEFAULT_LANGS,
    *,
    model: str = "scribe-v2",
    diarize: bool = False,
) -> dict[str, ScribeFn]:
    """One bound transcription function per language setting (``auto``/``""`` -> auto-detect).

    Each function transcribes a clip via the deployed ASR service and returns the job
    ``result`` dict. The service owns key rotation, so no key is taken or threaded here.
    """

    def make_fn(lang: str) -> ScribeFn:
        language = None if lang in ("auto", "") else lang

        def fn(clip: Path) -> dict[str, object]:
            return transcribe_file(
                clip, asr_model=model, language=language, diarize=diarize
            )

        return fn

    return {lang: make_fn(lang) for lang in langs}


def transcribe_clip(clip: Path, scribe_fns: Mapping[str, ScribeFn], *, runs: int = 1) -> list[str]:
    """Run every ensemble function (``runs`` times each) over one clip; return the transcripts.

    Raises :class:`ScribeError` on an errored call (auth-classified) — an API failure must
    surface as a failure, never silently become an empty transcript / empty label.
    """
    variants: list[str] = []
    for fn in scribe_fns.values():
        for _ in range(runs):
            result = fn(clip)
            raise_for_scribe_error(result)
            transcript = str(result.get("transcript") or "").strip()
            if transcript:
                variants.append(transcript)
    return variants
