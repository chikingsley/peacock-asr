"""Scribe ensemble: transcribe one clip several ways for the compile-down to fuse.

Each clip is sent to ElevenLabs Scribe under several language settings (e.g. ``auto`` to
code-switch, plus the target language to force it) and/or repeated runs, so cross-language
differences and run-to-run variance are both visible to ``fuse.compile_down``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from superwhisper_api.audio.transcribe import ProcessFn

#: Default ensemble: a single auto-detect / code-switching pass. Callers usually add the
#: target language code, e.g. ``("auto", "tgk")`` or ``("auto", "fr")``.
DEFAULT_LANGS = ("auto",)

_AUTH_ERROR_MARKERS = ("401", "unauthorized", "403", "forbidden")


class ScribeError(RuntimeError):
    """A Scribe call returned an error result. ``auth=True`` flags a dead/unauthorized key.

    Callers MUST treat ``auth`` errors as run-level events, never per-clip noise: renew the
    key (:func:`renew_scribe_key`) or abort. Retrying a dead key thousands of times is how
    a key source gets burned. ``generation`` carries the key generation the failing call was
    made with — consumers that renew mid-run use it to tell a stale in-flight failure (old
    generation, already handled) from a failure of the current key.
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


def default_key() -> str:
    """Resolve the ElevenLabs key (env -> last self-minted key -> macOS cache -> Mac-mirror)."""
    from superwhisper_api.auth import ensure_elevenlabs_key

    return ensure_elevenlabs_key()


def renew_scribe_key() -> str:
    """Mint a fresh batch key via the Superwhisper proxy and make it the process default.

    Sets ``ELEVENLABS_API_KEY`` so every later :func:`default_key` in this process resolves
    to the renewed key; the mint also persists it for future processes. Raises if the proxy
    refuses — callers must then abort, not keep calling with the dead key.
    """
    import os

    from superwhisper_api.auth import mint_elevenlabs_batch_key

    key = mint_elevenlabs_batch_key()
    os.environ["ELEVENLABS_API_KEY"] = key
    return key


def make_scribe_fns(
    key: str,
    langs: tuple[str, ...] = DEFAULT_LANGS,
    *,
    model: str = "scribe-v2",
    diarize: bool = False,
) -> dict[str, ProcessFn]:
    """One bound transcription function per language setting (``auto``/``""`` -> auto-detect)."""
    from superwhisper_api.audio.models import audio_model
    from superwhisper_api.audio.transcribe import create_process_fn

    spec = audio_model(model)
    return {
        lang: create_process_fn(
            spec, key, language=(None if lang in ("auto", "") else lang), diarize=diarize
        )
        for lang in langs
    }


def transcribe_clip(clip: Path, scribe_fns: Mapping[str, ProcessFn], *, runs: int = 1) -> list[str]:
    """Run every ensemble function (``runs`` times each) over one clip; return the transcripts.

    Raises :class:`ScribeError` on an errored call (auth-classified) — an API failure must
    surface as a failure, never silently become an empty transcript / empty label.
    """
    variants: list[str] = []
    for fn in scribe_fns.values():
        for _ in range(runs):
            result = fn(clip).as_dict()
            raise_for_scribe_error(result)
            transcript = str(result.get("transcript") or "").strip()
            if transcript:
                variants.append(transcript)
    return variants
