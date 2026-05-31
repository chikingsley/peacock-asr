"""Scribe ensemble: transcribe one clip several ways for the compile-down to fuse.

Each clip is sent to ElevenLabs Scribe under several language settings (e.g. ``auto`` to
code-switch, plus the target language to force it) and/or repeated runs, so cross-language
differences and run-to-run variance are both visible to ``fuse.compile_down``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

#: Default ensemble: a single auto-detect / code-switching pass. Callers usually add the
#: target language code, e.g. ``("auto", "tgk")`` or ``("auto", "fr")``.
DEFAULT_LANGS = ("auto",)


def default_key() -> str:
    """Resolve the ElevenLabs key (env -> macOS cache -> Mac-mirror)."""
    from superwhisper_api.auth import ensure_elevenlabs_key

    return ensure_elevenlabs_key()


def make_scribe_fns(
    key: str,
    langs: tuple[str, ...] = DEFAULT_LANGS,
    *,
    model: str = "scribe-v2",
    diarize: bool = False,
) -> dict[str, Any]:
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


def transcribe_clip(clip: Path, scribe_fns: dict[str, Any], *, runs: int = 1) -> list[str]:
    """Run every ensemble function (``runs`` times each) over one clip; return the transcripts."""
    variants: list[str] = []
    for fn in scribe_fns.values():
        for _ in range(runs):
            transcript = str(fn(clip).as_dict().get("transcript") or "").strip()
            if transcript:
                variants.append(transcript)
    return variants
