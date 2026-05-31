"""Final proofreading pass: fix obvious machine glitches without touching natural speech.

One more language-model pass over the assembled transcript whose only job is to repair clear
speech-to-text artifacts (a word split into pieces or wrongly joined, an unambiguous
mis-transcription) while preserving everything that makes the transcript a faithful record of
real speech: repetitions, restarts, filler, dialect, code-switching, punctuation. The prompt is
deliberately generic (no task/dataset specifics) and conservative ("when in doubt, do nothing"),
so it applies to any language or source and cannot tidy up genuine disfluencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omni_curator.create.fuse._extract import extract_transcript

if TYPE_CHECKING:
    from superwhisper_api.text.client import SuperwhisperClient

DEFAULT_MODEL = "claude-sonnet-4-6"

_POLISH = (
    "You are proofreading an automatic speech-to-text transcript for obvious MACHINE errors only. "
    "Return the same transcript with the smallest possible number of corrections.\n"
    "FIX only clear, unambiguous speech-to-text glitches:\n"
    "- a single word wrongly split into pieces, or two words wrongly run together "
    "(numbers, names, and compounds are common victims);\n"
    "- an obvious mis-transcription where the surrounding context makes the intended word "
    "certain;\n"
    "- a word left in the wrong spelling or script when the correct form is certain from context.\n"
    "PRESERVE everything else EXACTLY — this is a record of real speech, which is messy on "
    "purpose:\n"
    "- keep all repetitions, restarts, false starts, filler words and hesitations;\n"
    "- keep the speaker's own wording, grammar, dialect, and any switching between languages;\n"
    "- keep the existing punctuation, capitalization and spacing;\n"
    "- keep the full length and order — do NOT summarize, paraphrase, translate, reorder or omit "
    "anything.\n"
    "If you are not certain a change repairs a machine error, leave the text unchanged. When in "
    "doubt, do nothing.\n"
    "Output the full transcript inside <transcript></transcript> tags and NOTHING else."
)


def polish(
    transcript: str,
    *,
    model: str = DEFAULT_MODEL,
    client: SuperwhisperClient | None = None,
    max_tokens: int = 8000,
) -> str:
    """Repair obvious machine glitches in one transcript; return it otherwise unchanged."""
    text = transcript.strip()
    if not text:
        return ""
    if client is None:
        from superwhisper_api.text.client import SuperwhisperClient as _Client

        client = _Client()
    response = client.generate(
        model, [{"role": "user", "content": f"{_POLISH}\n\n{text}"}], max_tokens=max_tokens
    )
    return extract_transcript(response.text)
