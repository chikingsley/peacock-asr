"""Compile-down: fuse a segment's ASR variants into the final training label.

In ASR each example is ``(audio, text)`` — the *text is the label*. This takes the several
speech-to-text transcripts of one short audio segment (repeated runs and/or different
language settings, optionally plus our own model's output) and returns the single consensus
transcript via generative error correction.

It runs **claude-sonnet-4-6 through the SuperWhisper text endpoint** — free inference, and on
this task it matches paid codex gpt-5.5 (mini/nano fail the Cyrillic conversion). The model
is told to output the transcript inside ``<transcript>`` tags so we can strip any preamble.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superwhisper_api.text.client import SuperwhisperClient

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_LANGUAGE = "Tajik"
DEFAULT_SCRIPT = "Cyrillic script (tgk_Cyrl)"

_INSTRUCTION = (
    "You are doing ASR consensus fusion (generative error correction). Below are several "
    "transcripts of the SAME short audio segment, produced by speech-to-text under different "
    "language settings and/or repeated runs. Produce the single most accurate transcript.\n"
    "Rules:\n"
    "- The target language is {lang}. ALWAYS write every {lang} word in {script}, "
    "transliterating from whatever script a hypothesis uses (romanized Latin or Perso-Arabic). "
    "NEVER leave {lang} in Latin or Arabic script — the output {lang} MUST be in {script}.\n"
    "- Keep any other language (e.g. English) EXACTLY as spoken; do NOT paraphrase or translate "
    "it.\n"
    "- Use cross-run and cross-language agreement plus linguistic sense to fix mishearings and "
    "hallucinations; correct a word even when it appears in none of the hypotheses if grammar "
    "or meaning require it.\n"
    "- Output the final transcript inside <transcript></transcript> tags and NOTHING else."
)

_TAG = re.compile(r"<transcript>(.*?)</transcript>", re.S)


def compile_down(
    variants: list[str],
    *,
    language: str = DEFAULT_LANGUAGE,
    script: str = DEFAULT_SCRIPT,
    model: str = DEFAULT_MODEL,
    client: SuperwhisperClient | None = None,
    max_tokens: int = 1500,
) -> str:
    """Fuse ASR transcripts of one segment into the consensus label."""
    cleaned = [v.strip() for v in variants if v and v.strip()]
    if not cleaned:
        return ""
    if client is None:
        from superwhisper_api.text.client import SuperwhisperClient as _Client

        client = _Client()
    body = "\n".join(f"[hypothesis {i + 1}] {v}" for i, v in enumerate(cleaned))
    instruction = _INSTRUCTION.format(lang=language, script=script)
    response = client.generate(
        model,
        [{"role": "user", "content": f"{instruction}\n\n{body}"}],
        max_tokens=max_tokens,
    )
    match = _TAG.search(response.text)
    return (match.group(1) if match else response.text).strip()
