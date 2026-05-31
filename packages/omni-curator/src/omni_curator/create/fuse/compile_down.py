"""Compile-down: fuse a clip's ASR variants into one consensus label (generative error correction).

Takes the several transcripts of one short clip (repeated runs and/or different language settings)
and returns the single best transcript via an LLM on the free SuperWhisper text endpoint. The
default prompt targets a non-Latin language and forces ``{lang}`` into ``{script}`` (transliterating
romanized/other-script hypotheses); pass ``instruction`` to override it for Latin or bilingual
sources (see ``groundtruth_eval`` for a French example).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omni_curator.create.fuse._extract import extract_transcript

if TYPE_CHECKING:
    from superwhisper_api.text.client import SuperwhisperClient

DEFAULT_MODEL = "claude-sonnet-4-6"

_INSTRUCTION = (
    "You are doing ASR consensus fusion (generative error correction). Below are several "
    "transcripts of the SAME short audio segment, produced by speech-to-text under different "
    "language settings and/or repeated runs. Produce the single most accurate transcript.\n"
    "Rules:\n"
    "- The target language is {lang}, written in {script}. ALWAYS write every {lang} word in "
    "{script}. If a hypothesis renders a {lang} word in a different script (romanized Latin, "
    "Perso-Arabic, etc.), convert it to {script}; NEVER leave a {lang} word in another script.\n"
    "- Keep any other language (e.g. English) EXACTLY as spoken; do NOT paraphrase or translate "
    "it.\n"
    "- Use cross-run and cross-language agreement plus linguistic sense to fix mishearings and "
    "hallucinations; correct a word even when it appears in none of the hypotheses if grammar "
    "or meaning require it.\n"
    "- Output the final transcript inside <transcript></transcript> tags and NOTHING else."
)


def compile_down(
    variants: list[str],
    *,
    language: str,
    script: str,
    model: str = DEFAULT_MODEL,
    client: SuperwhisperClient | None = None,
    max_tokens: int = 1500,
    instruction: str | None = None,
) -> str:
    """Fuse ASR transcripts of one clip into the consensus label.

    ``instruction`` overrides the default (transliteration) prompt; it is ``.format(lang=...,
    script=...)``-ed, so it may use those placeholders or ignore them.
    """
    cleaned = [v.strip() for v in variants if v and v.strip()]
    if not cleaned:
        return ""
    if client is None:
        from superwhisper_api.text.client import SuperwhisperClient as _Client

        client = _Client()
    body = "\n".join(f"[hypothesis {i + 1}] {v}" for i, v in enumerate(cleaned))
    prompt = (instruction or _INSTRUCTION).format(lang=language, script=script)
    response = client.generate(
        model, [{"role": "user", "content": f"{prompt}\n\n{body}"}], max_tokens=max_tokens
    )
    return extract_transcript(response.text)
