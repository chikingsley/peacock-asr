"""Fuse: turn raw ASR variants into finished text via the free SuperWhisper LLM endpoint.

- :func:`compile_down` — fuse one clip's ensemble variants into a consensus label (generative
  error correction). The default prompt targets a non-Latin language and forces ``{lang}`` into
  ``{script}`` (transliterating romanized/other-script hypotheses); pass ``instruction`` to
  override it for Latin or bilingual sources.
- :func:`transliterate` — convert text into the target script, content untouched. Used by the
  Scribe verification when a hypothesis comes back in a different script than the stored label
  (WER across scripts compares alphabets, not speech). The prompt sees ONLY the hypothesis —
  never the label — so the verification stays independent of the labeler.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superwhisper_api.text.client import SuperwhisperClient

DEFAULT_MODEL = "claude-sonnet-4-6"

TRANSCRIPT_TAG = re.compile(r"<transcript>(.*?)</transcript>", re.DOTALL)

_COMPILE_INSTRUCTION = (
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

_TRANSLITERATE_INSTRUCTION = (
    "Transliterate the following text into {lang} written in {script}. Convert the SCRIPT "
    "only — do not correct, paraphrase, translate, add or remove words. Keep any other "
    "language (e.g. English) exactly as written. Output only the result inside "
    "<transcript></transcript> tags.\n\n{text}"
)


def default_client() -> SuperwhisperClient:
    """A SuperWhisper text client (free inference) for the LLM fusion steps."""
    from superwhisper_api.text.client import SuperwhisperClient

    return SuperwhisperClient()


def extract_transcript(text: str) -> str:
    """Return the tagged transcript, or the whole text if the model omitted the tags."""
    match = TRANSCRIPT_TAG.search(text)
    return (match.group(1) if match else text).strip()


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
        client = default_client()
    body = "\n".join(f"[hypothesis {i + 1}] {v}" for i, v in enumerate(cleaned))
    prompt = (instruction or _COMPILE_INSTRUCTION).format(lang=language, script=script)
    response = client.generate(
        model, [{"role": "user", "content": f"{prompt}\n\n{body}"}], max_tokens=max_tokens
    )
    return extract_transcript(response.text)


def transliterate(
    text: str,
    *,
    language: str,
    script: str,
    model: str = DEFAULT_MODEL,
    client: SuperwhisperClient | None = None,
    max_tokens: int = 1500,
) -> str:
    """Transliterate ``text`` into ``language`` written in ``script`` (content-preserving).

    ``language``/``script`` are human-readable-ish names fed to the prompt (the FLORES code
    ``tgk_Cyrl`` + ``"Cyrillic"`` work fine). Validated on real stored pairs: matched content
    mean WER 0.30 after transliteration; a shuffled-garbage control stays >5.0 with zero false
    accepts (the model does not pull mismatched content into agreement). Raises on an empty
    model response — a scoring caller must treat that as a failed clip, never as an empty
    hypothesis.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    if client is None:
        client = default_client()
    prompt = _TRANSLITERATE_INSTRUCTION.format(lang=language, script=script, text=stripped)
    response = client.generate(model, [{"role": "user", "content": prompt}], max_tokens=max_tokens)
    result = extract_transcript(response.text)
    if not result.strip():
        msg = "transliteration returned no <transcript> content"
        raise RuntimeError(msg)
    return result
