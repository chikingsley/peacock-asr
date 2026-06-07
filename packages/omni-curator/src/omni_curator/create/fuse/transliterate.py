"""Script transliteration: convert text into the target language's script, content untouched.

Used by the Scribe verification when a raw hypothesis comes back in a different script than the
stored label (e.g. Scribe rendering Tajik speech in Perso-Arabic): WER across scripts is
meaningless, so the hypothesis is transliterated to the label's script first. The prompt sees
ONLY the hypothesis — never the label — so the comparison stays independent. Validated on real
stored pairs: matched content mean WER 0.30 after transliteration; shuffled-garbage control
stays >5.0 with zero false accepts (the model does not pull mismatched content into agreement).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omni_curator.create.fuse._client import default_client
from omni_curator.create.fuse._extract import extract_transcript

if TYPE_CHECKING:
    from superwhisper_api.text.client import SuperwhisperClient

DEFAULT_MODEL = "claude-sonnet-4-6"

_INSTRUCTION = (
    "Transliterate the following text into {lang} written in {script}. Convert the SCRIPT "
    "only — do not correct, paraphrase, translate, add or remove words. Keep any other "
    "language (e.g. English) exactly as written. Output only the result inside "
    "<transcript></transcript> tags.\n\n{text}"
)


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
    ``tgk_Cyrl`` + ``"Cyrillic"`` work fine). Raises on an empty model response — a scoring
    caller must treat that as a failed clip, never as an empty hypothesis.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    if client is None:
        client = default_client()
    prompt = _INSTRUCTION.format(lang=language, script=script, text=stripped)
    response = client.generate(model, [{"role": "user", "content": prompt}], max_tokens=max_tokens)
    result = extract_transcript(response.text)
    if not result.strip():
        msg = "transliteration returned no <transcript> content"
        raise RuntimeError(msg)
    return result
