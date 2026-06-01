"""Stitch overlapping-chunk labels into one continuous transcript (LLM seam reconciliation).

The chunk segmenter keeps 100% of the audio by overlapping neighbours, so every label's head
repeats the previous label's tail — but transcribed independently, so the wording differs and
naive string dedup fails. This reconciles by MEANING: a rolling fold where the model sees the
running tail plus the next chunk and returns only the genuinely new continuation. The VAD path
has no overlap and needs none of this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omni_curator.create.fuse._client import default_client
from omni_curator.create.fuse._extract import extract_transcript

if TYPE_CHECKING:
    from superwhisper_api.text.client import SuperwhisperClient

DEFAULT_MODEL = "claude-sonnet-4-6"

_STITCH = (
    "You are reconstructing ONE continuous {lang} transcript from overlapping audio windows.\n"
    "TAIL is the end of the transcript reconstructed so far. NEXT is the following window: the "
    "audio of NEXT's BEGINNING is the SAME speech as TAIL's END (the windows overlap by about "
    "{overlap} seconds), but the two were transcribed separately so the exact wording of the "
    "overlap may differ.\n"
    "Rules:\n"
    "- Return ONLY the part of NEXT that comes AFTER the overlap with TAIL — the genuinely new "
    "speech. Do NOT repeat words already present at the end of TAIL.\n"
    "- Keep every {lang} word in {script}; keep any other language (English, etc.) verbatim. Do "
    "NOT translate or change scripts.\n"
    "- If NEXT is entirely contained in TAIL (no new speech), return nothing between the tags.\n"
    "- Output the continuation inside <transcript></transcript> tags and NOTHING else."
)


def stitch(
    labels: list[str],
    *,
    language: str,
    script: str,
    overlap: float = 10.0,
    model: str = DEFAULT_MODEL,
    client: SuperwhisperClient | None = None,
    tail_chars: int = 240,
    max_tokens: int = 1200,
) -> str:
    """Fold overlapping-chunk labels into one transcript, reconciling each seam by meaning."""
    cleaned = [t.strip() for t in labels if t and t.strip()]
    if not cleaned:
        return ""
    if client is None:
        client = default_client()
    prompt = _STITCH.format(lang=language, script=script, overlap=overlap)
    result = cleaned[0]
    for nxt in cleaned[1:]:
        tail = result[-tail_chars:]
        message = f"{prompt}\n\n[TAIL]\n{tail}\n\n[NEXT]\n{nxt}"
        response = client.generate(
            model, [{"role": "user", "content": message}], max_tokens=max_tokens
        )
        cont = extract_transcript(response.text)
        if cont:
            result = f"{result} {cont}"
    return result
