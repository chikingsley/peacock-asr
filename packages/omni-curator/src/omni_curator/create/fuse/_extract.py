"""Shared helper: pull the final text out of the ``<transcript>...</transcript>`` tags."""

from __future__ import annotations

import re

TRANSCRIPT_TAG = re.compile(r"<transcript>(.*?)</transcript>", re.DOTALL)


def extract_transcript(text: str) -> str:
    """Return the tagged transcript, or the whole text if the model omitted the tags."""
    match = TRANSCRIPT_TAG.search(text)
    return (match.group(1) if match else text).strip()
