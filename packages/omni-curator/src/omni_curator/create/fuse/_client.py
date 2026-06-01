"""Shared SuperWhisper text-client factory for the fuse steps (compile_down / stitch / polish)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superwhisper_api.text.client import SuperwhisperClient


def default_client() -> SuperwhisperClient:
    """A SuperWhisper text client (free inference) for the LLM fusion steps."""
    from superwhisper_api.text.client import SuperwhisperClient

    return SuperwhisperClient()
