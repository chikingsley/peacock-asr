"""Shared fixtures: tiny Sample factory used across the suite."""

from __future__ import annotations

import dataclasses

import pytest

from omni_curator.data.sample import Sample

_BASE = Sample(
    id="chan_vid001_0000",
    source="youtube-chan",
    language="tgk_Cyrl",
    text="ин матни тоҷикӣ аст",
    audio_path="/nonexistent/clip.flac",
    duration=5.0,
    sample_rate=16_000,
    split="train",
)


@pytest.fixture
def make_sample():
    """Factory for a minimal valid Sample; override any field via kwargs."""

    def _make(**overrides: object) -> Sample:
        return dataclasses.replace(_BASE, **overrides)  # type: ignore[arg-type]

    return _make
