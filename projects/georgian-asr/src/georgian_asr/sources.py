"""Georgian data sources — the language-specific config (FLEURS, Common Voice, YouTube channels).

Pure data: *which* datasets to ingest and *which* channels to pull. All curation LOGIC lives in
omni-curator; ``curate.py`` wires these into it. Adding/removing a source is an edit here only.
"""

from __future__ import annotations

from dataclasses import dataclass

#: google/fleurs config for Georgian.
FLEURS_CONFIG = "ka_ge"

#: Common Voice via Mozilla Data Collective — Georgian dataset ids (need an MDC key to download).
COMMONVOICE: dict[str, str] = {
    "scripted-25": "cmn2h4m7901gzo1072qn7zoes",
    "spontaneous-3": "cmmysmqds00fwmf07e72ap8dg",
}


@dataclass(frozen=True)
class Channel:
    """A vetted YouTube source: where to pull + how clean to expect it."""

    slug: str
    url: str
    tier: str  # "clean" = scripted/single-speaker (chunks->align) | "noisy" = conversational (VAD)
    note: str


#: Georgian YouTube channels for the create path (none vetted yet — ingest covers Georgian for now).
YOUTUBE_CHANNELS: tuple[Channel, ...] = ()

CHANNELS_BY_SLUG = {c.slug: c for c in YOUTUBE_CHANNELS}
