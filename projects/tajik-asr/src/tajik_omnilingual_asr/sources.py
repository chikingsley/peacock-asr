"""Tajik data sources — the language-specific config (FLEURS, Common Voice, YouTube channels).

Pure data: *which* datasets to ingest and *which* channels to pull. All curation LOGIC lives in
omni-curator; ``curate.py`` wires these into it. Adding/removing a source is an edit here only.
"""

from __future__ import annotations

from dataclasses import dataclass

#: google/fleurs config for Tajik.
FLEURS_CONFIG = "tg_tj"

#: Common Voice via Mozilla Data Collective — dataset ids (filled when discovered; needs MDC key).
COMMONVOICE: dict[str, str] = {}


@dataclass(frozen=True)
class Channel:
    """A vetted Tajik YouTube source: where to pull + how clean to expect it."""

    slug: str
    url: str
    tier: str  # "clean" = scripted/single-speaker (chunks->align) | "noisy" = conversational (VAD)
    note: str


#: Vetted in docs/tajik_youtube_channels.md + channel research. Clean/scripted/pure-Tajik first.
YOUTUBE_CHANNELS: tuple[Channel, ...] = (
    # --- Tier 1: clean, single-speaker, scripted (the best ASR material) ---
    Channel(
        "asiaplus", "https://www.youtube.com/@asiaplustj", "clean",
        "Asia-Plus: Аудиокитоб studio audiobook series + news (filter Russian titles).",
    ),
    Channel(
        "jahonnamo", "https://www.youtube.com/channel/UCPZvWb6IeZvqb0ORMBp-P5A", "clean",
        "Jahonnamo TV: 24/7 anchor-read national news bulletins.",
    ),
    Channel(
        "akhbori_tojikiston", "https://www.youtube.com/channel/UC4GcNBiE59EtMaQF0kqJllg", "clean",
        "Daily anchor-read 'Ахбори Тоҷикистон ва ҷаҳон' bulletins.",
    ),
    Channel(
        "radio_ozodi", "https://www.youtube.com/@Radio-Ozodi", "clean",
        "RFE/RL Tajik service: anchor segments clean, field reports noisier.",
    ),
    Channel(
        "ilm_va_tabiat", "https://www.youtube.com/channel/UCRMoGKKQyKW2VCnx4O6LwDQ", "clean",
        "Илм ва табиат TV: Tajik science/education, formal lecture narration.",
    ),
    # --- Tier 2: noisier but definitely Tajik (conversational register) ---
    Channel(
        "tajik_show", "https://www.youtube.com/@TAJIKSHOW_OFFICIAL", "noisy",
        "TAJIK SHOW: talk/interview, multi-speaker studio audience.",
    ),
    Channel(
        "alifbo_podcast", "https://www.youtube.com/channel/UCMfbonlj1-gWdFnHSJTlMcQ", "noisy",
        "Alifbo Comms: long-form Tajik podcast, native conversational speech.",
    ),
)

CHANNELS_BY_SLUG = {c.slug: c for c in YOUTUBE_CHANNELS}
