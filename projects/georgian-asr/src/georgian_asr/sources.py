"""Georgian data sources — the language-specific config (FLEURS, Common Voice, YouTube channels).

Pure data: *which* datasets to ingest and *which* channels to pull. All curation LOGIC lives in
omni-curator; ``curate.py`` wires these into it. Adding/removing a source is an edit here only.
"""

from __future__ import annotations

from omni_curator.create.youtube import Channel
from omni_curator.create.youtube import channel as _ch

#: google/fleurs config for Georgian.
FLEURS_CONFIG = "ka_ge"

#: Common Voice via Mozilla Data Collective — Georgian dataset ids (need an MDC key to download).
COMMONVOICE: dict[str, str] = {
    "scripted-25": "cmn2h4m7901gzo1072qn7zoes",
    "spontaneous-3": "cmmysmqds00fwmf07e72ap8dg",
}





#: Georgian YouTube channels for the create path. Policy: a channel qualifies when a meaningful
#: share of its content is spoken Georgian (bilingual is fine — the export language gate filters
#: per-clip); only pure music/song channels are skipped. Vetted 2026-06-07 (web-searched handles).
YOUTUBE_CHANNELS: tuple[Channel, ...] = (
    # -- clean: scripted / anchor / single-speaker narration ----------------------------------
    _ch("gpb_first_channel", "https://www.youtube.com/channel/UCNFAwk1As-qyvbbm2almNpQ", "clean",
        "Georgian Public Broadcaster 1TV — news bulletins, current affairs, documentaries"),
    _ch("gpb_1tv_live", "https://www.youtube.com/@GPB1tvLive", "clean",
        "GPB 1TV live/feed channel — anchor-read news and live reports"),
    _ch("adjara_tv", "https://www.youtube.com/@AjaraTVBroadcaster", "clean",
        "Ajara public broadcaster (Batumi) — news, cultural, documentary"),
    _ch("radio_tavisupleba", "https://www.youtube.com/@Radio-Tavisupleba", "clean",
        "RFE/RL Georgian Service — scripted VO + reporting, high volume; some RU/EN segments"),
    _ch("audiobooks_geo_ka", "https://www.youtube.com/channel/UCUuogpIEDKdE17EQ45diGsg", "clean",
        "Georgian-language audiobooks — clean single-speaker narration"),
    _ch("geobooks_audio", "https://www.youtube.com/@geobooks-8895", "clean",
        "GeoBooks audiobooks — narrated literature, single-speaker"),
    _ch("tsu_university", "https://www.youtube.com/channel/UCNwzigiNbSzcm7sdffbAhqg", "clean",
        "Tbilisi State University — lectures, academic talks"),
    # -- noisy: talk shows / interviews / conversational --------------------------------------
    _ch("imedi_tv", "https://www.youtube.com/@tvimedi", "noisy",
        "TV Imedi — talk shows, live political programs, news; very large catalog"),
    _ch("rustavi2", "https://www.youtube.com/@rustavi2official", "noisy",
        "Rustavi 2 — news, talk, entertainment; large back-catalog"),
    _ch("mtavari_arkhi", "https://www.youtube.com/@mtavariarkhi", "noisy",
        "Mtavari Arkhi — ~26k videos; defunct since 2025 but catalog online"),
    _ch("formula_tv", "https://www.youtube.com/@TVFormula", "noisy",
        "Formula TV — news + analytical talk shows"),
    _ch("tv_pirveli", "https://www.youtube.com/@tvpirveli1", "noisy",
        "TV Pirveli — news + current-affairs talk"),
    _ch("palitra_news", "https://www.youtube.com/@PalitraNews", "noisy",
        "PalitraNews — 24/7 news, interviews; high upload frequency"),
    _ch("netgazeti_news", "https://www.youtube.com/@netgazeti_news", "noisy",
        "Netgazeti — independent outlet; reports, interviews, street audio"),
    _ch("setanta_adjarasport", "https://www.youtube.com/c/adjarasportofficial", "noisy",
        "Setanta Sports Georgia — live commentary, analysis, interviews"),
    _ch("science_for_everyone", "https://www.youtube.com/channel/UCx_jQeVsEwNw_BuVy-nCbiw", "noisy",
        "'Science for Everyone' podcast — long-form conversational, multi-speaker"),
    _ch("shabatis_show", "https://www.youtube.com/@shabatisshow", "noisy",
        "Shabatis Show — comedy/entertainment talk; conversational speech"),
    _ch("komedi_shou", "https://www.youtube.com/@komedishou3378", "noisy",
        "Komedi Shou (Rustavi 2 sketches) — scripted dialogue; re-upload channel, verify size"),
)

