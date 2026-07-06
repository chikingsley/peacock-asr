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
#: per-clip); only pure music/song channels are skipped. Vetted 2026-06-07 and expanded from the
#: 2026-06-13 research pass. `@interpressnews` is excluded because it still 404s in yt-dlp.
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
    # -- news / TV broadcasters (2026-06-13 research pass) ------------------------------------
    _ch("maestro_tv", "@maestrotv3750", "noisy",
        "Maestro TV — news + analytical talk shows; large catalog."),
    _ch("kavkasia_tv", "@KavkasiaTelevision", "noisy",
        "Telecompany Kavkasia — political talk shows, debates, analytics."),
    _ch("postv", "@POSTV", "noisy",
        "POSTV — talk/analytical programs; pro-gov slant, Georgian-dominant."),
    _ch("postv_analytics", "@POSTV.Analytics", "noisy",
        "POSTV analytics sub-channel — long-form panel discussion."),
    _ch("publika_tv", "@publika_news", "noisy",
        "Publika — independent socio-political outlet; reports, interviews, street audio."),
    _ch("batumelebi", "@BatumelebiNews", "noisy",
        "Batumelebi — independent Batumi/Adjara newsroom; reports, interviews."),
    _ch("forbes_georgia", "@ForbesGeo", "clean",
        "Forbes Georgia — studio business interviews; small channel."),
    _ch("tv_georgian_times", "@TVGeorgianTimes", "noisy",
        "TV Georgian Times — news + reporting."),
    # -- talk / podcast / interview -----------------------------------------------------------
    _ch("nanukas_channel", "@nanukaschannel6465", "noisy",
        "Nanuka Zhorzholiani's Show — flagship social/celebrity talk show, multi-guest."),
    _ch("gattsu", "@Gattsu", "noisy",
        "Gattsu — geopolitics/culture commentary + interviews; spontaneous speech."),
    _ch("octopus", "@Octopusi", "noisy",
        "Octopus — entertainment + sports talk shows, panel formats."),
    _ch("studio_monitori", "@studiomonitori", "noisy",
        "Studio Monitor — investigative docs + on-camera interviews."),
    _ch("tabula_tv", "@TabulaTelevision", "noisy",
        "Tabula — political analysis, interviews, longform talk."),
    # -- comedy / entertainment / variety -----------------------------------------------------
    _ch("comedy_group", "@c-comedy", "noisy",
        "Comedy Group — sketches, talk shows, variety."),
    _ch("komedi_arxi", "https://www.youtube.com/channel/UCMwRaK_tGpS8oCMOJk7s5Hg",
        "noisy", "Komedi Arxi — comedy sketches/shows; small re-upload, verify size."),
    _ch("nichieri", "@nichieri", "noisy",
        "Nichieri — Georgia's Got Talent; live banter, judge/host speech."),
    _ch("colis_dakalebi", "@ColisDakalebiTELEGE", "noisy",
        "'Chemi Tsolis Dakalebi' sitcom — scripted conversational dialogue."),
    _ch("hungryman", "@hungrymantv", "noisy",
        "Hungryman — food challenges, pranks, street social experiments."),
    # -- vlogs / lifestyle --------------------------------------------------------------------
    _ch("soso_around_world", "@SosoAroundTheWorld", "noisy",
        "Soso Nebieridze — travel vlogger; narration + field audio."),
    # -- educational / children ---------------------------------------------------------------
    _ch("eduwoes", "@Eduwoes", "clean",
        "Eduwoes — educational explainers, mostly single-speaker narration."),
    _ch("emili_tv", "@EmiliTV", "noisy",
        "Emili TV — father/daughter edutainment for kids/teens, conversational."),
    _ch("jirafi_joze", "https://www.youtube.com/channel/UC2VLd27TDiI8ZTpVg2wdL7Q",
        "clean", "Giraffe Jose — children's educational narration, clean single-speaker."),
    # -- religious / literature / culture -----------------------------------------------------
    _ch("patriarchate_georgia", "@patriarchateofgeorgia797", "clean",
        "Patriarchate of Georgia — sermons, liturgy readings; some chant."),
    _ch("nplg_official", "@NPLGofficial", "clean",
        "National Parliamentary Library — audiobook readings by actors/writers."),
    _ch("voices_ancestors", "@voicesoftheancestors", "clean",
        "Polyphony podcast — partly English, language gate will filter."),
)
