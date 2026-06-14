"""Dari (Afghan Persian) data sources — language-specific config.

Pure data: which channels/local audio to pull. All curation LOGIC lives in omni-curator;
``curate.py`` wires these in. Adding/removing a source is an edit here only.

NO ingest sources exist for Dari: FLEURS has no Dari (only ``fa_ir`` Iranian, ``ps_af`` Pashto),
Common Voice has no Dari, and there is no open HF Dari ASR set. Everything comes through the create
path (YouTube + Pimsleur → Scribe labels). Do NOT wire ``fa_ir`` — that is Iranian Farsi and would
contaminate the variety.
"""

from __future__ import annotations

from omni_curator.create.youtube import Channel
from omni_curator.create.youtube import channel as _ch

#: No FLEURS / Common Voice / HF dataset for Dari — create path only.
FLEURS_CONFIG: str | None = None
COMMONVOICE: dict[str, str] = {}

#: Pimsleur Dari (clean scripted seed) — staged as a local create "channel". Both levels live in
#: the hub catalogue (catalog/Pimsleur/Dari Persian/); convert m4a -> 16 kHz mono FLAC into
#: data/create/pimsleur_dari/ and `enqueue` picks it up like any channel (Scribe labels it; the
#: booklet is reference scripting, not time-aligned, so it is not used as labels directly).
PIMSLEUR_DARI = "pimsleur_dari"

#: Dari YouTube channels (vetted 2026-06-13). Many Afghan outlets are bilingual Dari+Pashto — the
#: export language gate filters per-clip; prefer the language-split feeds. DENY-LIST (Pashto-
#: dominant, do NOT add): @lemartelevision, Shamshad TV, Zhwandoon.
YOUTUBE_CHANNELS: tuple[Channel, ...] = (
    # -- national news / current affairs (TV, Dari) -------------------------------------------
    _ch("tolonews", "@tolonews", "clean", "Afghanistan's first 24h news; Dari bulletins (Pashto interleave — filter)."),
    _ch("tolonews_talkshows", "@tolonewstalkshows", "noisy", "TOLOnews talk-show feed (Farakhabar/Mehwar); multi-speaker Dari."),
    _ch("ariana_news", "@ArianaNewsTV", "clean", "Ariana News; studio Dari bulletins (segment-filter Pashto)."),
    _ch("ariana_television", "@Ariana.Television", "noisy", "ATN flagship; magazine/social-report shows, mostly Dari."),
    _ch("one_tv_kabul", "@1TVKabul", "clean", "1TV Afghanistan, Kabul; Dari news + Cactus+ interviews."),
    _ch("afghanistan_intl", "@afintltv", "clean", "Afghanistan International; launched Persian-Dari, news + analysis."),
    _ch("amu_tv", "@AmuTelevision", "noisy", "Amu TV (Virginia); Dari news + Mawj interviews (some Pashto/diaspora)."),
    _ch("khurshid_tv", "@KhurshidTV", "noisy", "Khurshid TV, Kabul; Dari general entertainment, talk + magazine."),
    _ch("tamadon_tv", "@TamadonTV", "noisy", "Tamadon TV, Kabul; Dari, Hazara-leaning; religious + current affairs."),
    _ch("negaah_tv", "@NegaahTV", "noisy", "Negaah TV, Kabul; Dari talk + entertainment."),
    _ch("rah_e_farda", "@RaheFardaTV", "noisy", "Rah-e-Farda, Kabul; Dari talk + call-in, Hazara-Dari register."),
    # -- independent / diaspora exile media (Dari) --------------------------------------------
    _ch("hasht_e_subh", "@HashteSubhDaily", "clean", "8AM Media; independent daily, Dari reports + analysis."),
    _ch("etilaatroz", "@Etilaatroz", "clean", "Etilaatroz; investigative outlet, strongly Dari news + explainers."),
    _ch("voa_dari", "@voaafghanistandari", "clean", "VOA Dari service; news, interviews, documentary (Dari-dedicated)."),
    _ch("rta_dari", "@RTADari", "clean", "RTA Dari; state broadcaster, clean anchor read (split from RTA Pashto)."),
    _ch("azadi_radio", "@Azadiradio", "clean", "RFE/RL Radio Azadi; Dari + Pashto mixed — segment-filter for Dari."),
    _ch("begum_tv", "@BegumTelevision", "noisy", "Begum TV (Paris); women's channel, educational + talk, Dari."),
    _ch("rukhshana_media", "@RukhshanaMedia", "noisy", "Rukhshana Media; women-focused Dari interviews (+English — filter)."),
    # -- talk / podcast / interview (conversational Dari) -------------------------------------
    _ch("afghan_voice_radio", "@afghanvoiceradio", "noisy", "Afghan Voice Radio (London); Dari interviews/podcasts (diaspora)."),
    # -- vlogs / travel / documentary / daily-life (conversational Dari) ----------------------
    _ch("ebrahim_danish", "@EbrahimDanish", "noisy", "Village life/travel/food; Dari narration + street talk (filter Pashto)."),
    _ch("daricha_watan", "@DarichaWatan", "noisy", "Dareecha Watan; Afghan documentary/vlog, predominantly Dari."),
    _ch("ahmad_shahram_wafaee", "@ahmadshahramwafaee", "noisy", "Lecturer/researcher; Islamic + documentary series, Dari."),
    _ch("sunrise_afghanistan", "@SunriseAfghanistan", "noisy", "Culture + travel; Dari-dominant vlog/field audio."),
    _ch("afghan_vlogs", "@afghanvlogs", "noisy", "Daily-life vlogging in Dari (some diaspora-accent creators)."),
    _ch("afghan_daily_life", "@AfghanDailyLife", "noisy", "Urban/rural daily-life reports in Dari."),
    # -- music / culture / personalities (spoken segments only) -------------------------------
    _ch("farhad_darya", "@FarhadDaryaOfficial", "noisy", "Afghan artist; Dari interviews/spoken intros — keep talk, drop song."),
    _ch("afghan_comedy", "@AfghanComedyOfficial", "noisy", "Dari comedy skits; scripted dialogue (verify each isn't Pashto)."),
)
