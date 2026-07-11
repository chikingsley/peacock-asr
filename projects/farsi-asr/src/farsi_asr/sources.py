"""Persian data sources — the language-specific config (FLEURS, Common Voice, YouTube channels).

Pure data: *which* datasets to ingest and *which* channels to pull. All curation LOGIC lives in
omni-curator; ``curate.py`` wires these into it. Adding/removing a source is an edit here only.
"""

from __future__ import annotations

from dataclasses import dataclass

from omni_curator.create.youtube import Channel
from omni_curator.create.youtube import channel as _ch


@dataclass(frozen=True)
class CorpusSource:
    """A prepared or ingestable source in the Persian ASR project."""

    name: str
    source_config: str
    raw_path: str
    canonical_path: str
    split_origin: str
    license: str
    label_provenance: str
    note: str = ""


#: google/fleurs config for Persian (Iran).
FLEURS_CONFIG = "fa_ir"

#: Common Voice via Mozilla Data Collective — dataset ids (none known yet; needs an MDC key).
#: The legacy corpus ``common_voice_25`` was a local Mozilla CV v25 dump, ingested by
#: ``farsi_asr_dataset.canonical.cv25_samples`` — not an MDC id. Register
#: ``commonvoice_source(...)`` in curate.py when an MDC dataset id is found.
COMMONVOICE: dict[str, str] = {}

#: Full source registry for the prepared Persian corpora. Some of these are legacy canonical
#: corpora whose loaders still live in the old farsi_asr_dataset path; ``curate.py`` wires only the
#: new omni-curator ingest sources that have been ported.
CANONICAL_CORPORA: tuple[CorpusSource, ...] = (
    CorpusSource(
        name="common_voice_25",
        source_config="mozilla_data_collective/cv-corpus-25.0/fa",
        raw_path="data/raw/mozilla_data_collective",
        canonical_path="data/common_voice_25",
        split_origin="native",
        license="cc0-1.0",
        label_provenance="External Mozilla Common Voice community/human reference text.",
        note="Local Mozilla CV v25 dump; MDC dataset id wiring is still empty.",
    ),
    CorpusSource(
        name="fleurs",
        source_config="google/fleurs/fa_ir",
        raw_path="data/raw/fleurs_fa_ir_omni",
        canonical_path="data/fleurs",
        split_origin="native",
        license="cc-by-4.0",
        label_provenance="External Google FLEURS official reference text.",
        note="Currently the only prepared corpus wired through omni-curator ingest.",
    ),
    CorpusSource(
        name="mana_tts",
        source_config="MahtaFetrat/Mana-TTS",
        raw_path="data/raw/mana_tts",
        canonical_path="data/mana_tts",
        split_origin="derived",
        license="cc0-1.0",
        label_provenance="External synthetic/TTS source text; local cleanup metadata includes ASR match quality.",
    ),
    CorpusSource(
        name="neyshekar",
        source_config="Peacockery/neyshekar-v3-asr-aligned",
        raw_path="data/raw/neyshekar_v3_asr_aligned",
        canonical_path="data/neyshekar",
        split_origin="asr_aligned_derived",
        license="cc0-1.0",
        label_provenance="External Neyshekar source references after local repair/alignment; not Scribe or YouTube pseudo labels.",
        note="ASR-aligned means the local mirror was segmented/repaired so audio rows align to transcript rows; it is processing lineage, not ownership.",
    ),
    CorpusSource(
        name="worldspeech",
        source_config="disco-eth/WorldSpeech/fa_ir",
        raw_path="data/raw/worldspeech_fa_ir",
        canonical_path="data/worldspeech",
        split_origin="native_test_plus_derived_dev",
        license="cc-by-nc-4.0",
        label_provenance="External WorldSpeech references; local metadata includes ASR/CER/DNSMOS fields.",
    ),
    CorpusSource(
        name="youtube",
        source_config="pourmand1376/asr-farsi-youtube-chunked-10-seconds",
        raw_path="data/raw/asr_farsi_youtube_pourmand1376",
        canonical_path="data/youtube",
        split_origin="native",
        license="unknown",
        label_provenance="External legacy chunked YouTube corpus references; source URLs retained.",
        note="Separate from YOUTUBE_CHANNELS, which is the new create-pipeline channel registry.",
    ),
    CorpusSource(
        name="thomcles",
        source_config="Thomcles/Persian-Farsi-Speech",
        raw_path="data/raw/thomcles_persian_omni",
        canonical_path="data/thomcles",
        split_origin="derived",
        license="unknown",
        label_provenance="External Thomcles Persian-Farsi-Speech references after local Omni parquet prep.",
    ),
)

#: Persian YouTube channels for the create pipeline. The legacy "youtube" corpus is NOT a channel registry — it is the already-chunked HF dataset
#: pourmand1376/asr-farsi-youtube-chunked-10-seconds, ingested by
#: ``farsi_asr_dataset.canonical.youtube_samples``.
#:
#: Seeded from the earlier Persian/Dari research pass and maintained here as executable config.
#: Iranian Persian only;
#: Dari stays in ``projects/dari-asr``. Domestic Iranian outlets may need the project cookie/VPN
#: setup; per-clip language/quality gates handle code-switching, music beds, and unusable uploads.
YOUTUBE_CHANNELS: tuple[Channel, ...] = (
    # --- News: diaspora / international Persian networks ---------------------------------------
    _ch(
        "iran_international",
        "@IRANINTL",
        "noisy",
        "24/7 Persian news; anchors, live call-ins, panel debate, field reports.",
    ),
    _ch(
        "manoto",
        "@manototv",
        "clean",
        "London-based Persian general/news network; studio bulletins and documentaries.",
    ),
    _ch(
        "bbc_persian",
        "@BBCNewsPersian",
        "clean",
        "BBC Persian service; studio anchors, interviews, analysis.",
    ),
    _ch("voa_farsi", "@VOAFarsi", "clean", "Voice of America Persian; news bulletins and reports."),
    _ch(
        "voa_persian_ofogh",
        "@voacapitol",
        "noisy",
        "VOA Persian Ofogh talk/panel show; multi-guest debate and call-in.",
    ),
    _ch(
        "radio_farda", "@radiofarda", "clean", "RFE/RL Persian; news reads, explainers, interviews."
    ),
    _ch("dw_persian", "@dwpersian", "clean", "Deutsche Welle Persian; news and reportage."),
    _ch(
        "euronews_persian",
        "@euronewspe",
        "clean",
        "Euronews in Persian; voiced-over world-news bulletins.",
    ),
    _ch(
        "independent_persian",
        "@IranIndependentTV",
        "noisy",
        "Independent Persian / Iran Independent TV; interviews, panels, talk shows.",
    ),
    _ch(
        "iranwire",
        "@iranwire",
        "noisy",
        "IranWire citizen journalism; field reports, interviews, mixed audio.",
    ),
    _ch(
        "kayhan_london",
        "@KayhanLondonOnline",
        "clean",
        "Kayhan London online; news segments and interviews.",
    ),
    _ch(
        "radio_zamaneh",
        "@Radiozamanehplus",
        "noisy",
        "Radio Zamaneh; interviews, discussion, reportage.",
    ),
    _ch(
        "iran_tv_network",
        "@IranTVNetwork",
        "noisy",
        "LA-based Persian satellite network; interviews and talk programming.",
    ),
    _ch(
        "channel_one",
        "UCHhiGJozZgFKidRv5IL4c6g",
        "noisy",
        "Channel One TV: diaspora nightly talk and viewer call-ins.",
    ),
    # --- News: domestic Iranian outlets; may need Iran-friendly network/cookies -----------------
    _ch(
        "isna",
        "@ISNAVideo",
        "noisy",
        "Iranian Students' News Agency; studio talks, interviews, field clips.",
    ),
    _ch(
        "khabar_online",
        "@KHABARONLINIR",
        "noisy",
        "Khabar Online; interviews and panel discussion.",
    ),
    _ch("entekhab", "@EntekhabNewsSite", "noisy", "Entekhab; video interviews and debates."),
    _ch("rouydad24", "@rouydad2443", "noisy", "Rouydad24; news analysis and interviews."),
    _ch(
        "etemad_online",
        "@etemadonline",
        "noisy",
        "Etemad Online; reformist daily interviews and discussion.",
    ),
    _ch(
        "irna", "@IRNA_en", "clean", "IRNA state news agency; Persian news despite the _en handle."
    ),
    _ch(
        "didban_iran",
        "@didbaniran-newsagency",
        "noisy",
        "Didban Iran; investigative interviews and reports.",
    ),
    _ch(
        "hammihan", "@hammihanonline", "noisy", "Ham-Mihan reformist daily; interviews and reports."
    ),
    _ch("fars_news", "@farsna", "noisy", "Fars News video desk; reports and interviews."),
    # --- Call-in / interview / podcast / debate -------------------------------------------------
    _ch(
        "holakouee",
        "UCEUO9scRXBptMsPycZygmbg",
        "noisy",
        "Dr. Holakouee: call-in psychology show with many unscripted callers.",
    ),
    _ch(
        "poletik",
        "UCEt1wJTpIl4oKMEEuT1V3vw",
        "noisy",
        "Poletik / Kambiz Hosseini: weekly political interviews.",
    ),
    _ch("tabaghe16", "@Tabaghe16", "noisy", "Tabaghe 16: long-form interview podcast."),
    _ch("king_raam", "@KingRaam", "noisy", "Masty o Rasty: informal long-form interview podcast."),
    _ch(
        "channelb_podcast",
        "@channelbpodcast",
        "noisy",
        "ChannelB: long-form Persian narrative/documentary podcast.",
    ),
    _ch(
        "bplus_podcast",
        "@bpluspodcast",
        "noisy",
        "BPlus: Persian non-fiction book-summary / ideas podcast.",
    ),
    _ch(
        "radio_cafe", "@radiocafee", "noisy", "Radio Cafe; conversational podcast, double-e handle."
    ),
    _ch(
        "radio_marz",
        "@radiomarz",
        "noisy",
        "Radio Marz; narrative/conversational social-topics podcast.",
    ),
    _ch(
        "soma_podcast",
        "@somapodcast",
        "noisy",
        "Soma; conversational psychology and mental-health podcast.",
    ),
    _ch(
        "aknon_podcast",
        "@aknonpodcast",
        "noisy",
        "Aknon; interview podcast on people's work and life stories.",
    ),
    _ch("rokh_podcast", "@rokhpodcast", "noisy", "Rokh; biographical and historical podcast."),
    _ch(
        "bedahe_podcast",
        "@bedahepodcast",
        "noisy",
        "Bedahe; unscripted two-host conversational podcast.",
    ),
    _ch("chai_plus", "@ChaiPlus", "noisy", "Chai Plus; conversational/interview podcast."),
    _ch(
        "everything_podcast",
        "@everythingpodcast",
        "noisy",
        "Everything / Hamechiz; wide-topic conversational podcast.",
    ),
    _ch("raft_podcast", "@raftpodcast", "noisy", "Raft; conversational podcast."),
    _ch("movarekh_podcast", "@movarekhpodcast", "noisy", "Movarekh; Persian history podcast."),
    _ch("ravikade", "@ravikade", "clean", "Ravikade; single-host narrative-history podcast."),
    _ch(
        "dialogue_box",
        "@DialogueBoxMedia",
        "noisy",
        "Dialogue Box; cinema, literature, and music conversations.",
    ),
    _ch("jabe_siah", "@jabesiah", "noisy", "Jabe Siah; arts and culture review podcast."),
    _ch(
        "dastan_podcast", "@dastanpodcast", "clean", "Dastan; single-narrator storytelling podcast."
    ),
    _ch(
        "jorjandi",
        "@jorjandi",
        "noisy",
        "Mohammad Jorjandi: cybercrime/scam exposes and spontaneous commentary.",
    ),
    _ch(
        "jadi",
        "UCgKePkWtPuF36bJy0n2cEMQ",
        "noisy",
        "Jadi: tech/freedom monologues and livestreams.",
    ),
    # --- Talk shows / interviews / debate -------------------------------------------------------
    _ch("dorehami", "@Dorehami", "noisy", "Dorehami: comedy talk show with celebrity interviews."),
    _ch(
        "khandevaneh",
        "@khandevane",
        "noisy",
        "Khandevaneh: comedy/variety show, stand-up, music, interviews.",
    ),
    _ch("hamrafigh", "@Hamrafigh", "noisy", "Hamrafigh: long-form one-on-one artist interviews."),
    _ch("iran_talk", "@irantalks", "noisy", "IranTalk: weekly long-form interview show."),
    _ch(
        "studio_paat",
        "@Studio_patt",
        "noisy",
        "Studio Paat: political, economic, and sports interviews/debates.",
    ),
    _ch(
        "football_360",
        "@football_360",
        "noisy",
        "Football 360: football talk, analysis, interviews, panels.",
    ),
    _ch(
        "cafe_khabar",
        "@cafekhabarofficial",
        "noisy",
        "Cafe Khabar: Tehran news/interview channel with long conversations.",
    ),
    _ch(
        "ecoiran",
        "@ecoiran",
        "noisy",
        "EcoIran: economics outlet with studio interviews and expert discussions.",
    ),
    _ch(
        "didar_news",
        "@didarnews",
        "noisy",
        "Didar News: independent interviews and political conversation.",
    ),
    _ch("ali_sabouri", "UCbGE6dIy5S7RxovBbEGzwsg", "noisy", "Ali Sabouri: comedy talk show."),
    _ch(
        "reyvandi",
        "@hassan_reyvandi",
        "noisy",
        "Hasan Reyvandi: stand-up clips with crowd noise and music intros.",
    ),
    # --- Street interviews / vlogs / daily life -------------------------------------------------
    _ch("iranopedia", "@iranopedia", "noisy", "Street interviews and vox-pop filmed in Iran."),
    _ch(
        "dar_shahr",
        "UCh15eeQnv1zL5l4zBwKaVYg",
        "noisy",
        "Dar Shahr: street interviews and blind-date conversations in Iran.",
    ),
    _ch(
        "sedaye_mardom",
        "@Sedaye_mardom1",
        "noisy",
        "Sedaye Mardom: small street-interview channel.",
    ),
    _ch(
        "golshid",
        "UChHzEpcNzipFcDAV9LezYww",
        "noisy",
        "Golshid: migration interviews and live Q&A.",
    ),
    _ch(
        "iran_village_cooking",
        "@Iran.village.Cooking",
        "noisy",
        "Northern-Iran village cooking/daily-life vlog with conversational Farsi.",
    ),
    _ch(
        "iran_village_life",
        "@Iran_village_life",
        "noisy",
        "Northern-Iran village cooking vlog with family conversation.",
    ),
    _ch(
        "villagehouse_golbanoo",
        "@Villagehouse_golbanoo",
        "noisy",
        "Kurdistan-Iran rural-life cooking vlog with conversational Persian.",
    ),
    _ch(
        "putak",
        "UCa20QQoV4gaLv9XRki-ynWQ",
        "noisy",
        "PUTAK: entertainment commentary and youth slang.",
    ),
    _ch(
        "sogang", "UCxym7GWKXPXYISgy7U-AOYw", "noisy", "SoGang: vlogs/challenges and youth banter."
    ),
    _ch(
        "miaplays", "@Miaplays", "noisy", "Mia Plays: vlogs and gaming with some English mixed in."
    ),
    _ch(
        "javad_javadi",
        "UCqu8mtd9NBjgSaHUtHjRiLw",
        "noisy",
        "Chef Javad Javadi: cooking vlogs with running commentary.",
    ),
    _ch(
        "mr_taster",
        "UCAkoSMgQOmGyXYYg-HWnmEg",
        "noisy",
        "mr taster: street-food vlogs; recent uploads may trend English.",
    ),
    # --- Audiobooks / narration ------------------------------------------------------------------
    _ch("avas_book_club", "@AvasBookClub", "clean", "Avas Book Club: Persian narrated audiobooks."),
    _ch("ketabe_goya", "@ketabegoya8305", "clean", "Ketabe Goya: Persian audiobook channel."),
    _ch(
        "taaghche_audiobook",
        "@taaghchee-bookandaudiobook1279",
        "clean",
        "Taaghche audiobook/e-book platform clips.",
    ),
    _ch("ketabsoti", "@ketabsoti", "clean", "Persian audiobooks and narrated literature."),
    _ch("navar_media", "@navarmedia", "clean", "Navar Iranian audiobook/podcast platform outlet."),
    _ch(
        "mashaledanesh_audiobooks",
        "UCG-NudOQ-LqfPfWhUR2DE4Q",
        "clean",
        "Mashale Danesh audiobooks; single-narrator Persian literature.",
    ),
    _ch(
        "persian_audio_books",
        "UCZJFid4Yt-WQiuukRC72wbw",
        "clean",
        "Persian Audio Books; Farsi classics and modern literature.",
    ),
    _ch(
        "donyaye_ketab_soti",
        "UCfdc4HIplyS66VGURD7BjlA",
        "clean",
        "Donyaye Ketab Soti: Persian audiobooks and storytelling.",
    ),
    _ch(
        "iranseda_ketab_soti",
        "UCczbnOdlq_GZAzYp2Ll1hWg",
        "clean",
        "IranSeda audiobook channel; Persian radio-style narration.",
    ),
    _ch(
        "mobonews",
        "UCal5H0vgmsKcwMKiUflVpEw",
        "clean",
        "MoboNews: studio gadget reviews and voice-over.",
    ),
    _ch("zoomit", "@ZoomitTV", "noisy", "Zoomit: tech reviews and studio debates."),
    # --- Educational / lectures ------------------------------------------------------------------
    _ch(
        "roshd_channel",
        "@ROSHDCHANNEL",
        "clean",
        "Roshd Channel: children's stories, read-aloud books, educational content.",
    ),
    _ch("maktabkhooneh_yt", "@MaktabkhoonehYT", "clean", "Maktabkhooneh course and lecture clips."),
    _ch("faradars_free", "@Faradars_Free", "clean", "FaraDars free lecture content."),
    _ch("faradars", "@faradars1", "clean", "FaraDars main online-ed lecture channel."),
    _ch("alaa_sharif", "@SanatiSharifIr", "clean", "Alaa: exam-prep lectures, single instructor."),
    _ch(
        "khan_academy_farsi",
        "@KhanAcademyFarsi",
        "clean",
        "Khan Academy Farsi educational lessons.",
    ),
    _ch(
        "maktabkhooneh",
        "UCFwPa1mBpYCUmWUnvp03LAw",
        "clean",
        "Maktabkhooneh MOOC; university lectures.",
    ),
    _ch(
        "iran_academia",
        "UCRLzQ330Tkize2MvnAfauaQ",
        "clean",
        "Iran Academia; Persian academic lectures in social sciences/humanities.",
    ),
    _ch(
        "utv_tehran",
        "@شبکهدانشگاهتهران",
        "clean",
        "University of Tehran TV: formal lectures, interviews, cultural programming.",
    ),
    _ch(
        "jahaneirani_ut",
        "@jahaneirani_ut",
        "clean",
        "University of Tehran Iranian-world center: humanities and culture talks.",
    ),
    _ch("sharif_ee", "@SharifEE", "clean", "Sharif EE: STEM lectures, seminars, and interviews."),
    _ch(
        "iasa_ut",
        "@iasaut",
        "clean",
        "University of Tehran architecture/humanities webinars and long-form talks.",
    ),
    _ch(
        "miras_maktoob",
        "@miras.maktoob",
        "clean",
        "Miras Maktoob: Persian manuscript and cultural-scholarship talks.",
    ),
    _ch("azad_social", "@azadsocial", "noisy", "Azad Social: public talks, panels, and debates."),
    _ch("hokmran_tv", "@HokmranTV", "noisy", "Hokmran TV: public-affairs interviews and debates."),
    _ch(
        "clubfars",
        "@Clubfars",
        "noisy",
        "Clubfars: topic-focused Persian cultural and public conversations.",
    ),
    _ch(
        "otaq_e_jang", "@otaqejang", "noisy", "Otaq-e Jang: political debate and interview speech."
    ),
    _ch(
        "bijan_abdolkarimi",
        "@bijan.abdolkarimi",
        "clean",
        "Bijan Abdolkarimi: philosophy lectures and debates.",
    ),
    _ch(
        "mostafa_malekian",
        "@MostafaMalekian_official",
        "clean",
        "Mostafa Malekian: clean philosophy and culture lectures.",
    ),
    _ch(
        "soroush_dabbagh",
        "@Soroushdabbagh_Official",
        "clean",
        "Soroush Dabbagh: philosophy, religion, and culture lectures/interviews.",
    ),
    _ch(
        "jafekri", "@jafekri", "noisy", "Jafekri: Persian self-improvement podcast and interviews."
    ),
    _ch(
        "mehran_rowshan_persian",
        "@MehranRowshanPersian",
        "noisy",
        "Mehran Rowshan Persian: interviews and monologues; some English/code-switching.",
    ),
    _ch(
        "solmaz_barghgir",
        "@solmazbarghgir",
        "noisy",
        "Solmaz Barghgir: Persian relationship/coaching podcast archive.",
    ),
    _ch(
        "radio_joloun",
        "@RadioJoloun",
        "noisy",
        "Radio Joloun: travel podcast with conversational interviews.",
    ),
    _ch(
        "deep_podcast",
        "@DeepPodcast",
        "clean",
        "Deep Podcast: Persian documentary/history narration.",
    ),
    _ch(
        "resetsho",
        "@resetsho",
        "noisy",
        "Resetsho: social and psychology interview podcast; sensitive topics possible.",
    ),
    _ch("chenarestan", "@chenarestan", "noisy", "Chenarestan: psychology/interview podcast."),
    _ch(
        "chaos_farsi",
        "@ChaosFarsi",
        "clean",
        "Chaos Farsi: long-form history/storytelling narration.",
    ),
    _ch(
        "radio_sedaye_aramesh",
        "@radiosedayearamesh",
        "clean",
        "Radio Sedaye Aramesh: book summaries and clean narration.",
    ),
    _ch(
        "hamidreza_faryaghoobi",
        "@HamidrezaFaryaghoobi",
        "clean",
        "Hamidreza Faryaghoobi: audiobook/storytelling narration.",
    ),
    _ch(
        "bookolony",
        "@bookolony",
        "clean",
        "Bookolony: Persian audiobooks; rights/provenance need filtering.",
    ),
    _ch(
        "ghese_goo",
        "@Ghesee",
        "clean",
        "Ghese Goo: Persian children's story narration; filter songs/rhymes.",
    ),
    _ch(
        "khalenaghmeh",
        "@Khalenaghmeh",
        "clean",
        "Children's stories with Khale Naghmeh; filter background music.",
    ),
    _ch(
        "iran_tarikh", "@Irantarix", "clean", "Tarikh Iran: Persian history/documentary narration."
    ),
    _ch(
        "sedaye_tarikh",
        "@sedaye.tarikh",
        "clean",
        "Sedaye Tarikh: Persian history/documentary narration.",
    ),
    # --- Documentary / culture / religious lectures ---------------------------------------------
    _ch(
        "iran_documentary",
        "@IranDocumentary",
        "clean",
        "Persian documentary narration; screen for music beds.",
    ),
    _ch(
        "avaye_buf",
        "@avayebuf",
        "noisy",
        "Avaye Buf: Persian audiobook and cultural discussion channel.",
    ),
    _ch(
        "raefipour",
        "@raefipour_official",
        "clean",
        "Raefipour official lectures; single-speaker Persian.",
    ),
    _ch(
        "panahian", "@PanahianFarsi", "clean", "Panahian Farsi religious lectures; single speaker."
    ),
    _ch(
        "qaraati_quran",
        "UC5ilho5Q-YSIn3dn1FTcWNw",
        "clean",
        "Qaraati Quran lessons; Persian lecture/tafsir.",
    ),
    _ch(
        "darolerfan_ir",
        "@darolerfan_ir",
        "clean",
        "Hossein Ansarian / Darol Erfan: official Persian religious lectures.",
    ),
    _ch(
        "fateminia_media",
        "@fateminiamedia7283",
        "clean",
        "Fateminia Media: archival Persian sermon and lecture speech.",
    ),
    _ch("pooyaee", "@Pooyaee", "clean", "Pooyaee: cultural and social long-form lectures."),
    _ch(
        "farhangsaz",
        "@farhangsaz",
        "clean",
        "Farhangsaz: philosophy and cultural talks/debates; provenance varies.",
    ),
    _ch(
        "masnavi_manavi",
        "@masnavimanavi5428",
        "clean",
        "Masnavi Manavi: Rumi/Masnavi lecture speech.",
    ),
)
