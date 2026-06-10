"""Tajik data sources — the language-specific config (FLEURS, Common Voice, YouTube channels).

Pure data: *which* datasets to ingest and *which* channels to pull. All curation LOGIC lives in
omni-curator; ``curate.py`` wires these into it. Adding/removing a source is an edit here only.
"""

from __future__ import annotations

from omni_curator.create.youtube import Channel
from omni_curator.create.youtube import channel as _ch

#: google/fleurs config for Tajik.
FLEURS_CONFIG = "tg_tj"

#: Common Voice via Mozilla Data Collective — dataset ids (filled when discovered; needs MDC key).
COMMONVOICE: dict[str, str] = {}

#: HuggingFace audio datasets (the legacy v0 corpora + vetted candidates), ingested via the
#: generic loader: name -> (repo, config). Common Voice 25 is HF-gated (HF_TOKEN + accepted
#: terms). Candidates not yet wired: shunyalabs/tajik-speech-dataset, WueNLP/sib-fleurs +
#: WueNLP/belebele-fleurs (config tgk_Cyrl).
HUGGINGFACE: dict[str, tuple[str, str | None]] = {
    # Augmented set; historically dragged the Scribe WER macro — the WER gate decides per clip.
    "muhtasham": ("muhtasham/tajik-asr-augmented-test", None),
}

#: Common Voice via the fsicoli HF mirror (Mozilla stopped publishing to HF at v17, which has
#: no Tajik; 22 is the newest mirror with tg). Raw-CV-layout, ingested by
#: commonvoice_hf_mirror_source.
COMMONVOICE_HF_MIRROR = ("fsicoli/common_voice_22_0", "tg")


#: Vetted in docs/tajik_youtube_channels.md + channel research. Every genuinely-Tajik channel found;
#: per-video language/quality filtering happens later in the pipeline. Only pure-music/song
#: channels are excluded; bilingual channels are IN — the tgk_Cyrl language gate (exclusive
#: letters + function-word tiebreak) drops the non-Tajik clips at export.
YOUTUBE_CHANNELS: tuple[Channel, ...] = (
    # --- clean: news / broadcast / audiobook / narration / lessons / science / stories ---
    _ch("asiaplus", "@asiaplustj", "clean", "Asia-Plus: Аудиокитоб studio audiobooks + news."),
    _ch(
        "jahonnamo",
        "UCPZvWb6IeZvqb0ORMBp-P5A",
        "clean",
        "Jahonnamo TV: anchor-read news bulletins.",
    ),
    _ch(
        "akhbori_tojikiston",
        "UC4GcNBiE59EtMaQF0kqJllg",
        "clean",
        "Ахбори Тоҷикистон news bulletins.",
    ),
    _ch("radio_ozodi", "@Radio-Ozodi", "clean", "RFE/RL Tajik service: news + shows."),
    _ch(
        "tv_tajikistan",
        "UC0NDr4nXx-c4cto_RLz9UPA",
        "clean",
        "TV TAJIKISTAN: national news uploads.",
    ),
    _ch(
        "tvt_tojikiston",
        "@TVTOJIKISTON",
        "clean",
        "TVT first state channel: news/programs (one RU bulletin).",
    ),
    _ch(
        "safina",
        "UCoKPXR1W74pKIgXzvMgf2Mg",
        "clean",
        "Televisioni Safina: culture/talk (music in places).",
    ),
    _ch(
        "ilm_va_tabiat",
        "UCRMoGKKQyKW2VCnx4O6LwDQ",
        "clean",
        "Илм ва табиат TV: science/education narration.",
    ),
    _ch(
        "zabon_omuzishi",
        "UC1A3lbraoH8suk3e13_daYA",
        "clean",
        "Забон: Tajik language lessons, clean speech.",
    ),
    _ch(
        "intellect_online",
        "UCth6mHHRSBxTsIzAtUo4I6A",
        "clean",
        "Intellect Online Tj: school grammar lessons.",
    ),
    _ch(
        "4maghz", "UCFJJHJ-Ttel1JjCoUtnLQ6A", "clean", "4maghz: Tajik history/education narration."
    ),
    _ch("hikmat_tv", "UCLz6ZBwkj6vuked4v1aFxBQ", "clean", "Hikmat TV: health/religious narration."),
    _ch(
        "tafakkur_tv",
        "UCfmKtwKyaMnB84jjRb7-I1w",
        "clean",
        "Tafakkur TV: science/health explainer narration.",
    ),
    _ch(
        "timeless_tales",
        "UC1F6CkzuinA8WrSPPQ7IrZA",
        "clean",
        "Timeless Tajik Tales: narrated kids stories.",
    ),
    _ch(
        "alifbo_afsona",
        "UCs3-_kS3W6kzr0MiVhbuRkg",
        "clean",
        "Алифбо.Афсонаҳо: alphabet, stories, riddles.",
    ),
    _ch(
        "kitob_1000",
        "UCu2DqlrzjVrkSE45s_DrsMQ",
        "clean",
        "1000 Китоб: narrated Tajik folk stories.",
    ),
    _ch(
        "quran_tajik",
        "UCIXU4ymUyAKI6XlHd6Od0kA",
        "clean",
        "Quran meanings in Tajik (recitation + translation).",
    ),
    # --- noisy: talk / podcast / interview / vlog / cooking / motivation (conversational) ---
    _ch(
        "tajik_show", "@TAJIKSHOW_OFFICIAL", "noisy", "TAJIK SHOW: talk/interview, studio audience."
    ),
    _ch(
        "alifbo_podcast",
        "UCMfbonlj1-gWdFnHSJTlMcQ",
        "noisy",
        "Alifbo Comms: long-form Tajik podcast.",
    ),
    _ch(
        "horeca_podcast",
        "UC3WE_j0wgnE61CgXq7dqMNQ",
        "noisy",
        "HoReCa: food-industry interview podcast.",
    ),
    _ch(
        "rasonanigor",
        "UCKzS0So73sl9vvbFgy_S1zw",
        "noisy",
        "РАСОНАНИГОР: long-form Tajik interviews.",
    ),
    _ch("imshab", "UCHO6nwCa8_Xw7RMr8I8ZcGA", "noisy", "ИМШАБ: late-night talk/interview."),
    _ch(
        "king_i",
        "UCitWCMzAMvslaiouAsM1wXQ",
        "noisy",
        "KING I: суҳбат бо забони тоҷикӣ, clean talk.",
    ),
    _ch(
        "your_tajikistan", "UClie_j5Akjd2_r5pnVx7lcA", "noisy", "Your Tajikistan: interviews/talk."
    ),
    _ch("lobby", "UC4jEprvdtr0nILHDwUmfmdw", "noisy", "Lobby: Tajik talk/interview."),
    _ch("two_daqiqa", "UCyM92LdXr_EnuokJ2NPXYrA", "noisy", "2 дақиқаи тоҷикӣ: short Tajik talk."),
    _ch(
        "doctor_ibrohim", "UCdZzmg7_AfaPnjXWUF_2ZBA", "noisy", "DOCTOR IBROHIM: dental/health talk."
    ),
    _ch(
        "muvaffaq_bosh",
        "UCB74EB2AocZLtMvrVrtscgg",
        "noisy",
        "MUVAFFAQ BOSH: motivational monologues.",
    ),
    _ch(
        "biznes_tj", "UCqxE0aoGKXylnc1xFEZkQbg", "noisy", "BiZNES Tj: business talk (some Russian)."
    ),
    _ch("motivshahrom", "UCTqtMEY4Jpkn5hPguVw2aeQ", "noisy", "motivshahrom: motivation/talk."),
    _ch(
        "samo_tajikistan",
        "UC3K2P_X-KDgG-MTTDqSW89w",
        "noisy",
        "Само Таджикистан: talk/entertainment.",
    ),
    _ch("lazzatho", "UCL0So6jCIn2X47UFOPk8yBA", "noisy", "Лаззатҳои Гуногун: cooking voice-over."),
    _ch("puhtupaz", "UCqxpfmG_83-f0TS1end5PeA", "noisy", "ПУХТУПАЗХОИ ТОЧИКИ: cooking voice-over."),
    _ch(
        "modari_mehrubon",
        "UC6n_zsamupItB08nt9QlenA",
        "noisy",
        "Модари Мехрубон: family/lifestyle vlogs.",
    ),
    _ch("vashgird", "UCOMGlLskQ6oMQxdcAa9r7pg", "noisy", "VASHGIRD: Tajik talk/entertainment."),
    _ch("blogi_tojikon", "UC1an9Iqjb_j9MqSKOimishw", "noisy", "Блоги Тоҷикон: Tajik vlogs."),
    _ch("najm_tv", "UCtbkheg9Y_4Gqd5fJvTPnIA", "noisy", "Najm TV: Tajik programs."),
    _ch("dfilm_tj", "UCkWYn30XkWkBG7Xz3EUQdxA", "noisy", "Dfilm tj: Tajik film/talk."),
    _ch(
        "aziya_khujand",
        "UCvnaB27ZQkp4NHdspt8occA",
        "noisy",
        "Телеканал АЗИЯ (Khujand): regional talk/news.",
    ),
    _ch("avlod_media", "UCqd3RiPrwKD-MvPh2x-4O_w", "noisy", "Avlod Media: Tajik media/talk."),
    _ch("ozodagon", "@ozodagontv", "noisy", "Ozodagon: independent news/interviews."),
    # --- bilingual language-learning (Tajik segments survive the language gate) --------------
    _ch(
        "learning_tajik_achilovs",
        "UCFGB29XZkEGS1Vw7WplBqIg",
        "clean",
        "Learning Tajik with Achilovs: English+Tajik lessons; gate keeps the Tajik.",
    ),
    _ch(
        "chris_phrases",
        "UCCIWbFzZg_lmCg51-p9OPTA",
        "clean",
        "Learning Phrases with Chris & Friends: multilingual phrase pairs incl. Tajik.",
    ),
    _ch(
        "chris_chinese",
        "UCwd_tJB2TqFdOuKzPi0fyWg",
        "clean",
        "Useful Chinese with Chris: Chinese-Tajik phrase pairs.",
    ),
    _ch(
        "chris_german",
        "UCUjEem6VD2xmJDK-q7D1BCA",
        "clean",
        "Useful German with Chris: German-Tajik phrase pairs.",
    ),
)

