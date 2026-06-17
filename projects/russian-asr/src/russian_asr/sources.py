"""Russian data sources — language-specific config.

Pure data: which datasets to ingest. All curation LOGIC lives in omni-curator; ``curate.py``
wires these in. Russian is INGEST-heavy: FLEURS + Common Voice + the large local corpora below.
A YouTube channel registry can be added later if conversational scrape is wanted.

LOCAL CORPORA (under data/, migrated to /mnt/overflow): ru_open_stt, sova_dataset, TIMIT.
These are pre-labelled but in their own formats — each needs an ingest adapter (TODO) before it
lands in the store. FLEURS + Common Voice ingest via the standard factories today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni_curator.create.youtube import Channel

#: google/fleurs config for Russian.
FLEURS_CONFIG = "ru_ru"

#: Common Voice via Mozilla Data Collective — Russian dataset ids (need an MDC key). The user runs
#: their own (most-updated) Common Voice pipeline; fill the MDC id from that when ready.
COMMONVOICE: dict[str, str] = {}

#: HF-native clean corpora — pulled via huggingface_source (auto-detect column, 16 kHz mono FLAC,
#: forced to split=train). Only datasets with NO purer upstream source live here (the bond005
#: reposts are deliberately NOT used — we take canonical releases for provenance).
#: name -> (repo, text_column|None).
HUGGINGFACE: dict[str, tuple[str, str | None]] = {
    "tonebooks": ("Vikhrmodels/ToneBooks", None),       # ~179 h audiobook, Apache-2.0
}

#: ESpeech org's 5 Russian ASR datasets (each its own repo + license). They ship as split tars with
#: per-recording <uuid>.json metadata — HF's generic loader can't pair audio<->text, so they use the
#: dedicated espeech_source (download+extract+parse). Pseudo-labelled — Scribe-verify. name -> repo.
ESPEECH: dict[str, str] = {
    "espeech_webinars": "ESpeech/ESpeech-webinars2",      # 74 GB, Apache-2.0
    "espeech_buldjat": "ESpeech/ESpeech-buldjat",         # 4.8 GB, Apache-2.0
    "espeech_upvote": "ESpeech/ESpeech-upvote",           # 26 GB, Apache-2.0
    "espeech_tuchniyzhab": "ESpeech/ESpeech-tuchniyzhab",  # 27 GB, Apache-2.0
    "espeech_podcasts": "ESpeech/ESpeech-podcasts",       # 295 GB, CC-BY-NC (personal only)
}

#: PURE-SOURCE corpora needing a download+parse adapter (preferred over HF reposts for provenance):
#:   golos      -> OpenSLR SLR114 (https://openslr.org/114/) — full ~1,240 h, the big one
#:   rulibrispeech -> OpenSLR SLR96 (https://openslr.org/96/) — ~98 h audiobooks
#:   m_ailabs   -> caito.de M-AILABS Russian (~47 h);  ruslan -> github ruslan-corpus (~31 h)
PURE_SOURCES = ("golos_slr114", "rulibrispeech_slr96", "m_ailabs", "ruslan")

#: Per-source license registry -> (license_id, commercial_use). Stamped onto every exported row
#: (license / commercial_use columns); `export --commercial-only` drops the False (NC) sources.
LICENSES: dict[str, tuple[str, bool]] = {
    "fleurs": ("CC-BY-4.0", True),
    "tonebooks": ("Apache-2.0", True),
    "espeech_webinars": ("Apache-2.0", True),
    "espeech_buldjat": ("Apache-2.0", True),
    "espeech_upvote": ("Apache-2.0", True),
    "espeech_tuchniyzhab": ("Apache-2.0", True),
    "espeech_podcasts": ("CC-BY-NC-4.0", False),
    "golos": ("CC-BY-SA-4.0", True),
    "ruls": ("PD", True),
    "sova_dataset": ("CC-BY-4.0", True),
    "ru_open_stt": ("CC-BY-NC", False),
}

#: LOCAL pre-labelled corpora already on disk (data/<name> -> /mnt/overflow) — own formats, each
#: needs a manifest-parsing adapter. LICENSE-tagged (verified 2026-06-15):
#:   sova_dataset -> CC-BY-4.0 (commercial OK; attribute Virtual Assistant LLC)
#:   ru_open_stt  -> CC-BY-NC  (NON-COMMERCIAL — poisons a commercial release; keep behind opt-in)
#: REMOVED: the local "TIMIT" dir is the original ENGLISH LDC TIMIT mislabeled as Russian — wrong
#: language AND LDC-restricted (no redistribution / no commercial / no publishing models on it).
#: Quarantined at /mnt/overflow/.../russian-asr/data/TIMIT — do NOT ingest, train, or upload it.
LOCAL_CORPORA = ("sova_dataset", "ru_open_stt")  # NC: ru_open_stt

#: No YouTube scrape yet — Russian has ample labelled data. Add channels here if/when wanted.
YOUTUBE_CHANNELS: tuple[Channel, ...] = ()
