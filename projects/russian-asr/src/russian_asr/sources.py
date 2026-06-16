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
    "espeech_webinars": ("ESpeech/ESpeech-webinars2", None),  # ~800 h, Apache-2.0 (commercial OK)
    "espeech_podcasts": ("ESpeech/ESpeech-podcasts", None),   # ~3,200 h, CC-BY-NC (personal only)
    # ^ both ESpeech are pseudo-labelled — Scribe-verify before trusting; huge (~400 GB combined).
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
