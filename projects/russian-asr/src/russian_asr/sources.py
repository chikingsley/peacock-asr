"""Russian data sources — language-specific config.

Pure data: which datasets to ingest. All curation LOGIC lives in omni-curator; ``curate.py``
wires these in. Russian is INGEST-heavy: FLEURS + Common Voice + the large local corpora below.
A YouTube channel registry can be added later if conversational scrape is wanted.

LOCAL CORPORA (under data/, migrated to /mnt/overflow): ru_open_stt, sova_dataset, TIMIT.
These are pre-labelled but in their own formats — each needs an ingest adapter (TODO) before it
lands in the store. FLEURS + Common Voice ingest via the standard factories today.
"""

from __future__ import annotations

from omni_curator.create.youtube import Channel

#: google/fleurs config for Russian.
FLEURS_CONFIG = "ru_ru"

#: Common Voice via Mozilla Data Collective — Russian dataset ids (need an MDC key). The user runs
#: their own (most-updated) Common Voice pipeline; fill the MDC id from that when ready.
COMMONVOICE: dict[str, str] = {}

#: HF-hosted clean corpora — pulled via huggingface_source (auto-detect column, 16 kHz mono FLAC,
#: forced to split=train so their own splits don't enter the FLEURS benchmark partition).
#: name -> (repo, text_column|None). ~490 h of clean Russian (the easy one-liners).
HUGGINGFACE: dict[str, tuple[str, str | None]] = {
    "rulibrispeech": ("bond005/rulibrispeech", None),          # RuLS ~98 h, audiobooks (PD)
    "sova_rudevices": ("bond005/sova_rudevices", None),        # ~101 h, manual labels
    "golos_crowd": ("bond005/sberdevices_golos_10h_crowd", None),       # Golos crowd subset
    "golos_farfield": ("bond005/sberdevices_golos_100h_farfield", None),# Golos farfield ~100 h
    "tonebooks": ("Vikhrmodels/ToneBooks", None),              # ~179 h audiobook (Tier-3)
}

#: Need adapters / non-HF sources (follow-up): full Golos 1,240 h (SberDevices/Golos or OpenSLR
#: SLR114), M-AILABS ru (~47 h, caito.de), RUSLAN (~31 h, github), and the local raw corpora below.
#: Local pre-labelled corpora (data/<name>, migrated to overflow) — need per-dataset adapters.
LOCAL_CORPORA = ("ru_open_stt", "sova_dataset", "TIMIT")

#: No YouTube scrape yet — Russian has ample labelled data. Add channels here if/when wanted.
YOUTUBE_CHANNELS: tuple[Channel, ...] = ()
