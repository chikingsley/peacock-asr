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

#: HF-native clean corpora — pulled via huggingface_source (auto-detect column, 16 kHz mono FLAC,
#: forced to split=train). Only datasets with NO purer upstream source live here (the bond005
#: reposts are deliberately NOT used — we take canonical releases for provenance).
#: name -> (repo, text_column|None).
HUGGINGFACE: dict[str, tuple[str, str | None]] = {
    "tonebooks": ("Vikhrmodels/ToneBooks", None),  # ~179 h audiobook, HF-native (no purer source)
}

#: PURE-SOURCE corpora needing a download+parse adapter (preferred over HF reposts for provenance):
#:   golos      -> OpenSLR SLR114 (https://openslr.org/114/) — full ~1,240 h, the big one
#:   rulibrispeech -> OpenSLR SLR96 (https://openslr.org/96/) — ~98 h audiobooks
#:   m_ailabs   -> caito.de M-AILABS Russian (~47 h);  ruslan -> github ruslan-corpus (~31 h)
PURE_SOURCES = ("golos_slr114", "rulibrispeech_slr96", "m_ailabs", "ruslan")

#: LOCAL pre-labelled corpora already on disk (data/<name> -> /mnt/overflow) — own formats, each
#: needs a manifest-parsing adapter. These ARE the pure source (user downloaded them).
LOCAL_CORPORA = ("ru_open_stt", "sova_dataset", "TIMIT")

#: No YouTube scrape yet — Russian has ample labelled data. Add channels here if/when wanted.
YOUTUBE_CHANNELS: tuple[Channel, ...] = ()
