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

#: Common Voice via Mozilla Data Collective — Russian dataset ids (need an MDC key). Fill the id
#: from the MDC catalogue (Russian scripted/spontaneous) when wiring the ingest.
COMMONVOICE: dict[str, str] = {}

#: Local pre-labelled corpora (data/<name>) — need per-dataset ingest adapters (TODO).
LOCAL_CORPORA = ("ru_open_stt", "sova_dataset", "TIMIT")

#: No YouTube scrape yet — Russian has ample labelled data. Add channels here if/when wanted.
YOUTUBE_CHANNELS: tuple[Channel, ...] = ()
