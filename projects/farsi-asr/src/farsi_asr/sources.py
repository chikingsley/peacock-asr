"""Persian data sources — the language-specific config (FLEURS, Common Voice, YouTube channels).

Pure data: *which* datasets to ingest and *which* channels to pull. All curation LOGIC lives in
omni-curator; ``curate.py`` wires these into it. Adding/removing a source is an edit here only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni_curator.create.youtube import Channel

#: google/fleurs config for Persian (Iran).
FLEURS_CONFIG = "fa_ir"

#: Common Voice via Mozilla Data Collective — dataset ids (none known yet; needs an MDC key).
#: The legacy corpus ``common_voice_25`` was a local Mozilla CV v25 dump, ingested by
#: ``farsi_asr_dataset.canonical.cv25_samples`` — not an MDC id. Register
#: ``commonvoice_source(...)`` in curate.py when an MDC dataset id is found.
COMMONVOICE: dict[str, str] = {}

#: TODO: vet Persian YouTube channels for the create pipeline (none registered yet).
#: The legacy "youtube" corpus is NOT a channel registry — it is the already-chunked HF dataset
#: pourmand1376/asr-farsi-youtube-chunked-10-seconds, ingested by
#: ``farsi_asr_dataset.canonical.youtube_samples``. No vetted channel list exists anywhere in
#: the legacy code/docs, so the registry starts empty (channel policy: see NEW_LANGUAGE.md).
YOUTUBE_CHANNELS: tuple[Channel, ...] = ()
