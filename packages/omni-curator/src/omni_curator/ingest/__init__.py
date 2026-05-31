"""ingest: pull datasets that ALREADY have transcripts into standardized ``Sample``s.

The counterpart to :mod:`omni_curator.create` (which generates labels for raw audio). One loader
per source family, because each comes a different way:

- :mod:`omni_curator.ingest.huggingface` — datasets from the HF Hub (FLEURS, ...).
- :mod:`omni_curator.ingest.commonvoice` — Common Voice, pulled DIRECT from Mozilla (not the Hub).

Every loader yields ``Sample``s so the rest of the pipeline (process -> store -> benchmark) is
source-agnostic.
"""

from __future__ import annotations
