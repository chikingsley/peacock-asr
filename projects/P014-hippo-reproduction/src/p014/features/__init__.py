"""Feature extraction utilities for the HiPPO reproduction (P014).

Two extractors are exposed:

* :func:`extract_gop_for_split` — per-phone CTC-based GOP feature tensors.
* :func:`extract_ssl_utterance_for_split` — utterance-level SSL embeddings.

Both are designed to be idempotent: re-running after a successful extraction is
a no-op that just returns the cached ``.pt`` path.
"""

from __future__ import annotations

from p014.features.ctc_gop import (
    CTC_GOP_MODEL_ID,
    FeatureExtractionSettings,
    extract_gop_for_split,
    merge_gop_shards,
)
from p014.features.ssl_utterance import (
    SSL_MODEL_IDS,
    extract_ssl_utterance_for_split,
)

__all__ = [
    "CTC_GOP_MODEL_ID",
    "SSL_MODEL_IDS",
    "FeatureExtractionSettings",
    "extract_gop_for_split",
    "merge_gop_shards",
    "extract_ssl_utterance_for_split",
]
