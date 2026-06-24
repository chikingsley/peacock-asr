"""process: bring samples to the curator standard before they're trusted as training data.

Three stages applied to ``Sample``s after ingest/create, before training:
- **audio** — resample to 16 kHz mono FLAC (:mod:`omni_curator.process.audio`).
- **text** — per-language normalization (:mod:`omni_curator.process.normalize`, keyed by language).
- **language** — content-language gate (:mod:`omni_curator.process.language`), applied at export.

Tokenizer-coverage is checked at export time by the consuming project (it owns the fairseq2
tokenizer); see ``omni_curator.data.export``.
"""

from __future__ import annotations

from omni_curator.process.audio import resample_sample, resample_samples, to_16k_flac
from omni_curator.process.normalize import NORMALIZERS, base_normalize, normalize

__all__ = [
    "NORMALIZERS",
    "base_normalize",
    "normalize",
    "resample_sample",
    "resample_samples",
    "to_16k_flac",
]
