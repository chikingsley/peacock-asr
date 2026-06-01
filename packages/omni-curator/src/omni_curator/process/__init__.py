"""process: bring samples to the curator standard before they're trusted as training data.

Three checks applied to ``Sample``s after ingest/create, before training:
- **audio** — resample to 16 kHz mono FLAC (:mod:`omni_curator.process.audio`).
- **text** — per-language normalization (pluggable: e.g. Tajik ``tajiknlp``, Persian NeMo-derived).
- **coverage** — every character must be representable by the target model's tokenizer (Omni, ...).

Audio resample is here now; text normalization + tokenizer-coverage land next.
"""

from __future__ import annotations

from omni_curator.process.audio import resample_sample, resample_samples, to_16k_flac

__all__ = ["resample_sample", "resample_samples", "to_16k_flac"]
