"""Audio decode + resample to the omni training format (16 kHz mono FLAC).

The omni-parquet ``audio_bytes`` column holds encoded audio as int8; everything is fed to
the model at 16 kHz mono. ``to_flac_16k`` is the canonical conversion used when building
parquet shards from arbitrary source audio (e.g. transliteration augmentation).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import librosa
import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    import numpy.typing as npt

SAMPLE_RATE = 16_000


def decode(raw: bytes) -> tuple[npt.NDArray[np.float32], int]:
    """Decode encoded audio bytes to a mono float32 waveform and its native sample rate."""
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    mono = data if data.ndim == 1 else data.mean(axis=1)
    return mono.astype(np.float32), int(sr)


def to_flac_16k(raw: bytes) -> bytes:
    """Decode arbitrary audio bytes, resample to 16 kHz mono, return FLAC-encoded bytes."""
    mono, sr = decode(raw)
    if sr != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE)
    buf = io.BytesIO()
    sf.write(buf, mono, SAMPLE_RATE, format="FLAC")
    return buf.getvalue()
