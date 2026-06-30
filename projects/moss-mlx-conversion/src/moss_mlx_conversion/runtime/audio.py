from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from moss_mlx_conversion.paths import CACHE_DIR


def load_waveform(audio_path: Path | None, *, sample_rate: int = 16_000) -> tuple[np.ndarray, Path]:
    if audio_path is None:
        audio_path = default_fixture_path()

    waveform, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    return np.asarray(waveform, dtype=np.float32), audio_path


def default_fixture_path() -> Path:
    fixture_dir = CACHE_DIR / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    target = fixture_dir / "librosa-libri1-16k.wav"
    if target.exists():
        return target

    source = Path(librosa.ex("libri1"))
    waveform, _ = librosa.load(str(source), sr=16_000, mono=True)
    sf.write(target, waveform, 16_000)
    return target
