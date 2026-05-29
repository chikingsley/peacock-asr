from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000


def load_audio_16k(path: str | Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    return audio


def audio_duration_seconds(path: str | Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def ensure_wav_16k(path: str | Path, output_path: str | Path) -> Path:
    audio = load_audio_16k(path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, TARGET_SAMPLE_RATE)
    return output
