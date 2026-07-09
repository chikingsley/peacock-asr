"""The in-memory cutter: decode the source once, slice spans, write 16 kHz mono FLAC clips."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from omni_curator.create.queue import QVideo
from omni_curator.create.segment import _cut_clips
from omni_curator.process.audio import load_16k_mono


def test_cut_clips_decodes_once_and_slices(tmp_path) -> None:
    # 8 kHz STEREO source -> exercises both the resample-to-16k and the mono-downmix paths.
    sr = 8_000
    t = np.linspace(0.0, 4.0, sr * 4, endpoint=False, dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    src = tmp_path / "src.flac"
    sf.write(str(src), np.stack([tone, tone], axis=1), sr, format="FLAC")

    video = QVideo(video_id="vid1", channel="chan", path=str(src), tier="t1", citation="http://x")
    spans = [(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)]
    audio = load_16k_mono(src)
    clips = _cut_clips(
        video,
        spans,
        clips_root=tmp_path / "clips",
        language="tgk_Cyrl",
        script="Cyrl",
        audio=audio,
    )

    assert len(clips) == 3
    for i, c in enumerate(clips):
        info = sf.info(c.clip_path)
        assert info.samplerate == 16_000  # resampled from 8 kHz
        assert info.channels == 1  # downmixed from stereo
        assert abs(info.frames / info.samplerate - 1.0) < 0.05  # ~1.0 s clip
        assert c.clip_index == i
        assert c.clip_id == f"vid1_{i:04d}"

    data, _ = sf.read(clips[0].clip_path)
    assert np.abs(data).mean() > 0.01  # real audio, not silence
    assert not list((tmp_path / "clips").rglob(".*tmp*"))  # temp files renamed away
