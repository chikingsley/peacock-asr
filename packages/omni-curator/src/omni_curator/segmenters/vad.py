"""NeMo frame-VAD segmenter: cut at silences, non-overlapping speech windows.

Right for dense, continuous speech where silences are real boundaries. For sparse / drill-style
audio (many short utterances separated by pauses) the overlapping-chunk segmenter is safer —
VAD's silence cuts and minimum-duration floor can drop the short pieces. See README.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

NEMO_VAD_MODEL_NAME = "vad_multilingual_frame_marblenet"
#: Env var pointing at a local ``.nemo`` frame-VAD checkpoint; otherwise it is downloaded.
VAD_MODEL_ENV = "OMNI_CURATOR_VAD_MODEL"


@dataclass(frozen=True)
class SpeechWindow:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def merge_windows(
    windows: list[SpeechWindow], *, merge_gap_seconds: float, hard_max_seconds: float
) -> list[SpeechWindow]:
    """Merge windows separated by <= ``merge_gap_seconds``, never exceeding ``hard_max_seconds``."""
    merged: list[SpeechWindow] = []
    for window in windows:
        if (
            merged
            and window.start - merged[-1].end <= merge_gap_seconds
            and window.end - merged[-1].start <= hard_max_seconds
        ):
            merged[-1] = SpeechWindow(merged[-1].start, window.end)
            continue
        merged.append(window)
    return merged


def boolean_windows(
    flags: list[bool],
    *,
    frame_seconds: float,
    min_duration_seconds: float,
    merge_gap_seconds: float,
    hard_max_seconds: float,
) -> list[SpeechWindow]:
    """Turn a per-frame speech/not-speech mask into merged, length-filtered speech windows."""
    raw: list[SpeechWindow] = []
    start_index: int | None = None
    for index, flag in enumerate(flags):
        if flag and start_index is None:
            start_index = index
        elif not flag and start_index is not None:
            raw.append(SpeechWindow(start_index * frame_seconds, index * frame_seconds))
            start_index = None
    if start_index is not None:
        raw.append(SpeechWindow(start_index * frame_seconds, len(flags) * frame_seconds))
    merged = merge_windows(
        raw, merge_gap_seconds=merge_gap_seconds, hard_max_seconds=hard_max_seconds
    )
    return [w for w in merged if w.duration >= min_duration_seconds]


def _load_model(model_path: Path | str | None) -> Any:
    import torch
    from nemo.collections.asr.parts.utils.vad_utils import EncDecFrameClassificationModel

    cpu = torch.device("cpu")
    path = model_path or os.environ.get(VAD_MODEL_ENV)
    if path:
        return EncDecFrameClassificationModel.restore_from(str(path), map_location=cpu)
    return EncDecFrameClassificationModel.from_pretrained(NEMO_VAD_MODEL_NAME, map_location=cpu)


def segment_vad(
    audio: Path,
    *,
    threshold: float = 0.5,
    min_dur: float = 1.0,
    merge_gap: float = 1.5,
    max_dur: float = 30.0,
    frame_seconds: float = 0.02,
    speech_class_index: int = 1,
    model_path: Path | str | None = None,
) -> list[tuple[float, float]]:
    """Speech windows from the NeMo frame-VAD (speech-bounded, language-blind, no overlap)."""
    import numpy as np
    import torch

    model = _load_model(model_path)
    model.eval()
    with torch.no_grad():
        logits = model.transcribe([str(audio)], batch_size=1, logprobs=True)[0]
    probs = torch.softmax(torch.from_numpy(np.asarray(logits)), dim=-1).numpy()
    speech = probs[:, speech_class_index]
    windows = boolean_windows(
        [float(p) >= threshold for p in speech],
        frame_seconds=frame_seconds,
        min_duration_seconds=min_dur,
        merge_gap_seconds=merge_gap,
        hard_max_seconds=max_dur,
    )
    return [(w.start, w.end) for w in windows]
