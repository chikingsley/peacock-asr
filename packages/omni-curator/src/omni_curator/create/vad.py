"""NeMo frame-VAD segmenter: cut at silences, non-overlapping speech windows.

Right for dense, continuous speech where silences are real boundaries. For sparse / drill-style
audio (many short utterances separated by pauses) the overlapping-chunk segmenter is safer —
VAD's silence cuts and minimum-duration floor can drop the short pieces. See README.
"""

from __future__ import annotations

import math
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


def split_window(
    window: SpeechWindow, *, hard_max_seconds: float, min_duration_seconds: float
) -> list[SpeechWindow]:
    """Split a window longer than ``hard_max_seconds`` into near-equal contiguous sub-windows.

    Merging caps the *length of a merge*, but a single raw speech island with no internal silence
    can still exceed ``hard_max_seconds`` — VAD found no boundary to cut at. Rather than emit one
    over-length clip (the bug: dari had 60k clips >30s, up to 777s), slice the island into
    ``ceil(duration / hard_max)`` equal contiguous chunks so every chunk is ``<= hard_max_seconds``.

    The tail must not be a sub-``min_duration_seconds`` sliver: if the final equal chunk would fall
    below the floor, it is folded into the previous chunk instead (which may then be slightly over
    ``hard_max``, but never by more than ``min_duration_seconds`` and never a dropped sliver). A
    window already within the cap is returned unchanged.
    """
    if window.duration <= hard_max_seconds:
        return [window]
    n = math.ceil(window.duration / hard_max_seconds)
    step = window.duration / n
    bounds = [window.start + i * step for i in range(n)]
    bounds.append(window.end)
    chunks = [SpeechWindow(bounds[i], bounds[i + 1]) for i in range(n)]
    # Equal slicing keeps every chunk <= hard_max and >= step; step dips below the floor only when
    # min_duration_seconds > hard_max (a misconfiguration), so guard the tail defensively anyway.
    fold_threshold = 2  # need at least two chunks to fold a sub-min tail into the previous one
    if len(chunks) >= fold_threshold and chunks[-1].duration < min_duration_seconds:
        tail = chunks.pop()
        chunks[-1] = SpeechWindow(chunks[-1].start, tail.end)
    return chunks


def boolean_windows(
    flags: list[bool],
    *,
    frame_seconds: float,
    min_duration_seconds: float,
    merge_gap_seconds: float,
    hard_max_seconds: float,
) -> list[SpeechWindow]:
    """Turn a per-frame speech/not-speech mask into merged, length-filtered speech windows.

    No output window exceeds ``hard_max_seconds``: merging caps merged length, and any *raw* island
    longer than the cap (a continuous-speech stretch with no silence) is :func:`split_window` into
    contiguous sub-windows.
    """
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
    out: list[SpeechWindow] = []
    for window in merged:
        if window.duration < min_duration_seconds:
            continue
        out.extend(
            split_window(
                window,
                hard_max_seconds=hard_max_seconds,
                min_duration_seconds=min_duration_seconds,
            )
        )
    return out


def _load_model(model_path: Path | str | None, *, device: str) -> Any:
    import torch
    from nemo.collections.asr.parts.utils.vad_utils import EncDecFrameClassificationModel

    # Loaded onto the caller's device. The segmenter runs a mix of GPU and CPU workers (NeMo's
    # high-level ``model.transcribe`` follows the model's device), so the GPU AND the cores work.
    dev = torch.device(device)
    path = model_path or os.environ.get(VAD_MODEL_ENV)
    cls = EncDecFrameClassificationModel
    if path:
        model = cls.restore_from(str(path), map_location=dev)
    else:
        model = cls.from_pretrained(NEMO_VAD_MODEL_NAME, map_location=dev)
    return model.to(dev)


#: Rough resident-VAD GPU footprint per worker (CUDA context + model + activations), for deciding
#: how many GPU workers the free VRAM can hold before spilling the rest to CPU.
_VAD_GPU_GB_PER_WORKER = 3.0


def resolve_devices(gpu_procs: int, cpu_procs: int) -> tuple[int, int]:
    """Resolve requested (gpu, cpu) worker counts against the GPU's *current* free memory.

    Honors the user's rule "don't pile onto a busy GPU": if there's no CUDA device, or another
    process already holds the VRAM, the GPU workers that won't fit are moved to the CPU instead —
    so the run still uses all the compute it can without OOM-ing the card.
    """
    if gpu_procs <= 0:
        return 0, cpu_procs
    free_gb = _gpu_free_gb()
    if free_gb is None:  # no GPU visible -> everything to CPU
        return 0, cpu_procs + gpu_procs
    affordable = int(free_gb // _VAD_GPU_GB_PER_WORKER)
    if affordable < gpu_procs:
        return max(0, affordable), cpu_procs + (gpu_procs - affordable)
    return gpu_procs, cpu_procs


def _gpu_free_gb() -> float | None:
    """Free VRAM (GB) via ``nvidia-smi`` — NVML, so it does NOT open a CUDA context in this parent
    process (which would waste VRAM and, pre-spawn, risk leaking state into workers). ``None`` if
    no GPU / nvidia-smi is present.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()
    try:
        return int(line[0]) / 1024 if line else None
    except ValueError:
        return None


def load_vad_model(model_path: Path | str | None = None, *, device: str | None = None) -> Any:
    """Load the NeMo frame-VAD model once on ``device`` (``"cuda"``/``"cpu"``; auto if ``None``).

    The split (segment) pipeline keeps one model resident per process and runs it across many files
    via :func:`segment_vad_with`.
    """
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(model_path, device=device)
    model.eval()
    return model


def segment_vad_with(
    model: Any,
    audio: Path,
    *,
    threshold: float = 0.5,
    min_dur: float = 1.0,
    merge_gap: float = 1.5,
    max_dur: float = 30.0,
    frame_seconds: float = 0.02,
    speech_class_index: int = 1,
) -> list[tuple[float, float]]:
    """Speech windows from an already-loaded frame-VAD ``model`` (see :func:`load_vad_model`)."""
    import numpy as np
    import torch

    from omni_curator.process.audio import load_16k_mono

    # NeMo frame-VAD needs MONO — a stereo file silently errors and the video fails. Load (and
    # downmix) to a 16 kHz mono array and hand transcribe the array, not the path, so stereo
    # sources segment cleanly instead of being dropped.
    samples = load_16k_mono(audio)
    with torch.no_grad():
        logits = model.transcribe([samples], batch_size=1, logprobs=True)[0]
    # transcribe() may hand back a CUDA tensor when the model is on GPU -> bring it to CPU/numpy.
    arr = logits.detach().cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
    probs = torch.softmax(torch.from_numpy(arr), dim=-1).numpy()
    speech = probs[:, speech_class_index]
    windows = boolean_windows(
        [float(p) >= threshold for p in speech],
        frame_seconds=frame_seconds,
        min_duration_seconds=min_dur,
        merge_gap_seconds=merge_gap,
        hard_max_seconds=max_dur,
    )
    return [(w.start, w.end) for w in windows]
