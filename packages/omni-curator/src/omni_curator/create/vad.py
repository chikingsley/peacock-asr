"""Multi-engine VAD contract and deterministic shared interval post-processing.

Adapters accept one decoded 16 kHz mono float array and return engine-native speech intervals.
They do not decode files, cut clips, pad boundaries, merge gaps, or enforce the ASR model's hard
duration limit.  Every adapter is followed by :func:`postprocess_intervals`, so engine comparisons
and production segmentation use one observable policy instead of three subtly different ones.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, replace
from importlib import import_module, metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

SAMPLE_RATE = 16_000
NEMO_VAD_MODEL_NAME = "vad_multilingual_frame_marblenet"
VAD_MODEL_ENV = "OMNI_CURATOR_VAD_MODEL"
MARBLENET_V2_SHA256 = "84bda37e925ac6fd740c2ced55642cb79f94f81348e1fa0db992ca50d4b09706"
BENCHMARK_MARBLENET_PATH = Path(
    "/home/simon/docker/a-vad-bench/models/marblenet-v2/frame_vad_multilingual_marblenet_v2.0.nemo"
)
POLICY_REVISION = "peacock-multi-vad-v1"

VadEngineName = Literal["cobra", "silero", "marblenet"]
Interval = tuple[float, float]


@dataclass(frozen=True, slots=True)
class SpeechWindow:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class PostprocessProfile:
    """The shared interval policy applied after every engine prediction."""

    name: str
    min_speech_s: float
    merge_gap_s: float
    pad_start_s: float
    pad_end_s: float
    max_speech_s: float
    cap_merges_at_max_speech: bool

    def __post_init__(self) -> None:
        values = (
            self.min_speech_s,
            self.merge_gap_s,
            self.pad_start_s,
            self.pad_end_s,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("VAD post-processing durations must be finite and non-negative")
        if not math.isfinite(self.max_speech_s) or self.max_speech_s <= 0:
            raise ValueError("VAD max_speech_s must be finite and positive")

    def as_dict(self) -> dict[str, str | float]:
        return {
            "name": self.name,
            "min_speech_s": self.min_speech_s,
            "merge_gap_s": self.merge_gap_s,
            "pad_start_s": self.pad_start_s,
            "pad_end_s": self.pad_end_s,
            "max_speech_s": self.max_speech_s,
            "cap_merges_at_max_speech": self.cap_merges_at_max_speech,
        }


def postprocess_profile(name: str, *, max_speech_s: float = 30.0) -> PostprocessProfile:
    """Resolve a named shared profile.

    ``legacy-marblenet-v1`` preserves the curator's pre-migration boundaries and remains the
    production default until a bounded pilot selects a new policy. ``conservative-v1`` mirrors
    the common 250/100/30 ms policy used by the VAD benchmark and is the fair pilot profile.
    """
    profiles = {
        "legacy-marblenet-v1": PostprocessProfile(
            name="legacy-marblenet-v1",
            min_speech_s=1.0,
            merge_gap_s=1.5,
            pad_start_s=0.0,
            pad_end_s=0.0,
            max_speech_s=max_speech_s,
            cap_merges_at_max_speech=True,
        ),
        "conservative-v1": PostprocessProfile(
            name="conservative-v1",
            min_speech_s=0.25,
            merge_gap_s=0.1,
            pad_start_s=0.03,
            pad_end_s=0.03,
            max_speech_s=max_speech_s,
            cap_merges_at_max_speech=False,
        ),
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"unknown VAD profile {name!r}; choose one of {sorted(profiles)}") from exc


@dataclass(frozen=True, slots=True)
class VadPolicy:
    """Serializable engine + postprocessor selection stamped onto every emitted clip."""

    engine: VadEngineName = "marblenet"
    threshold: float = 0.5
    profile: PostprocessProfile = PostprocessProfile(
        name="legacy-marblenet-v1",
        min_speech_s=1.0,
        merge_gap_s=1.5,
        pad_start_s=0.0,
        pad_end_s=0.0,
        max_speech_s=30.0,
        cap_merges_at_max_speech=True,
    )
    model_path: str | None = None
    silero_backend: str = "auto"
    policy_revision: str = POLICY_REVISION

    def __post_init__(self) -> None:
        if self.engine not in {"cobra", "silero", "marblenet"}:
            raise ValueError(f"unsupported VAD engine: {self.engine}")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("VAD threshold must be between 0 and 1")
        if self.silero_backend not in {"auto", "onnx", "jit"}:
            raise ValueError("Silero backend must be one of: auto, onnx, jit")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "threshold": self.threshold,
            "model_path": self.model_path,
            "silero_backend": self.silero_backend,
            "postprocess": self.profile.as_dict(),
            "policy_revision": self.policy_revision,
        }

    @property
    def profile_id(self) -> str:
        raw = json.dumps(
            self.canonical_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
        return f"vad-{hashlib.sha256(raw).hexdigest()[:16]}"

    def as_dict(self) -> dict[str, object]:
        return {"profile_id": self.profile_id, **self.canonical_dict()}


def build_vad_policy(
    *,
    engine: VadEngineName = "marblenet",
    profile: str = "legacy-marblenet-v1",
    max_speech_s: float = 30.0,
    threshold: float = 0.5,
    model_path: str | Path | None = None,
    silero_backend: str = "auto",
) -> VadPolicy:
    return VadPolicy(
        engine=engine,
        threshold=threshold,
        profile=postprocess_profile(profile, max_speech_s=max_speech_s),
        model_path=str(model_path) if model_path is not None else None,
        silero_backend=silero_backend,
    )


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    raw_intervals: tuple[Interval, ...]
    intervals: tuple[Interval, ...]
    audio_seconds: float


@runtime_checkable
class VadEngine(Protocol):
    name: str
    model_revision: str
    runtime_metadata: dict[str, object]

    def predict(self, audio: np.ndarray, sample_rate: int) -> Sequence[Interval]: ...

    def close(self) -> None: ...


def split_window(window: SpeechWindow, *, hard_max_seconds: float) -> list[SpeechWindow]:
    """Split a long interval into near-equal contiguous pieces under the hard cap."""
    if not math.isfinite(hard_max_seconds) or hard_max_seconds <= 0:
        raise ValueError("hard_max_seconds must be finite and positive")
    if window.duration <= hard_max_seconds:
        return [window]
    count = math.ceil(window.duration / hard_max_seconds)
    step = window.duration / count
    bounds = [window.start + index * step for index in range(count)] + [window.end]
    return [SpeechWindow(bounds[index], bounds[index + 1]) for index in range(count)]


def merge_windows(
    windows: list[SpeechWindow], *, merge_gap_seconds: float, hard_max_seconds: float
) -> list[SpeechWindow]:
    """Legacy boolean-mask helper retained for compatibility and regression tests."""
    merged: list[SpeechWindow] = []
    for window in windows:
        if (
            merged
            and window.start - merged[-1].end <= merge_gap_seconds
            and window.end - merged[-1].start <= hard_max_seconds
        ):
            merged[-1] = SpeechWindow(merged[-1].start, max(merged[-1].end, window.end))
        else:
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
    """Turn frame decisions into windows with the historical MarbleNet semantics."""
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
    return [
        piece
        for window in merged
        if window.duration >= min_duration_seconds
        for piece in split_window(window, hard_max_seconds=hard_max_seconds)
    ]


def postprocess_intervals(
    intervals: Sequence[Interval],
    *,
    audio_seconds: float,
    profile: PostprocessProfile,
) -> list[Interval]:
    """Sanitize, clamp, pad, sort, merge, filter, and hard-split engine intervals."""
    if not math.isfinite(audio_seconds) or audio_seconds < 0:
        raise ValueError("audio_seconds must be finite and non-negative")
    prepared: list[SpeechWindow] = []
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        clamped_start = max(0.0, min(audio_seconds, start))
        clamped_end = max(0.0, min(audio_seconds, end))
        if clamped_end <= clamped_start:
            continue
        prepared.append(
            SpeechWindow(
                max(0.0, clamped_start - profile.pad_start_s),
                min(audio_seconds, clamped_end + profile.pad_end_s),
            )
        )
    prepared.sort(key=lambda item: (item.start, item.end))
    merged: list[SpeechWindow] = []
    for window in prepared:
        can_merge = (
            merged
            and window.start - merged[-1].end <= profile.merge_gap_s
            and (
                not profile.cap_merges_at_max_speech
                or max(merged[-1].end, window.end) - merged[-1].start <= profile.max_speech_s
            )
        )
        if can_merge:
            merged[-1] = SpeechWindow(merged[-1].start, max(merged[-1].end, window.end))
        else:
            merged.append(window)
    emitted: list[Interval] = []
    for window in merged:
        if window.duration < profile.min_speech_s:
            continue
        emitted.extend(
            (piece.start, piece.end)
            for piece in split_window(window, hard_max_seconds=profile.max_speech_s)
        )
    return emitted


class CobraEngine:
    name = "cobra"

    def __init__(self, *, threshold: float, device: str) -> None:
        try:
            pvcobra = import_module("pvcobra")
        except ImportError as exc:
            raise RuntimeError(
                "Cobra VAD is not installed; run `uv sync --extra vad-cobra` for omni-curator"
            ) from exc
        access_key = os.environ.get("PICOVOICE_API_KEY")
        if not access_key:
            raise RuntimeError("PICOVOICE_API_KEY is not set")
        kwargs: dict[str, str] = {"access_key": access_key}
        cobra_device = _pvcobra_device(device)
        if cobra_device is not None:
            kwargs["device"] = cobra_device
        self._cobra = pvcobra.create(**kwargs)
        self._threshold = threshold
        self.model_revision = (
            f"pvcobra-wheel-{_package_version('pvcobra')}:engine-{self._cobra.version}"
        )
        self.runtime_metadata: dict[str, object] = {
            "device": cobra_device,
            "pvcobra_wheel": _package_version("pvcobra"),
            "cobra_engine": self._cobra.version,
            "native_options": {"threshold": threshold, "frame_samples": self._cobra.frame_length},
        }

    def predict(self, audio: np.ndarray, sample_rate: int) -> list[Interval]:
        import numpy as np

        _require_16k(sample_rate, self.name)
        frame_length = self._cobra.frame_length
        pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        flags = [
            self._cobra.process(pcm[offset : offset + frame_length].tolist()) >= self._threshold
            for offset in range(0, len(pcm) - frame_length + 1, frame_length)
        ]
        return _bool_frames_to_intervals(flags, frame_length / sample_rate)

    def close(self) -> None:
        delete = getattr(self._cobra, "delete", None)
        if callable(delete):
            delete()


class SileroEngine:
    name = "silero"

    def __init__(self, *, threshold: float, backend: str, device: str) -> None:
        try:
            silero_vad = import_module("silero_vad")
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD is not installed; run `uv sync --extra vad-silero` for omni-curator"
            ) from exc
        self._backend = _resolve_silero_backend(backend=backend, device=device)
        self._device = device
        self._torch: Any | None = None
        if self._backend == "onnx":
            self._model = silero_vad.load_silero_vad(onnx=True)
        else:
            self._torch = import_module("torch")
            self._model = silero_vad.load_silero_vad(onnx=False).to(device)
        self._get_speech_timestamps = silero_vad.get_speech_timestamps
        self._threshold = threshold
        asset = Path(silero_vad.__file__).parent / "data" / f"silero_vad.{self._backend}"
        self.model_revision = (
            f"silero-vad-{_package_version('silero-vad')}:{self._backend}:sha256:{_sha256(asset)}"
        )
        self.runtime_metadata: dict[str, object] = {
            "device": device,
            "backend": self._backend,
            "silero_vad": _package_version("silero-vad"),
            "torch": _package_version("torch"),
            "onnxruntime": _package_version("onnxruntime") if self._backend == "onnx" else None,
            "native_options": {
                "threshold": threshold,
                "min_speech_duration_ms": 0,
                "max_speech_duration_s": "infinity",
                "min_silence_duration_ms": 0,
                "speech_pad_ms": 0,
                "negative_threshold": "silero-default",
            },
        }

    def predict(self, audio: np.ndarray, sample_rate: int) -> list[Interval]:
        import numpy as np

        _require_16k(sample_rate, self.name)
        model_input: Any = audio.astype(np.float32, copy=False)
        if self._torch is not None:
            model_input = self._torch.from_numpy(model_input).to(self._device)
        timestamps = self._get_speech_timestamps(
            model_input,
            self._model,
            sampling_rate=sample_rate,
            threshold=self._threshold,
            min_speech_duration_ms=0,
            max_speech_duration_s=float("inf"),
            min_silence_duration_ms=0,
            speech_pad_ms=0,
        )
        return [(item["start"] / sample_rate, item["end"] / sample_rate) for item in timestamps]

    def close(self) -> None:
        return


class MarbleNetEngine:
    name = "marblenet"

    def __init__(self, *, threshold: float, model_path: str | None, device: str) -> None:
        torch = import_module("torch")
        nemo_asr = import_module("nemo.collections.asr")
        chosen = _resolve_marblenet_path(model_path)
        cls = nemo_asr.models.EncDecFrameClassificationModel
        self._model = cls.restore_from(str(chosen), map_location=torch.device(device), strict=False)
        self.model_revision = f"{chosen.name}:sha256:{_sha256(chosen)}"
        self._torch = torch
        self._device = device
        self._threshold = threshold
        self._model.to(device)
        self._model.eval()
        self.runtime_metadata: dict[str, object] = {
            "device": device,
            "torch": _package_version("torch"),
            "nemo_toolkit": _package_version("nemo-toolkit"),
            "native_options": {"threshold": threshold, "speech_class_index": 1},
        }

    def predict(self, audio: np.ndarray, sample_rate: int) -> list[Interval]:
        _require_16k(sample_rate, self.name)
        signal = self._torch.from_numpy(audio).unsqueeze(0).float().to(self._device)
        signal_length = self._torch.tensor([signal.shape[1]]).long().to(self._device)
        with self._torch.no_grad():
            logits = self._model(input_signal=signal, input_signal_length=signal_length)
            probabilities = self._torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
        if len(probabilities) == 0:
            return []
        frame_seconds = len(audio) / sample_rate / len(probabilities)
        return _bool_frames_to_intervals(
            [float(value) >= self._threshold for value in probabilities], frame_seconds
        )

    def close(self) -> None:
        return


def load_vad_engine(policy: VadPolicy, *, device: str | None = None) -> VadEngine:
    """Preflight and load exactly the adapter selected by ``policy``."""
    resolved_device = device or _automatic_device()
    if policy.engine == "cobra":
        return CobraEngine(threshold=policy.threshold, device=resolved_device)
    if policy.engine == "silero":
        return SileroEngine(
            threshold=policy.threshold,
            backend=policy.silero_backend,
            device=resolved_device,
        )
    return MarbleNetEngine(
        threshold=policy.threshold, model_path=policy.model_path, device=resolved_device
    )


def segment_audio_with(
    engine: VadEngine,
    audio: np.ndarray,
    *,
    policy: VadPolicy,
    sample_rate: int = SAMPLE_RATE,
) -> SegmentationResult:
    """Predict raw intervals and apply the policy's shared postprocessor."""
    raw = tuple((float(start), float(end)) for start, end in engine.predict(audio, sample_rate))
    audio_seconds = len(audio) / sample_rate
    emitted = tuple(postprocess_intervals(raw, audio_seconds=audio_seconds, profile=policy.profile))
    return SegmentationResult(raw, emitted, audio_seconds)


def segmentation_metadata(
    policy: VadPolicy, engine: VadEngine, result: SegmentationResult
) -> dict[str, object]:
    """Compact, downstream-safe provenance stored in every clip's metadata."""
    return {
        **policy.as_dict(),
        "policy_id": policy.profile_id,
        "profile_id": effective_profile_id(
            policy, engine.model_revision, runtime_metadata=engine.runtime_metadata
        ),
        "model_revision": engine.model_revision,
        "runtime": engine.runtime_metadata,
        "sample_rate": SAMPLE_RATE,
        "audio_seconds": round(result.audio_seconds, 6),
        "raw_interval_count": len(result.raw_intervals),
        "emitted_interval_count": len(result.intervals),
    }


def effective_profile_id(
    policy: VadPolicy,
    model_revision: str,
    *,
    runtime_metadata: dict[str, object] | None = None,
) -> str:
    """Hash effective policy plus the exact runtime/model revision used to produce clips."""
    value = {
        **policy.canonical_dict(),
        "model_revision": model_revision,
        "runtime": runtime_metadata or {},
    }
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return f"vad-{hashlib.sha256(raw).hexdigest()[:16]}"


def _bool_frames_to_intervals(flags: Sequence[bool], frame_seconds: float) -> list[Interval]:
    intervals: list[Interval] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            intervals.append((start * frame_seconds, index * frame_seconds))
            start = None
    if start is not None:
        intervals.append((start * frame_seconds, len(flags) * frame_seconds))
    return intervals


def _require_16k(sample_rate: int, engine: str) -> None:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{engine} adapter expects 16 kHz audio")


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_marblenet_path(model_path: str | None) -> Path:
    configured = model_path or os.environ.get(VAD_MODEL_ENV)
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        _require_marblenet_v2(path)
        return path
    if BENCHMARK_MARBLENET_PATH.is_file():
        _require_marblenet_v2(BENCHMARK_MARBLENET_PATH)
        return BENCHMARK_MARBLENET_PATH
    raise FileNotFoundError(
        "MarbleNet v2 checkpoint is not configured; set OMNI_CURATOR_VAD_MODEL or pass "
        "--vad-model (the old vad_multilingual_frame_marblenet model is not an equivalent fallback)"
    )


def _require_marblenet_v2(path: Path) -> None:
    actual = _sha256(path)
    if actual != MARBLENET_V2_SHA256:
        raise ValueError(
            f"MarbleNet checkpoint hash mismatch for {path}: expected "
            f"{MARBLENET_V2_SHA256}, got {actual}"
        )


def _automatic_device() -> str:
    try:
        torch = import_module("torch")
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pvcobra_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized in {"", "cpu"}:
        return "cpu"
    if normalized in {"cuda", "gpu"}:
        return "gpu:0"
    if normalized.startswith("cuda:"):
        return f"gpu:{normalized.removeprefix('cuda:')}"
    return device


def _resolve_silero_backend(*, backend: str, device: str) -> str:
    normalized = backend.strip().lower()
    is_cpu = device.strip().lower() in {"", "cpu"}
    if normalized == "auto":
        return "onnx" if is_cpu else "jit"
    if normalized not in {"onnx", "jit"}:
        raise ValueError("Silero backend must be one of: auto, onnx, jit")
    if normalized == "onnx" and not is_cpu:
        raise ValueError("Silero ONNX backend is CPU-only; use backend=jit for CUDA")
    return normalized


_VAD_GPU_GB_PER_WORKER = 3.0


def resolve_devices(gpu_procs: int, cpu_procs: int) -> tuple[int, int]:
    """Move unaffordable requested CUDA workers to CPU without opening a CUDA context."""
    if gpu_procs <= 0:
        return 0, cpu_procs
    free_gb = _gpu_free_gb()
    if free_gb is None:
        return 0, cpu_procs + gpu_procs
    affordable = int(free_gb // _VAD_GPU_GB_PER_WORKER)
    if affordable < gpu_procs:
        return max(0, affordable), cpu_procs + (gpu_procs - affordable)
    return gpu_procs, cpu_procs


def _gpu_free_gb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.stdout.strip().splitlines()
    try:
        return int(lines[0]) / 1024 if lines else None
    except ValueError:
        return None


def with_max_speech(policy: VadPolicy, max_speech_s: float) -> VadPolicy:
    """Return ``policy`` with only its model-specific hard cap changed."""
    return replace(policy, profile=replace(policy.profile, max_speech_s=max_speech_s))
