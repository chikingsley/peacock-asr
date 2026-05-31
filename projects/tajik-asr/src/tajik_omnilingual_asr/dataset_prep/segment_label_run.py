"""Batched segment -> ensemble -> compile-down runner (VAD or overlapping-chunk).

Two interchangeable segmenters (NeMo VAD; fixed overlapping chunks) feed one per-segment
pipeline: cut 16 kHz audio -> Scribe ensemble (en + tgk) -> compile_down -> Cyrillic label.
Per-segment work runs in a thread pool because the Scribe and SuperWhisper calls are
I/O-bound, so an hour of audio runs as a pool of workers instead of one segment at a time.

  uv run python -m tajik_omnilingual_asr.dataset_prep.segment_label_run <audio> \
      --segmenter vad   --out-dir <dir>/vad
  uv run python -m tajik_omnilingual_asr.dataset_prep.segment_label_run <audio> \
      --segmenter chunks --out-dir <dir>/chunks --chunk 27 --overlap 5
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tajik_omnilingual_asr.dataset_prep.compile_down import compile_down
from tajik_omnilingual_asr.dataset_prep.youtube.process import cut_audio
from tajik_omnilingual_asr.dataset_prep.youtube.segment import NEMO_VAD_MODEL_PATH, boolean_windows

if TYPE_CHECKING:
    from superwhisper_api.text.client import SuperwhisperClient

DEFAULT_LANGS = ("en", "tgk")


@dataclass
class Segment:
    index: int
    start: float
    end: float
    audio_path: str
    variants: list[str]
    label: str


def segment_vad(
    audio: Path,
    *,
    threshold: float = 0.5,
    min_dur: float = 1.0,
    merge_gap: float = 1.5,
    max_dur: float = 30.0,
) -> list[tuple[float, float]]:
    """Speech windows from the NeMo frame-VAD (speech-bounded, language-blind)."""
    import numpy as np
    import torch
    from nemo.collections.asr.parts.utils.vad_utils import EncDecFrameClassificationModel

    model = EncDecFrameClassificationModel.restore_from(
        str(NEMO_VAD_MODEL_PATH), map_location="cpu"
    )
    model.eval()
    with torch.no_grad():
        logits = model.transcribe([str(audio)], batch_size=1, logprobs=True)[0]
    speech = torch.softmax(torch.from_numpy(np.asarray(logits)), dim=-1).numpy()[:, 1]
    windows = boolean_windows(
        [float(p) >= threshold for p in speech],
        frame_seconds=0.02,
        min_duration_seconds=min_dur,
        merge_gap_seconds=merge_gap,
        hard_max_seconds=max_dur,
    )
    return [(w.start, w.end) for w in windows]


def segment_chunks(
    duration: float, *, chunk: float = 27.0, overlap: float = 5.0
) -> list[tuple[float, float]]:
    """Fixed overlapping windows tiling [0, duration] — blind to speech (VAD-free fallback)."""
    step = max(chunk - overlap, 1.0)
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        spans.append((start, min(start + chunk, duration)))
        if start + chunk >= duration:
            break
        start += step
    return spans


def _audio_duration(audio: Path) -> float:
    import soundfile as sf

    info = sf.info(audio)
    return float(info.frames) / float(info.samplerate)


def _make_scribe(key: str, langs: tuple[str, ...]) -> dict[str, Any]:
    from superwhisper_api.audio.models import audio_model
    from superwhisper_api.audio.transcribe import create_process_fn

    spec = audio_model("scribe-v2")
    return {lang: create_process_fn(spec, key, language=lang, diarize=False) for lang in langs}


def _process_segment(
    item: tuple[int, tuple[float, float]],
    *,
    source: Path,
    out_dir: Path,
    scribe_fns: dict[str, Any],
    runs: int,
    client: SuperwhisperClient,
) -> Segment:
    index, (start, end) = item
    clip = out_dir / "cuts" / f"seg_{index:04d}.flac"
    cut_audio(source, clip, start, end)
    variants: list[str] = []
    for fn in scribe_fns.values():
        for _ in range(runs):
            transcript = str(fn(clip).as_dict().get("transcript") or "").strip()
            if transcript:
                variants.append(transcript)
    label = compile_down(variants, client=client) if variants else ""
    return Segment(index, round(start, 2), round(end, 2), str(clip), variants, label)


def run(
    audio: Path,
    *,
    segmenter: str,
    out_dir: Path,
    langs: tuple[str, ...] = DEFAULT_LANGS,
    runs: int = 1,
    workers: int = 8,
    chunk: float = 27.0,
    overlap: float = 5.0,
) -> list[Segment]:
    """Segment, transcribe (ensemble, parallel), compile down to labels. Writes a JSON."""
    from superwhisper_api.auth import ensure_elevenlabs_key
    from superwhisper_api.text.client import SuperwhisperClient

    (out_dir / "cuts").mkdir(parents=True, exist_ok=True)
    if segmenter == "vad":
        spans = segment_vad(audio)
    else:
        spans = segment_chunks(_audio_duration(audio), chunk=chunk, overlap=overlap)

    scribe_fns = _make_scribe(ensure_elevenlabs_key(), langs)
    client = SuperwhisperClient()
    worker = partial(
        _process_segment,
        source=audio,
        out_dir=out_dir,
        scribe_fns=scribe_fns,
        runs=runs,
        client=client,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(worker, enumerate(spans)))
    results.sort(key=lambda s: s.index)
    (out_dir / f"{segmenter}_segments.json").write_text(
        json.dumps([asdict(s) for s in results], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batched segment -> ensemble -> compile-down runner.")
    p.add_argument("audio", type=Path)
    p.add_argument("--segmenter", choices=("vad", "chunks"), default="vad")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--langs", default="en,tgk")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=float, default=27.0)
    p.add_argument("--overlap", type=float, default=5.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run(
        args.audio,
        segmenter=args.segmenter,
        out_dir=args.out_dir,
        langs=tuple(args.langs.split(",")),
        runs=args.runs,
        workers=args.workers,
        chunk=args.chunk,
        overlap=args.overlap,
    )
    speech = sum(s.end - s.start for s in results)
    print(f"{args.segmenter}: {len(results)} segments, {speech:.0f}s covered, out={args.out_dir}")
    for s in results[:6]:
        print(f"  [{s.start:6.1f}-{s.end:6.1f}] {s.label[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
