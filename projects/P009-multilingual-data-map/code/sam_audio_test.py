#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0",
#     "torchaudio>=2.0",
#     "python-dotenv>=1.0",
#     "setuptools<70",
#     "sam-audio @ git+https://github.com/facebookresearch/sam-audio.git",
# ]
# ///
"""
SAM-Audio before/after test with judge scoring.

Run:
    ./code/sam_audio_test.py --input path/to/file.wav
    ./code/sam_audio_test.py --input_dir data/ru_open_stt --n 5

Outputs (in --out_dir, default sam_audio_out/):
    <stem>_original.wav   re-saved mono at model sample rate
    <stem>_target.wav     isolated speech

Note: sam-audio models are gated on HF. Accept access at:
    https://huggingface.co/facebook/sam-audio-small
    https://huggingface.co/facebook/sam-audio-judge
Then set HF_TOKEN in env or confirm ~/.cache/huggingface/token exists.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import os
import random
from pathlib import Path

import torch
import torchaudio

DEFAULT_MODEL_SAMPLE_RATE = 48_000


def _hf_cache() -> Path:
    from dotenv import load_dotenv
    repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(repo_root / ".env")
    cache = repo_root / ".cache" / "models" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    return cache


def _patch_sam_audio_compat() -> None:
    """huggingface_hub >=0.27 dropped proxies/resume_download from _from_pretrained calls,
    but sam-audio's BaseModel still declares them as required. Inject defaults."""
    from sam_audio.model.base import BaseModel
    orig = BaseModel.__dict__["_from_pretrained"].__func__

    @classmethod  # type: ignore[misc]
    def _fixed(cls, *, proxies=None, resume_download=False, **kwargs):
        return orig(cls, proxies=proxies, resume_download=resume_download, **kwargs)

    BaseModel._from_pretrained = _fixed  # type: ignore[method-assign]


def _patch_no_imagebind() -> None:
    """Skip loading ImageBind (4.5 GB visual ranker) — not needed for audio-only separation."""
    import sam_audio.ranking as _r
    _r.create_ranker = lambda _cfg: None  # type: ignore[assignment]


def load_separator(model_id: str, device: torch.device):
    from sam_audio import SAMAudio, SAMAudioProcessor
    _patch_sam_audio_compat()
    _patch_no_imagebind()
    print(f"loading  {model_id}")
    m = SAMAudio.from_pretrained(
        model_id,
        span_predictor=None,
        text_ranker=None,
        visual_ranker=None,
    ).to(device).eval()
    p = SAMAudioProcessor.from_pretrained(model_id)
    return m, p


def load_judge(device: torch.device):
    from sam_audio import SAMAudioJudgeModel, SAMAudioJudgeProcessor
    _patch_sam_audio_compat()
    model_id = "facebook/sam-audio-judge"
    print(f"loading  {model_id}")
    m = SAMAudioJudgeModel.from_pretrained(model_id).to(device).eval()
    p = SAMAudioJudgeProcessor.from_pretrained(model_id)
    return m, p


def to_model_rate_mono(
    waveform: torch.Tensor,
    sr: int,
    target_sr: int = DEFAULT_MODEL_SAMPLE_RATE,
) -> torch.Tensor:
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform


def separate_chunked(
    waveform: torch.Tensor,
    model,
    processor,
    device: torch.device,
    description: str,
    tmp_dir: Path,
    stem: str,
    sample_rate: int,
    chunk_s: float = 25.0,
    overlap_s: float = 2.0,
) -> torch.Tensor:
    chunk_n = int(chunk_s * sample_rate)
    overlap_n = int(overlap_s * sample_rate)
    step = chunk_n - overlap_n
    pieces: list[torch.Tensor] = []
    pos, idx = 0, 0

    while pos < waveform.shape[1]:
        chunk = waveform[:, pos : pos + chunk_n]
        tmp = tmp_dir / f"_tmp_{stem}_{idx}.wav"
        torchaudio.save(str(tmp), chunk, sample_rate)

        inp = processor(audios=[str(tmp)], descriptions=[description]).to(device)
        with torch.inference_mode():
            out = model.separate(inp)

        piece = out.target[0].cpu()
        if idx > 0:
            piece = piece[overlap_n:]
        pieces.append(piece)
        tmp.unlink()
        pos += step
        idx += 1

    return torch.cat(pieces).unsqueeze(0)


def run_judge(
    judge,
    judge_proc,
    device: torch.device,
    description: str,
    original: Path,
    target: Path,
) -> float | None:
    if judge is None:
        return None
    try:
        inp = judge_proc(
            text=[description],
            input_audio=[str(original)],
            separated_audio=[str(target)],
        ).to(device)
        with torch.inference_mode():
            s = judge(**inp)
        return float(s.overall[0])
    except Exception as exc:
        print(f"    judge error: {exc}")
        return None


def process_file(
    audio_path: Path,
    separator,
    sep_proc,
    judge,
    judge_proc,
    device: torch.device,
    description: str,
    out_dir: Path,
    sample_rate: int,
    chunk_s: float,
    overlap_s: float,
) -> dict:
    waveform, sr = torchaudio.load(str(audio_path))
    waveform = to_model_rate_mono(waveform, sr, sample_rate)
    dur = waveform.shape[1] / sample_rate
    print(f"\n  {audio_path.name}  ({dur:.1f}s)")

    target = separate_chunked(
        waveform, separator, sep_proc, device,
        description, out_dir, audio_path.stem, sample_rate, chunk_s, overlap_s,
    )

    original_out = out_dir / f"{audio_path.stem}_original.wav"
    target_out   = out_dir / f"{audio_path.stem}_target.wav"
    torchaudio.save(str(original_out), waveform.cpu(), sample_rate)
    torchaudio.save(str(target_out), target, sample_rate)
    print(f"    original → {original_out.name}")
    print(f"    target   → {target_out.name}")

    score = run_judge(judge, judge_proc, device, description, original_out, target_out)
    if score is not None:
        print(f"    judge    → {score:.3f}")

    return {"file": audio_path.name, "dur": dur, "score": score}


@lru_cache(maxsize=4)
def _get_runtime(model_id: str, with_judge: bool):
    _hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    separator, sep_proc = load_separator(model_id, device)
    judge, judge_proc = (None, None) if not with_judge else load_judge(device)
    sample_rate = getattr(sep_proc, "audio_sampling_rate", DEFAULT_MODEL_SAMPLE_RATE)
    return separator, sep_proc, judge, judge_proc, device, sample_rate


def run_single_file(
    audio_path: Path,
    description: str,
    out_dir: Path,
    model_id: str = "facebook/sam-audio-small",
    with_judge: bool = False,
    chunk_s: float = 25.0,
    overlap_s: float = 2.0,
) -> dict:
    (
        separator,
        sep_proc,
        judge,
        judge_proc,
        device,
        sample_rate,
    ) = _get_runtime(model_id, with_judge)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = process_file(
        audio_path=audio_path,
        separator=separator,
        sep_proc=sep_proc,
        judge=judge,
        judge_proc=judge_proc,
        device=device,
        description=description,
        out_dir=out_dir,
        sample_rate=sample_rate,
        chunk_s=chunk_s,
        overlap_s=overlap_s,
    )
    return {
        **result,
        "original_path": out_dir / f"{audio_path.stem}_original.wav",
        "target_path": out_dir / f"{audio_path.stem}_target.wav",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="SAM-Audio before/after + judge test")
    ap.add_argument("--input",       type=Path, help="Single audio file")
    ap.add_argument("--input_dir",   type=Path, help="Directory to sample from")
    ap.add_argument("--n",           type=int, default=5)
    ap.add_argument("--model",       default="facebook/sam-audio-small")
    ap.add_argument("--description", default="a person speaking")
    ap.add_argument("--out_dir",     type=Path, default=Path("sam_audio_out"))
    ap.add_argument("--no_judge",    action="store_true")
    ap.add_argument("--chunk_s",     type=float, default=25.0)
    ap.add_argument("--overlap_s",   type=float, default=2.0)
    args = ap.parse_args()

    if args.input is None and args.input_dir is None:
        ap.error("pass --input <file> or --input_dir <dir>")

    _hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"vram:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    separator, sep_proc = load_separator(args.model, device)
    judge, judge_proc   = (None, None) if args.no_judge else load_judge(device)
    sample_rate = getattr(sep_proc, "audio_sampling_rate", DEFAULT_MODEL_SAMPLE_RATE)

    if args.input:
        files = [args.input]
    else:
        candidates = (
            list(args.input_dir.rglob("*.wav"))
            + list(args.input_dir.rglob("*.opus"))
            + list(args.input_dir.rglob("*.flac"))
        )
        files = random.sample(candidates, min(args.n, len(candidates)))

    results = []
    for f in files:
        try:
            results.append(process_file(
                f, separator, sep_proc, judge, judge_proc,
                device, args.description, args.out_dir,
                sample_rate,
                args.chunk_s, args.overlap_s,
            ))
        except Exception as exc:
            print(f"  ERROR {f.name}: {exc}")

    print("\n--- results ---")
    for r in results:
        score_str = f"  score={r['score']:.3f}" if r["score"] is not None else ""
        print(f"  {r['file']}  {r['dur']:.1f}s{score_str}")

    scored = [r for r in results if r["score"] is not None]
    if scored:
        avg = sum(r["score"] for r in scored) / len(scored)
        print(f"\n  mean judge: {avg:.3f}  (n={len(scored)})")

    print(f"\noutputs → {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
