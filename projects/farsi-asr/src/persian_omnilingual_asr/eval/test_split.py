"""Score fine-tuned Persian omni CTC models on a held-out parquet split.

Entry point ``persian-omni-eval``. Runs one or more registered model cards over the
omni-parquet shards of a dataset, transcribes (CPU by default so it doesn't contend with
the GPU), normalizes both sides with the Persian normalizer, and reports corpus-level
metrics (WER/CER/MER + S/D/I/H) overall and per corpus via the shared
:func:`omni_finetune_core.metrics.compute_measures`.

Persian's training parquet currently carries only ``train``/``dev`` splits (no ``test``),
so the default split is ``dev``. Point ``--data-dir`` at any ``version=0`` root and pass
``--models label=card_name`` to compare checkpoints.

  persian-omni-eval                       # scribe-v4 dev, default model, CPU
  persian-omni-eval --limit 5             # smoke test
  persian-omni-eval --models v4=omni_ctc_300m_v2_scribe_v4_20260530_best
  persian-omni-eval --device cuda         # if the GPU is free

Audio is read straight from the parquet (audio_bytes = FLAC int8) and handed to the
pipeline as int8 arrays; no temp files. Clips longer than the omni pipeline's 40s cap are
excluded (and counted) since the pipeline raises on them.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omni_finetune_core.metrics import compute_measures

from persian_asr_dataset.text_normalization import maybe_normalize

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "src/finetune_omni/data/training/omnilingual/scribe-v4/version=0"
LANG = "fas_Arab"
SAMPLE_RATE = 16_000
MAX_AUDIO_SEC = 40.0  # omni ASRInferencePipeline hard cap (assert_max_length)
DEFAULT_MODELS = [("scribe-v4", "omni_ctc_300m_v2_scribe_v4_20260530_best")]


def normalize_text(text: str) -> str:
    """Persian normalizer wrapper. ``maybe_normalize`` returns None for skipped rows."""
    return maybe_normalize(text) or ""


def load_split(
    data_dir: Path, split: str, limit: int, max_dur: float
) -> tuple[list, list[str], list[str], int]:
    audio, refs, corpora = [], [], []
    excluded = 0
    for path in sorted(data_dir.glob(f"corpus=*/split={split}/language={LANG}/*.parquet")):
        corpus = next(p.split("=")[1] for p in path.parts if p.startswith("corpus="))
        t = pq.read_table(path, columns=["text", "audio_bytes", "audio_size"])
        for text, ab, size in zip(
            t.column("text").to_pylist(),
            t.column("audio_bytes").to_pylist(),
            t.column("audio_size").to_pylist(),
            strict=True,
        ):
            if size / SAMPLE_RATE > max_dur:
                excluded += 1
                continue
            audio.append(np.asarray(ab, dtype=np.int8))
            refs.append(text)
            corpora.append(corpus)
            if limit and len(audio) >= limit:
                return audio, refs, corpora, excluded
    return audio, refs, corpora, excluded


def measures(refs: list[str], hyps: list[str]) -> dict[str, float]:
    """Corpus-level metrics via the shared core scorer. WER/MER/WIL + S/D/I/H per alignment."""
    m = compute_measures(refs, hyps)
    return {
        "wer": m.wer,
        "cer": m.cer,
        "mer": m.mer,
        "wil": m.wil,
        "sub": float(m.substitutions),
        "del": float(m.deletions),
        "ins": float(m.insertions),
        "hits": float(m.hits),
    }


def parse_models(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_MODELS
    out = []
    for v in values:
        label, _, card = v.partition("=")
        out.append((label, card) if card else (label, label))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score Persian omni CTC models on a parquet split.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="version=0 parquet root")
    p.add_argument("--split", default="dev", help="parquet split partition (default: dev)")
    p.add_argument("--models", nargs="+", default=None, help="label=card_name (repeatable)")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-duration", type=float, default=MAX_AUDIO_SEC, help="drop clips longer")
    p.add_argument("--limit", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("FAIRSEQ2_CACHE_DIR", str(ROOT / ".fairseq2-cache/assets"))

    import torch

    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16

    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    models = parse_models(args.models)
    audio, refs, corpora, excluded = load_split(
        args.data_dir, args.split, args.limit, args.max_duration
    )
    print(
        f"{args.split} rows: {len(audio)} (excluded {excluded} > {args.max_duration:.0f}s) | "
        f"corpora: {sorted(set(corpora))} | device: {args.device}",
        flush=True,
    )
    refs_norm = [normalize_text(r) for r in refs]

    summary = {}
    for label, card in models:
        print(f"\n=== {label} ({card}) ===", flush=True)
        pipe = ASRInferencePipeline(card, device=args.device, dtype=dtype)
        hyps_raw = pipe.transcribe(audio, lang=[LANG] * len(audio), batch_size=args.batch_size)
        del pipe
        hyps = [normalize_text(h) for h in hyps_raw]
        m = measures(refs_norm, hyps)
        summary[label] = m
        print(
            f"{label}: WER {m['wer']:.2f}%  CER {m['cer']:.2f}%  MER {m['mer']:.2f}%  "
            f"WIL {m['wil']:.2f}%  |  sub {m['sub']:.0f} del {m['del']:.0f} "
            f"ins {m['ins']:.0f} hits {m['hits']:.0f}",
            flush=True,
        )
        for corp in sorted(set(corpora)):
            idx = [i for i, x in enumerate(corpora) if x == corp]
            cm = measures([refs_norm[i] for i in idx], [hyps[i] for i in idx])
            print(
                f"    {corp:<28} WER {cm['wer']:6.2f}%  CER {cm['cer']:6.2f}%  (n={len(idx)})",
                flush=True,
            )

    print(f"\n=== SUMMARY (Persian {args.split} split, corpus-level, jiwer) ===")
    print(f"{'model':<16}{'WER':>9}{'CER':>9}{'MER':>9}")
    for label, m in summary.items():
        print(f"{label:<16}{m['wer']:>8.2f}%{m['cer']:>8.2f}%{m['mer']:>8.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
