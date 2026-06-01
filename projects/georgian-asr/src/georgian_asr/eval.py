"""Score fine-tuned Georgian omni CTC models on the held-out test split.

Entry point ``georgian-eval``. Runs one or more registered model cards over the omni-parquet
test partitions (``data/datasets/v0/version=0/corpus=*/split=test/language=kat_Geor``), transcribes
on CPU by default so it doesn't contend with the GPU, normalizes both sides with the shared
Georgian normalizer (``omni_curator.process.normalize`` for ``kat_Geor``), and reports
corpus-level metrics (jiwer ``process_words`` / ``process_characters``) overall and per corpus.

  georgian-eval --models ft=omni_ctc_300m_v2_georgian_step_2000   # score a trained card
  georgian-eval --models ft=<card> --limit 5                       # smoke test
  georgian-eval --models ft=<card> --device cuda                   # if the GPU is free

A trained checkpoint must be registered as a ModelCard in ``assets.py`` first (point the card's
checkpoint at runs/.../checkpoints/step_N/model/pp_00/tp_00/sdp_00.pt). Audio is read straight from
the parquet (audio_bytes = FLAC int8); clips longer than the omni pipeline's 40 s cap are excluded.
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
from omni_curator.process.normalize import normalize
from omni_finetune_core.metrics import compute_measures

from georgian_asr import LANGUAGE, ROOT

if TYPE_CHECKING:
    from pathlib import Path

PARQUET_ROOT = ROOT / "data" / "datasets" / "v0" / "version=0"
SAMPLE_RATE = 16_000
MAX_AUDIO_SEC = 40.0  # omni ASRInferencePipeline hard cap (assert_max_length)


def test_parquets() -> list[Path]:
    return sorted(PARQUET_ROOT.glob(f"corpus=*/split=test/language={LANGUAGE}/*.parquet"))


def load_test(limit: int, max_dur: float) -> tuple[list, list[str], list[str], int]:
    audio, refs, corpora = [], [], []
    excluded = 0
    for parquet_path in test_parquets():
        path = str(parquet_path)
        corpus = next(p.split("=")[1] for p in path.split("/") if p.startswith("corpus="))
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
        raise SystemExit("--models is required: label=card_name (a trained card from assets.py)")
    out = []
    for v in values:
        label, _, card = v.partition("=")
        out.append((label, card) if card else (label, label))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score Georgian omni CTC models on the test split.")
    p.add_argument("--models", nargs="+", default=None, help="label=card_name (trained card)")
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
    audio, refs, corpora, excluded = load_test(args.limit, args.max_duration)
    print(
        f"test rows: {len(audio)} (excluded {excluded} > {args.max_duration:.0f}s) | "
        f"corpora: {sorted(set(corpora))} | device: {args.device}",
        flush=True,
    )
    refs_norm = [normalize(r, LANGUAGE) for r in refs]

    summary = {}
    for label, card in models:
        print(f"\n=== {label} ({card}) ===", flush=True)
        pipe = ASRInferencePipeline(card, device=args.device, dtype=dtype)
        hyps_raw = pipe.transcribe(audio, lang=[LANGUAGE] * len(audio), batch_size=args.batch_size)
        del pipe
        hyps = [normalize(h, LANGUAGE) for h in hyps_raw]
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

    print("\n=== SUMMARY (Georgian test split, corpus-level, jiwer) ===")
    print(f"{'model':<16}{'WER':>9}{'CER':>9}{'MER':>9}")
    for label, m in summary.items():
        print(f"{label:<16}{m['wer']:>8.2f}%{m['cer']:>8.2f}%{m['mer']:>8.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
