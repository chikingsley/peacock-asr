"""Bench raw omniASR-LLM variants on a project's test split (the LLM-ceiling probe).

Runs Meta's LLM-decoder omnilingual models (no fine-tuning) on the same rows our CTC
evals use, so the numbers are directly comparable to the per-project EXPERIMENTS tables.
Answers: how much does an LLM decoder buy on this language before we invest in any
LLM-in-the-loop training?

  uv run --project projects/tajik-asr omni-bench-llm \
      projects/tajik-asr/data/datasets/v3/version=0 tgk_Cyrl \
      --corpus-prefix fleurs --corpus-prefix youtube-
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version_root", type=Path)
    ap.add_argument("language")
    ap.add_argument("--model", action="append", default=None,
                    help="model cards (default: omniASR_LLM_300M_v2, omniASR_LLM_1B_v2)")
    ap.add_argument("--corpus-prefix", action="append", default=None,
                    help="score these test partitions separately (default: all rows pooled)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    models = args.model or ["omniASR_LLM_300M_v2", "omniASR_LLM_1B_v2"]
    prefixes = args.corpus_prefix or [""]

    import torch
    from omni_curator.process import normalize
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    from omni_finetune_core.metrics import compute_measures
    from omni_finetune_core.project import _load_test

    audio, refs, corpora, _ = _load_test(args.version_root, args.language, args.limit, 40.0)
    print(f"rows: {len(audio)}", flush=True)

    for card in models:
        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
        pipe = ASRInferencePipeline(card, device=args.device, dtype=dtype)
        t0 = time.monotonic()
        hyps = pipe.transcribe(
            audio, lang=[args.language] * len(audio), batch_size=args.batch_size
        )
        elapsed = time.monotonic() - t0
        for prefix in prefixes:
            keep = [i for i, c in enumerate(corpora) if c.startswith(prefix)]
            m = compute_measures(
                [normalize(refs[i], args.language) for i in keep],
                [normalize(hyps[i], args.language) for i in keep],
            )
            label = prefix or "all"
            print(f"{card:<22} {label:<10} ({len(keep)} rows)  "
                  f"WER {m.wer:6.2f}%  CER {m.cer:6.2f}%", flush=True)
        print(f"{card:<22} transcribe time {elapsed:.0f}s", flush=True)
        del pipe
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
