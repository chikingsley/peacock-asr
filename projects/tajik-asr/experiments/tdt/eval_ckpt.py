"""fp32 WER eval of a big_110m checkpoint on the held-out set (the real scorecard).

train_loss is noisy + not an accuracy number, and bf16 TDT decode is unreliable — so load the
checkpoint in fp32, transcribe held-out audio, and compute WER vs the reference (omni-normalized).

  uv run --project /home/simon/github/peacock-asr/projects/farsi-asr \
    python projects/tajik-asr/experiments/tdt/eval_ckpt.py --split dev --n 80 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from nemo.collections.asr.models import ASRModel

HERE = Path(__file__).resolve().parent
PEACOCK = HERE.parents[3]
HYBRID = PEACOCK / "base_models/parakeet/parakeet-tdt_ctc-110m-base-hybrid.nemo"
TOK = HERE.parents[1] / "data/parakeet/tokenizers/tokenizer_spe_bpe_v1024"
CKPT_DIR = HERE / "runs/big_110m/checkpoints"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "last.ckpt"))
    ap.add_argument("--base", default=str(HYBRID), help="base .nemo for the --ckpt path (110m or v3 0.6b)")
    ap.add_argument("--tokenizer-dir", default=str(TOK), help="tokenizer dir for change_vocabulary")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--manifest", default="", help="explicit manifest path (overrides --split)")
    ap.add_argument("--nemo", default="", help="restore a finished .nemo directly (skips base+ckpt)")
    ap.add_argument("--n", type=int, default=80, help="0 = all")
    ap.add_argument("--device", default="cpu", help="cpu (safe vs training) | cuda")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    manifest = (Path(args.manifest) if args.manifest
                else HERE / ("data/dev_big.jsonl" if args.split == "dev" else "data/test_big.jsonl"))
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
    if args.n:
        rows = rows[: args.n]
    paths = [r["audio_filepath"] for r in rows]
    refs = [r["text"] for r in rows]

    if args.nemo:
        model = ASRModel.restore_from(args.nemo, map_location=args.device)
        print(f"restored {Path(args.nemo).name}", flush=True)
    else:
        model = ASRModel.restore_from(args.base, map_location=args.device)
        model.change_vocabulary(new_tokenizer_dir=args.tokenizer_dir, new_tokenizer_type="bpe")
        state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        sd = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {Path(args.ckpt).name}: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    model = model.to(args.device).eval()

    print(f"transcribing {len(paths)} clips from {Path(manifest).name} (fp32, {args.device})...",
          flush=True)
    import time
    if args.device == "cuda" and len(paths) > 8:
        model.transcribe(paths[:8], batch_size=args.bs)  # warmup (CUDA init/compile)
    t0 = time.time()
    hyps = model.transcribe(paths, batch_size=args.bs)
    elapsed = time.time() - t0
    hyps = [h.text if hasattr(h, "text") else h for h in hyps]
    audio_sec = sum(float(r.get("duration", 0.0)) for r in rows)
    rtfx = audio_sec / elapsed if elapsed else float("nan")
    print(f"  RTFx={rtfx:.0f}x ({audio_sec:.0f}s audio / {elapsed:.1f}s, bs{args.bs}, {args.device})",
          flush=True)

    from omni_curator.process import normalize
    from omni_finetune_core.metrics import compute_measures
    wer = compute_measures([normalize(r, "tgk_Cyrl") for r in refs],
                           [normalize(h, "tgk_Cyrl") for h in hyps]).wer
    print("\n=== samples ===", flush=True)
    for h, r in list(zip(hyps, refs, strict=False))[:4]:
        print(f"  REF: {r[:70]}\n  HYP: {h[:70]}\n", flush=True)
    print(f"=== {Path(manifest).stem} WER ({len(paths)} clips, fp32): {wer:.2f}% ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
