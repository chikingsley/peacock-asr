"""Gate 1 — omni-parquet -> NeMo manifest (Tajik).

NeMo training needs JSONL manifests (audio_filepath/text/duration) + audio files. The omni
rows already store the audio as encoded FLAC bytes (verified: `fLaC` magic, decodes to
audio_size samples @16k), so we write the bytes straight to .flac (lossless, no re-encode)
and set duration = audio_size/16000. Text is normalized the same way our WER eval normalizes.

Note: the v3 export only has split=test partitions (FLEURS + youtube). For gate 0a (reproduce
the TDT stall) that's fine — we slice FLEURS test into pseudo train/dev. Real training data
(the 180k-row training export) is a separate source to wire up later.

  uv run python experiments/tdt/gate1_build_manifest.py --corpus-prefix fleurs --train 400 --dev 119
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET_ROOT = ROOT / "data/datasets/v3/version=0"
OUT = HERE / "data"
LANG = "tgk_Cyrl"
SR = 16000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-prefix", default="fleurs", help="corpus filter (fleurs / youtube-)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--dev", type=int, default=119)
    ap.add_argument("--max-dur", type=float, default=40.0)
    args = ap.parse_args()

    from omni_curator.process import normalize
    from omni_finetune_core.parquet import iter_split

    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(iter_split(DATASET_ROOT, args.split, max_duration=args.max_dur)):
        if not r.corpus.startswith(args.corpus_prefix):
            continue
        dur = r.audio_size / SR
        text = normalize(r.text, LANG).strip()
        if not text:
            continue
        fp = audio_dir / f"{args.corpus_prefix}_{i:05d}.flac"
        fp.write_bytes(r.audio.tobytes())          # already-encoded FLAC; lossless passthrough
        rows.append({"audio_filepath": str(fp), "text": text, "duration": round(dur, 3)})

    n_tr = min(args.train, len(rows))
    train, dev = rows[:n_tr], rows[n_tr : n_tr + args.dev]

    def dump(name: str, recs: list[dict]) -> Path:
        p = OUT / f"{name}.jsonl"
        with p.open("w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return p

    tr_p, dv_p = dump("train", train), dump("dev", dev)
    # tokenizer training corpus = train transcripts (gate 2)
    corpus_p = OUT / "train_text.txt"
    corpus_p.write_text("\n".join(r["text"] for r in train) + "\n")

    tot_h = sum(r["duration"] for r in rows) / 3600
    print(f"corpus={args.corpus_prefix!r}: {len(rows)} rows ({tot_h:.2f} h)  "
          f"-> train {len(train)} ({tr_p.name}), dev {len(dev)} ({dv_p.name})", flush=True)
    print(f"audio -> {audio_dir}  |  tokenizer corpus -> {corpus_p.name}", flush=True)
    print(f"sample: {json.dumps(train[0], ensure_ascii=False)[:110]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
