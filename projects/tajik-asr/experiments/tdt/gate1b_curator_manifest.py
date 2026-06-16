"""Big-run data: curator.sqlite -> NeMo manifests (real 1446h Tajik train).

The curator store has absolute `audio_path` to existing FLACs, so this is a SQL->JSONL query
(no audio copy). Gated train = scribe_wer<=0.35 (same bar omni training used). Val = dev split.

  uv run python experiments/tdt/gate1b_curator_manifest.py --max-wer 0.35 --max-dur 20
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE.parents[1] / "data/curator.sqlite"
OUT = HERE / "data"
LANG = "tgk_Cyrl"


def dump(name: str, rows: list[sqlite3.Row]) -> tuple[Path, float]:
    p = OUT / f"{name}.jsonl"
    n = 0
    hrs = 0.0
    with p.open("w") as f:
        for r in rows:
            txt = (r["text"] or "").strip()
            if not txt or not Path(r["audio_path"]).exists():
                continue
            f.write(json.dumps({"audio_filepath": r["audio_path"], "text": txt,
                                "duration": round(r["duration"], 3)}, ensure_ascii=False) + "\n")
            n += 1
            hrs += r["duration"]
    return p, hrs / 3600


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-wer", type=float, default=0.35)
    ap.add_argument("--min-dur", type=float, default=0.5)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all gated train")
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    base = (f"from samples where language='{LANG}' and duration between {args.min_dur} "
            f"and {args.max_dur}")
    tr_q = (f"select text,audio_path,duration {base} and split='train' "
            f"and scribe_wer is not null and scribe_wer<={args.max_wer} order by id")
    if args.limit:
        tr_q += f" limit {args.limit}"
    train = con.execute(tr_q).fetchall()
    dev = con.execute(f"select text,audio_path,duration {base} and split='dev' order by id").fetchall()

    tr_p, tr_h = dump("train_big", train)
    dv_p, dv_h = dump("dev_big", dev)
    # tokenizer corpus = all train transcripts
    (OUT / "train_big_text.txt").write_text(
        "\n".join((r["text"] or "").strip() for r in train) + "\n")
    print(f"train_big: {tr_p} ({tr_h:.1f}h)   dev_big: {dv_p} ({dv_h:.1f}h)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
