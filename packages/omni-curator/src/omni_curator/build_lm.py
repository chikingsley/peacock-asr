"""Build a KenLM word 4-gram from a dataset export's train labels.

Reads every corpus=*/split=train parquet under the version root, writes the text column
as corpus.txt, then runs lmplz + build_binary (packages/kenlm/build/bin). Labels are
already normalized at export time, so no further text processing happens here.

  uv run omni-build-lm projects/georgian-asr/data/datasets/v0/version=0 \
      projects/georgian-asr/experiments/lm_decoding
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

# packages/omni-curator/src/omni_curator/build_lm.py -> packages/kenlm/build/bin
KENLM_BIN = Path(__file__).resolve().parents[3] / "kenlm/build/bin"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--order", type=int, default=4)
    args = ap.parse_args()

    paths = sorted(args.version_root.glob("corpus=*/split=train/language=*/*.parquet"))
    if not paths:
        sys.exit(f"no train parquets under {args.version_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpus = args.out_dir / "corpus.txt"
    n = 0
    with corpus.open("w") as f:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(columns=["text"]):
                for text in batch.column("text").to_pylist():
                    line = " ".join(text.split())
                    if line:
                        f.write(line + "\n")
                        n += 1
    print(f"corpus.txt: {n} lines from {len(paths)} parquets")

    arpa = args.out_dir / f"lm{args.order}.arpa"
    binary = args.out_dir / f"lm{args.order}.bin"
    prune = ["0"] + ["1"] * (args.order - 1)
    with corpus.open() as fin, arpa.open("w") as fout:
        subprocess.run(
            [KENLM_BIN / "lmplz", "-o", str(args.order), "--prune", *prune, "-S", "40%"],
            stdin=fin,
            stdout=fout,
            check=True,
        )
    subprocess.run([KENLM_BIN / "build_binary", arpa, binary], check=True)
    print(f"built {binary} ({binary.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
