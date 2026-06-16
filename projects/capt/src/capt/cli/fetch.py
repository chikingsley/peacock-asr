"""Download the ZIPA ONNX recognizer model into `artifacts/` (replaces bootstrap_zipa.sh).

ZIPA inference itself is vendored in `capt.recognize`, so only the model weights + tokens are
fetched here. Run once after `uv sync`:

    capt-fetch-zipa
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "anyspeech/zipa-large-crctc-ns-800k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "artifacts" / "zipa-large-crctc-ns-800k",
        help="destination directory for the ONNX model + tokens.txt",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(REPO_ID, local_dir=str(args.out_dir))
    print(f"ZIPA model dir: {path}")


if __name__ == "__main__":
    main()
