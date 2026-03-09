#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.6,<2.9",
#     "torchaudio>=2.6,<2.9",
#     "fairseq2[arrow]>=0.5.2,<=0.6.0",
#     "pyarrow>=20.0.0",
#     "numpy>=1.23,<2",
#     "pandas>=2.2",
#     "polars>=1.29.0",
#     "soundfile>=0.13.1",
# ]
# ///
"""Probe omniASR_CTC_300M_v2 through the upstream Omnilingual inference pipeline.

This is an integration entrypoint, not a phoneme-scoring backend yet.
It confirms the local third-party checkout and fairseq2 inference stack work.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any


def _inject_omnilingual_src(repo_root: Path) -> None:
    omni_root = (
        repo_root
        / "projects"
        / "P004-training-from-scratch"
        / "third_party"
        / "omnilingual-asr"
    )
    omni_src = omni_root / "src"
    for candidate in (omni_root, omni_src):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def _preload_tbb() -> None:
    try:
        tbb_dist = distribution("tbb")
    except PackageNotFoundError as exc:
        raise SystemExit(
            "Missing `tbb` runtime. Launch via the wrapper or run with "
            "`uv run --python 3.12 --with tbb>=2021.8 --with-editable "
            "<omnilingual-asr-path> ...`."
        ) from exc

    lib_path = tbb_dist.locate_file("../../libtbb.so.12")
    ctypes.CDLL(str(lib_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        nargs="*",
        default=[],
        help="Audio file paths to transcribe. Required unless --check-only is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size (default: 1).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. cuda or cpu (default: auto).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load the pipeline and exit without transcribing audio.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.check_only and not args.audio:
        msg = "Pass --audio <files...> or use --check-only."
        raise SystemExit(msg)

    repo_root = Path(__file__).resolve().parents[3]
    _inject_omnilingual_src(repo_root)
    _preload_tbb()

    from omnilingual_asr.models.inference.pipeline import (  # noqa: PLC0415
        ASRInferencePipeline,
    )

    pipeline = ASRInferencePipeline(
        model_card="omniASR_CTC_300M_v2",
        device=args.device,
    )
    result: dict[str, Any] = {
        "model_card": "omniASR_CTC_300M_v2",
        "device": str(pipeline.device),
        "check_only": args.check_only,
    }

    if not args.check_only:
        transcriptions = pipeline.transcribe(args.audio, batch_size=args.batch_size)
        result["audio"] = args.audio
        result["transcriptions"] = transcriptions

    payload = json.dumps(result, indent=2, ensure_ascii=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    sys.stdout.write(f"{payload}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
