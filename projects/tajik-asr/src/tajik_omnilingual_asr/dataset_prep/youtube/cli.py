"""YouTube → Tajik ASR dataset pipeline (one CLI, sqlite-backed).

Stages, each reading/writing the artifact sqlite DB:
  download         video audio + captions          (ingest)
  transcribe-scribe  Scribe on full videos -> words (transcribe)
  segment          NeMo VAD + Scribe-word align     (segment)
  transcribe-omni  our omni model, per segment      (transcribe)
  export           Scribe<->omni agreement -> omni-parquet (export)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tajik_omnilingual_asr.dataset_prep.youtube import export, ingest, segment, transcribe
from tajik_omnilingual_asr.dataset_prep.youtube.paths import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CAPTION_LANGUAGES,
    DEFAULT_CHANNEL_URL,
    DEFAULT_LANGUAGE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YouTube -> Tajik ASR dataset pipeline.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--db", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-channel", help="list a channel's videos")
    p.add_argument("--channel-url", default=DEFAULT_CHANNEL_URL)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=ingest.cmd_list_channel)

    p = sub.add_parser("download", help="download videos (URLs or --channel-url) into the DB")
    p.add_argument("url", nargs="*", help="video URLs or 11-char IDs")
    p.add_argument("--channel-url")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-duration-seconds", type=float)
    p.add_argument("--exclude-title-regex")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-captions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--caption-languages", default=DEFAULT_CAPTION_LANGUAGES)
    p.add_argument("--fail-jsonl", type=Path)
    p.add_argument("--fail-on-error", action="store_true")
    p.set_defaults(func=ingest.cmd_download)

    p = sub.add_parser("transcribe-scribe", help="Scribe on full videos (provides word timing)")
    p.add_argument("video_id", nargs="*", help="video IDs (default: all)")
    p.add_argument("--model", default="scribe-v2")
    p.add_argument("--language", default=DEFAULT_LANGUAGE)
    p.add_argument("--key")
    p.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=transcribe.cmd_transcribe_scribe)

    p = sub.add_parser("segment", help="NeMo VAD + Scribe-word alignment -> segments")
    p.add_argument("video_id", nargs="*", help="video IDs (default: all)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--frame-seconds", type=float, default=0.02)
    p.add_argument("--speech-class-index", type=int, default=1)
    p.add_argument("--merge-gap-seconds", type=float, default=0.35)
    p.add_argument("--min-duration-seconds", type=float, default=1.0)
    p.add_argument("--max-duration-seconds", type=float, default=40.0)
    p.set_defaults(func=segment.cmd_segment)

    p = sub.add_parser("transcribe-omni", help="our omni model on each segment clip")
    p.add_argument("video_id", nargs="*", help="video IDs (default: all)")
    p.add_argument("--model-card", default="omni_ctc_300m_v2_tajik_step_1800")
    p.add_argument("--device", default="cpu", help="cpu (default, float32) or cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=transcribe.cmd_transcribe_omni)

    p = sub.add_parser("export", help="Scribe<->omni agreement gate -> omni-parquet")
    p.add_argument("--output-root", type=Path, required=True, help="omni_parquet/version=0 dir")
    p.add_argument("--split", default="train")
    p.add_argument("--max-agreement-wer", type=float, default=0.3)
    p.add_argument("--label-source", choices=("scribe", "omni"), default="scribe")
    p.add_argument("--rows-per-file", type=int, default=1000)
    p.set_defaults(func=export.cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
