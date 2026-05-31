"""Thin CLI: run one curation path over an audio file and write the transcript.

  omni-curator <audio> --path vad --out-dir D --language Tajik --script "Cyrillic" --langs auto,tgk
  omni-curator <audio> --path chunks --out-dir D --language French --script "French" --langs auto,fr
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omni_curator.create.pipeline import chunks_path, vad_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a curation path over an audio file.")
    p.add_argument("audio", type=Path)
    p.add_argument("--path", choices=("vad", "chunks"), default="vad")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--language", required=True)
    p.add_argument("--script", required=True)
    p.add_argument("--langs", default="auto", help="comma list; 'auto' = code-switching detect")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=float, default=40.0)
    p.add_argument("--overlap", type=float, default=10.0)
    p.add_argument("--no-polish", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    langs = tuple(args.langs.split(","))
    if args.path == "vad":
        result = vad_path(
            args.audio,
            out_dir=args.out_dir,
            language=args.language,
            script=args.script,
            langs=langs,
            runs=args.runs,
            workers=args.workers,
            do_polish=not args.no_polish,
        )
    else:
        result = chunks_path(
            args.audio,
            out_dir=args.out_dir,
            language=args.language,
            script=args.script,
            langs=langs,
            runs=args.runs,
            workers=args.workers,
            do_polish=not args.no_polish,
            chunk=args.chunk,
            overlap=args.overlap,
        )
    result.write_json(args.out_dir / f"{args.path}_transcript.json")
    speech = sum(c.end - c.start for c in result.clips)
    print(f"{args.path}: {len(result.clips)} clips, {speech:.0f}s covered, out={args.out_dir}")
    print(result.text[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
