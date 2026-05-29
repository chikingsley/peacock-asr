from __future__ import annotations

import argparse
import json
from pathlib import Path

from p016_compare.pipeline import PronunciationComparePipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="p016-compare")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("audio", type=Path)
    analyze.add_argument("--language", choices=["ru", "en_us", "en_gb"], default="ru")
    analyze.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.command == "analyze":
        result = PronunciationComparePipeline().analyze(args.audio, args.language)
        payload = result.as_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"ASR: {payload['asr']['text']}")
            for lane in payload["lanes"]:
                print(f"{lane['name']}: {lane['sentence']}")
