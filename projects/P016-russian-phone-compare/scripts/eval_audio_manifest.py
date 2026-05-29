#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from p016_compare.pipeline import (
    DEFAULT_LANE_CONFIGS,
    DIAGNOSTIC_LANE_CONFIGS,
    PronunciationComparePipeline,
)

SUMMARY_FIELDS = [
    "id",
    "source",
    "language",
    "category",
    "reference_text",
    "asr_text",
    "normalized_asr_text",
    "lane",
    "backend",
    "text_normalization_backend",
    "PER",
    "PFER",
    "feature_distance",
    "errors",
    "matches",
    "substitutions",
    "deletions",
    "insertions",
    "reference_count",
    "audio_seconds",
    "asr_seconds",
    "target_g2p_seconds",
    "recognizer_seconds",
    "recognizer_cached",
    "score_seconds",
    "lane_total_seconds",
    "pipeline_total_seconds",
    "pipeline_rtf",
    "audio",
]

WORD_FIELDS = [
    "id",
    "source",
    "language",
    "category",
    "reference_text",
    "asr_text",
    "normalized_asr_text",
    "lane",
    "word",
    "target_phones",
    "recognized_phones",
    "PER",
    "PFER",
    "feature_distance",
    "substitutions",
    "deletions",
    "insertions",
    "substitutions_detail",
    "deletions_detail",
    "insertions_detail",
    "audio",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run P016 pronunciation scoring over a JSONL audio manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/eval"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip the Russian comparison lanes for non-default target backends.",
    )
    args = parser.parse_args()

    rows = list(_read_jsonl(args.manifest))
    if args.limit is not None:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lane_configs = (
        DEFAULT_LANE_CONFIGS
        if args.no_diagnostics
        else DEFAULT_LANE_CONFIGS + DIAGNOSTIC_LANE_CONFIGS
    )
    pipeline = PronunciationComparePipeline(lane_configs=lane_configs)
    summary_path = args.out_dir / "summary.csv"
    words_path = args.out_dir / "words.csv"
    jsonl_path = args.out_dir / "results.jsonl"

    with (
        summary_path.open("w", newline="", encoding="utf-8") as summary_file,
        words_path.open("w", newline="", encoding="utf-8") as words_file,
        jsonl_path.open("w", encoding="utf-8") as jsonl_file,
    ):
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        word_writer = csv.DictWriter(words_file, fieldnames=WORD_FIELDS)
        summary_writer.writeheader()
        word_writer.writeheader()

        for index, row in enumerate(rows, start=1):
            audio = Path(row["audio"]).expanduser()
            if not audio.exists():
                raise FileNotFoundError(f"Missing audio for {row['id']}: {audio}")
            print(f"[{index}/{len(rows)}] scoring {row['id']} {audio}")
            result = pipeline.analyze(audio, str(row["language"]))
            payload = result.as_dict()
            jsonl_file.write(
                json.dumps({"sample": row, "result": payload}, ensure_ascii=False) + "\n"
            )

            common = {
                "id": row["id"],
                "source": row.get("source", ""),
                "language": row["language"],
                "category": row.get("category", ""),
                "reference_text": row.get("text", ""),
                "asr_text": payload["asr"]["text"],
                "audio": str(audio),
            }
            pipeline_timing = payload.get("timing", {})
            for lane in payload["lanes"]:
                lane_timing = lane.get("timing", {})
                summary_writer.writerow(
                    {
                        **common,
                        "lane": lane["name"],
                        "backend": lane["target"]["backend"],
                        "normalized_asr_text": lane["target"].get("normalized_text", ""),
                        "text_normalization_backend": lane["target"].get(
                            "text_normalization_backend", ""
                        ),
                        "audio_seconds": pipeline_timing.get("audio_seconds", ""),
                        "asr_seconds": pipeline_timing.get("asr_seconds", ""),
                        "target_g2p_seconds": lane_timing.get("target_g2p_seconds", ""),
                        "recognizer_seconds": lane_timing.get("recognizer_seconds", ""),
                        "recognizer_cached": lane_timing.get("recognizer_cached", ""),
                        "score_seconds": lane_timing.get("score_seconds", ""),
                        "lane_total_seconds": lane_timing.get("total_seconds", ""),
                        "pipeline_total_seconds": pipeline_timing.get("total_seconds", ""),
                        "pipeline_rtf": pipeline_timing.get("rtf", ""),
                        **lane["sentence"],
                    }
                )
                for word in lane["words"]:
                    word_writer.writerow(
                        {
                            **common,
                            "lane": lane["name"],
                            "normalized_asr_text": lane["target"].get("normalized_text", ""),
                            "word": word.get("word", ""),
                            "target_phones": word.get("target_phones", ""),
                            "recognized_phones": word.get("recognized_phones", ""),
                            "PER": word.get("PER", ""),
                            "PFER": word.get("PFER", ""),
                            "feature_distance": word.get("feature_distance", ""),
                            "substitutions": word.get("substitutions", ""),
                            "deletions": word.get("deletions", ""),
                            "insertions": word.get("insertions", ""),
                            "substitutions_detail": word.get("substitutions_detail", ""),
                            "deletions_detail": word.get("deletions_detail", ""),
                            "insertions_detail": word.get("insertions_detail", ""),
                        }
                    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {words_path}")
    print(f"Wrote {jsonl_path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            required = {"id", "language", "audio"}
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"{path}:{line_number} missing {', '.join(missing)}")
            rows.append(row)
    return rows


if __name__ == "__main__":
    main()
