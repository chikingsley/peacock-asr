#!/usr/bin/env python
"""Two-path pronunciation-scoring eval over a FLEURS-style manifest.

For each sample the audio is recognized once (ZIPA) and two *target* sources are scored
against that single recognition:

  - read-aloud : target phones from G2P of the KNOWN reference text (manifest ``text``).
  - free-form  : target phones from G2P of the ElevenLabs Scribe ASR hypothesis.

Both collapse to the same funnel — text -> G2P -> canonical IPA  vs  recognized IPA -> PER/PFER
— so the only thing that differs is where the target text comes from. We have ground truth
(the reference) for both, so the read-aloud numbers show the funnel's ceiling and the free-form
numbers show what deriving the target via ASR costs.

read-aloud needs no ASR and runs fully locally. free-form needs Scribe (superwhisper-api) auth +
network, so it is opt-in via ``--free-form``.

    uv run capt-eval --manifest runs/<name>/manifest.jsonl \
        --out-dir runs/<name>/two_paths [--free-form]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

from capt.asr import ScribeAsrTranscriber
from capt.pipeline import DEFAULT_LANE_CONFIGS, PronunciationComparePipeline

if TYPE_CHECKING:
    from capt.pipeline import LaneResult

READ_ALOUD = "read_aloud"
FREE_FORM = "free_form"

SUMMARY_FIELDS = [
    "id", "language", "mode", "lane",
    "PER", "PFER", "feature_distance", "errors", "reference_count",
    "recognizer_error", "g2p_warnings", "target_text",
]
WORD_FIELDS = [
    "id", "language", "mode", "lane", "word",
    "target_phones", "recognized_phones", "PER", "PFER",
    "substitutions_detail", "deletions_detail", "insertions_detail",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/two_paths"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--free-form",
        dest="free_form",
        action="store_true",
        help="Also score free-form (target = Scribe ASR). Needs superwhisper-api auth + network.",
    )
    args = parser.parse_args()

    modes = [READ_ALOUD] + ([FREE_FORM] if args.free_form else [])
    rows = _read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PronunciationComparePipeline(lane_configs=DEFAULT_LANE_CONFIGS)
    scribe = ScribeAsrTranscriber() if args.free_form else None

    summary_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []
    results_path = args.out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file:
        for index, row in enumerate(rows, start=1):
            print(f"[{index}/{len(rows)}] {row['id']} ({row['language']})")
            payload = _score_sample(pipeline, scribe, row, modes, summary_rows, word_rows)
            results_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    _write_csv(args.out_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(args.out_dir / "words.csv", WORD_FIELDS, word_rows)
    report_path = args.out_dir / "report.md"
    report_path.write_text(_render_report(summary_rows, modes), encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {args.out_dir / 'summary.csv'}")
    print(f"Wrote {args.out_dir / 'words.csv'}")
    print(f"Wrote {report_path}")


def _score_sample(
    pipeline: PronunciationComparePipeline,
    scribe: ScribeAsrTranscriber | None,
    row: dict[str, Any],
    modes: list[str],
    summary_rows: list[dict[str, Any]],
    word_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recognize the audio once, then score each requested mode; append CSV rows in place."""
    audio = Path(row["audio"]).expanduser()
    if not audio.exists():
        raise FileNotFoundError(f"Missing audio for {row['id']}: {audio}")
    language = str(row["language"])
    targets = {READ_ALOUD: str(row.get("text", "")).strip()}
    if FREE_FORM in modes and scribe is not None:
        targets[FREE_FORM] = scribe.transcribe(str(audio), language=language).text

    recognitions = pipeline.recognize(audio)
    payload: dict[str, Any] = {"sample": row, "modes": {}}
    for mode in modes:
        target_text = targets.get(mode, "")
        lanes = pipeline.score_text(language, target_text, recognitions)
        payload["modes"][mode] = {
            "target_text": target_text,
            "lanes": [_lane_payload(lane) for lane in lanes],
        }
        for lane in lanes:
            summary_rows.append(_summary_row(row, mode, target_text, lane))
            word_rows.extend(_word_row(row, mode, lane, word) for word in lane.words)
    return payload


def _lane_payload(lane: LaneResult) -> dict[str, Any]:
    return {
        "lane": lane.name,
        "recognizer_error": lane.recognition.error,
        "target_phones": lane.target.flat_normalized,
        "recognized_phones": lane.recognition.normalized_tokens,
        "g2p_warnings": lane.target.warnings,
        "sentence": lane.sentence,
        "words": lane.words,
    }


def _summary_row(
    row: dict[str, Any],
    mode: str,
    target_text: str,
    lane: LaneResult,
) -> dict[str, Any]:
    sentence = lane.sentence
    return {
        "id": row["id"],
        "language": row["language"],
        "mode": mode,
        "lane": lane.name,
        "PER": sentence.get("PER", ""),
        "PFER": sentence.get("PFER", ""),
        "feature_distance": sentence.get("feature_distance", ""),
        "errors": sentence.get("errors", ""),
        "reference_count": sentence.get("reference_count", ""),
        "recognizer_error": lane.recognition.error or "",
        "g2p_warnings": "; ".join(lane.target.warnings),
        "target_text": target_text,
    }


def _word_row(
    row: dict[str, Any],
    mode: str,
    lane: LaneResult,
    word: dict[str, str | int | float],
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "language": row["language"],
        "mode": mode,
        "lane": lane.name,
        "word": word.get("word", ""),
        "target_phones": word.get("target_phones", ""),
        "recognized_phones": word.get("recognized_phones", ""),
        "PER": word.get("PER", ""),
        "PFER": word.get("PFER", ""),
        "substitutions_detail": word.get("substitutions_detail", ""),
        "deletions_detail": word.get("deletions_detail", ""),
        "insertions_detail": word.get("insertions_detail", ""),
    }


def _render_report(summary_rows: list[dict[str, Any]], modes: list[str]) -> str:
    """Aggregate avg PER/PFER by (language, mode, lane) into a markdown report."""
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for srow in summary_rows:
        buckets[(srow["language"], srow["mode"], srow["lane"])].append(srow)

    lines = [
        "# P016 Two-Path Eval",
        "",
        "Funnel: text -> G2P -> canonical IPA  vs  recognized IPA -> PER/PFER.",
        f"Modes: {', '.join(modes)} (read_aloud target = reference text; "
        "free_form target = Scribe ASR).",
        "",
        "| language | mode | lane | n | avg PER | avg PFER |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for key in sorted(buckets):
        language, mode, lane = key
        group = buckets[key]
        lines.append(
            f"| {language} | {mode} | {lane} | {len(group)} | "
            f"{_avg(group, 'PER')} | {_avg(group, 'PFER')} |"
        )
    return "\n".join(lines) + "\n"


def _avg(group: list[dict[str, Any]], field: str) -> str:
    values = [float(srow[field]) for srow in group if srow[field] != ""]
    return f"{mean(values):.4f}" if values else "n/a"


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            missing = sorted({"id", "language", "audio"} - set(row))
            if missing:
                raise ValueError(f"{path}:{line_number} missing {', '.join(missing)}")
            rows.append(row)
    return rows


if __name__ == "__main__":
    main()
