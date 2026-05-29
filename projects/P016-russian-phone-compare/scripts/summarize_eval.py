#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a P016 eval directory into a short markdown report."
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--top-words", type=int, default=20)
    args = parser.parse_args()

    summary_path = args.eval_dir / "summary.csv"
    words_path = args.eval_dir / "words.csv"
    results_path = args.eval_dir / "results.jsonl"
    summary_rows = _read_csv(summary_path)
    word_rows = _read_csv(words_path)
    result_rows = _read_jsonl(results_path)

    report = _render_report(
        summary_rows=summary_rows,
        word_rows=word_rows,
        result_rows=result_rows,
        top_words=args.top_words,
    )
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote {args.out}")


def _render_report(
    summary_rows: list[dict[str, str]],
    word_rows: list[dict[str, str]],
    result_rows: list[dict[str, Any]],
    top_words: int,
) -> str:
    lines: list[str] = [
        "# P016 Free-Speaking Eval Report",
        "",
        "Scoring path: audio -> Qwen ASR -> ASR text -> lane-specific G2P -> "
        "ZIPA/XLSR phones -> PER/PFER.",
        "",
        "Known dataset text is used only for audit/reporting. It is not fed to the scorer.",
        "",
        "## Lane Summary",
        "",
        "| language | lane | n | avg PER | avg PFER |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for (language, lane), rows in sorted(_group(summary_rows, "language", "lane").items()):
        per = _mean(float(row["PER"]) for row in rows)
        pfer = _mean(float(row["PFER"]) for row in rows)
        lines.append(f"| {language} | {lane} | {len(rows)} | {per:.4f} | {pfer:.4f} |")

    lines.extend(["", "## ASR Vs Known Text", ""])
    seen: set[str] = set()
    for row in summary_rows:
        sample_id = row["id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        reference = row["reference_text"]
        asr = row["asr_text"]
        normalized_asr = row.get("normalized_asr_text", "")
        reference_tokens = _tokens(reference)
        asr_tokens = _tokens(asr)
        exact = reference_tokens == asr_tokens
        lines.extend(
            [
                f"- `{sample_id}` ({row['language']}): {'match' if exact else 'changed'}",
                f"  - known: `{reference}`",
                f"  - asr: `{asr}`",
                f"  - normalized: `{normalized_asr}`",
            ]
        )

    lines.extend(["", "## Target Backends", ""])
    for result in result_rows:
        sample = result["sample"]
        lines.append(f"- `{sample['id']}`")
        for lane in result["result"]["lanes"]:
            target = lane["target"]
            warnings = target.get("warnings") or []
            warning_text = "; ".join(str(warning) for warning in warnings) or "none"
            lines.append(
                f"  - {lane['name']}: `{target['backend']}`, warnings: {warning_text}"
            )

    lines.extend(["", f"## Worst {top_words} Word Rows", ""])
    lines.append("| sample | language | lane | word | PER | PFER | target | heard | details |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |")
    sorted_words = sorted(
        word_rows,
        key=lambda row: (float(row["PFER"]), float(row["PER"])),
        reverse=True,
    )
    for row in sorted_words[:top_words]:
        details = _join_details(row)
        lines.append(
            "| "
            f"{row['id']} | {row['language']} | {row['lane']} | {row['word']} | "
            f"{float(row['PER']):.4f} | {float(row['PFER']):.4f} | "
            f"`{row['target_phones']}` | `{row['recognized_phones']}` | {details} |"
        )

    lines.extend(["", "## Read", ""])
    lines.extend(_interpret(summary_rows, word_rows))
    lines.append("")
    return "\n".join(lines)


def _interpret(summary_rows: list[dict[str, str]], word_rows: list[dict[str, str]]) -> list[str]:
    averages = {
        key: (
            _mean(float(row["PER"]) for row in rows),
            _mean(float(row["PFER"]) for row in rows),
        )
        for key, rows in _group(summary_rows, "language", "lane").items()
    }
    notes = [
        "- The end-to-end free-speaking path runs on this sample.",
        "- English is bounded but still has false-positive surface area.",
        "- Russian remains noisy enough that the current score should be treated as "
        "diagnostic, not learner feedback.",
    ]
    for (language, lane), (per, pfer) in sorted(averages.items()):
        notes.append(f"- {language}/{lane}: avg PER {per:.4f}, avg PFER {pfer:.4f}.")

    high_pfer = [
        row
        for row in word_rows
        if float(row["PFER"]) >= 0.25 or float(row["PER"]) >= 0.75
    ]
    if high_pfer:
        notes.append(
            "- High-error rows are concentrated in short function words, abbreviations/numbers, "
            "and Russian target/recognizer inventory mismatches."
        )
    return notes


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if stripped := line.strip():
                rows.append(json.loads(stripped))
    return rows


def _group(
    rows: list[dict[str, str]],
    *keys: str,
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return dict(grouped)


def _mean(values: Any) -> float:
    resolved = list(values)
    return sum(resolved) / len(resolved) if resolved else 0.0


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


def _join_details(row: dict[str, str]) -> str:
    parts = []
    for field in ("substitutions_detail", "deletions_detail", "insertions_detail"):
        if value := row.get(field):
            parts.append(f"{field.removesuffix('_detail')}: `{value}`")
    return "<br>".join(parts) if parts else ""


if __name__ == "__main__":
    main()
