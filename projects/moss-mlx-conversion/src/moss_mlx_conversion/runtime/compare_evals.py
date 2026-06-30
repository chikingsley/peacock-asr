from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from moss_mlx_conversion.dump import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two MOSS eval prediction JSONL files.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    left_rows = read_predictions(args.left)
    right_rows = read_predictions(args.right)
    common_ids = sorted(set(left_rows) & set(right_rows))
    diffs: list[dict[str, Any]] = []
    counts = {
        "exact_hypothesis_match": 0,
        "normalized_hypothesis_match": 0,
        "first_5_new_ids_match": 0,
        "left_lower_wer": 0,
        "right_lower_wer": 0,
        "equal_wer": 0,
    }

    for example_id in common_ids:
        left = left_rows[example_id]
        right = right_rows[example_id]
        exact_match = left.get("hypothesis") == right.get("hypothesis")
        normalized_match = left.get("hypothesis_normalized") == right.get(
            "hypothesis_normalized"
        )
        first_5_match = left.get("first_5_new_ids") == right.get("first_5_new_ids")
        left_wer = float(left["wer"])
        right_wer = float(right["wer"])
        if exact_match:
            counts["exact_hypothesis_match"] += 1
        if normalized_match:
            counts["normalized_hypothesis_match"] += 1
        if first_5_match:
            counts["first_5_new_ids_match"] += 1
        if left_wer < right_wer:
            counts["left_lower_wer"] += 1
        elif right_wer < left_wer:
            counts["right_lower_wer"] += 1
        else:
            counts["equal_wer"] += 1
        diffs.append(
            {
                "id": example_id,
                "row_idx": left.get("row_idx"),
                "reference": left.get("reference"),
                "exact_hypothesis_match": exact_match,
                "normalized_hypothesis_match": normalized_match,
                "first_5_new_ids_match": first_5_match,
                "left_wer": left_wer,
                "right_wer": right_wer,
                "wer_delta_left_minus_right": left_wer - right_wer,
                "left_hypothesis": left.get("hypothesis"),
                "right_hypothesis": right.get("hypothesis"),
            }
        )

    diffs_by_abs_delta = sorted(
        diffs,
        key=lambda row: abs(float(row["wer_delta_left_minus_right"])),
        reverse=True,
    )
    summary = {
        "left_name": args.left_name,
        "right_name": args.right_name,
        "left_path": str(args.left),
        "right_path": str(args.right),
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "compared": len(common_ids),
        "missing_left": sorted(set(right_rows) - set(left_rows)),
        "missing_right": sorted(set(left_rows) - set(right_rows)),
        **counts,
        "top_deltas": diffs_by_abs_delta[: args.top_n],
    }

    write_json(output_dir / "summary.json", summary)
    with (output_dir / "diffs.jsonl").open("w", encoding="utf-8") as handle:
        for row in diffs:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"comparison summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
