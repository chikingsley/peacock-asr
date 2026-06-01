#!/usr/bin/env python
"""Per-language G2P ablation — pick the best G2P backend per language for the ZIPA funnel.

For each language in the manifest we recognize every clip once with ZIPA, then score the KNOWN
reference text (read-aloud) against each candidate G2P backend by mean PFER (panphon feature
distance vs the ZIPA output). The lowest-PFER backend per language is written to the routing
table (src/p016_compare/g2p_routing.json), which TargetG2P(backend="routed") then consumes.

Processing is per-language and CHECKPOINTED: the routing table + report are updated after each
language finishes, so a long multi-language run is crash-resumable. ``--skip-routed`` skips
languages already in the table, so re-running continues where it left off. This is the loop for
finishing FLEURS — one big manifest, run once, resume if interrupted.

    uv run --extra zipa python scripts/g2p_ablation.py --manifest <manifest> --skip-routed
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from p016_compare.g2p import TargetG2P
from p016_compare.pipeline import PronunciationComparePipeline, _score_lane

DEFAULT_CANDIDATES = ["espeak", "epitran", "charsiu", "paper"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/two_paths/g2p_ablation"))
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--routing-out",
        type=Path,
        default=Path("src/p016_compare/g2p_routing.json"),
        help="Routing table to update (merged + checkpointed after each language).",
    )
    parser.add_argument(
        "--skip-routed",
        action="store_true",
        help="Skip languages already present in the routing table (resume an interrupted run).",
    )
    args = parser.parse_args()

    by_language = _group_by_language(_read_jsonl(args.manifest))
    pipeline = PronunciationComparePipeline()
    routing = _load_json(args.routing_out)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "report.md"

    for language, rows in by_language.items():
        if args.skip_routed and language in routing:
            print(f"-- skip {language} (already routed: {routing[language]})", flush=True)
            continue
        means, actual = _ablate_language(pipeline, language, rows, args.candidates)
        if not means:
            print(f"-- {language}: no scores", flush=True)
            continue
        best = min(means, key=lambda c: means[c])
        routing[language] = best
        _write_json(args.routing_out, routing)
        _append_report(report_path, language, best, means, actual, args.candidates)
        print(f"== {language} -> {best} ({means[best]:.4f})", flush=True)

    print(f"Done. Routing table: {args.routing_out} ({len(routing)} languages)")


def _ablate_language(
    pipeline: PronunciationComparePipeline,
    language: str,
    rows: list[dict[str, Any]],
    candidates: list[str],
) -> tuple[dict[str, float], dict[str, str]]:
    """Recognize each clip once, score the reference against each candidate; return mean PFER."""
    pfer: dict[str, list[float]] = defaultdict(list)
    actual: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        audio = Path(row["audio"]).expanduser()
        if not audio.exists():
            raise FileNotFoundError(f"Missing audio for {row['id']}: {audio}")
        print(f"   [{language} {index}/{len(rows)}] {row['id']}", flush=True)
        reference = str(row.get("text", "")).strip()
        recognition = pipeline.recognize(audio)["zipa"]
        for candidate in candidates:
            try:
                target = TargetG2P(candidate).from_text(reference, language)
            except Exception as exc:  # noqa: BLE001 - candidate can't handle this language; skip it
                print(f"      {candidate} unavailable: {exc}", flush=True)
                continue
            value = _score_lane("zipa", target, recognition).sentence.get("PFER")
            if value != "" and value is not None:
                pfer[candidate].append(float(value))
                actual[candidate] = target.backend
    means = {c: statistics.mean(v) for c, v in pfer.items() if v}
    return means, actual


def _append_report(
    path: Path,
    language: str,
    best: str,
    means: dict[str, float],
    actual: dict[str, str],
    candidates: list[str],
) -> None:
    lines = [f"## {language} -> best: **{best}** ({means[best]:.4f})"]
    lines.extend(
        f"- {c:9} {means[c]:.4f}  (ran: {actual.get(c)})" for c in candidates if c in means
    )
    lines.append("")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _group_by_language(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["language"])].append(row)
    return grouped


def _load_json(path: Path) -> dict[str, str]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_json(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: data[k] for k in sorted(data)}
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


if __name__ == "__main__":
    main()
