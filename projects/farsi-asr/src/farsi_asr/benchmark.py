from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

LANGUAGE = "fas_Arab"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Score a shared ASR benchmark run with Peacock's Persian normalizer."
    )
    result.add_argument("database", type=Path)
    result.add_argument("--run-id", required=True)
    return result


def run(args: argparse.Namespace) -> int:
    from omni_curator.process import normalize
    from omni_finetune_core.metrics import compute_measures

    if not args.database.is_file():
        raise SystemExit(f"benchmark database does not exist: {args.database}")
    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    run_row = connection.execute(
        "SELECT adapter, model_path, benchmark_path FROM runs WHERE run_id = ?", (args.run_id,)
    ).fetchone()
    if run_row is None:
        raise SystemExit(f"run_id does not exist: {args.run_id}")
    rows = connection.execute(
        "SELECT reference, hypothesis, audio_seconds, inference_seconds, error "
        "FROM predictions WHERE run_id = ? ORDER BY row_index",
        (args.run_id,),
    ).fetchall()
    connection.close()

    pairs = [
        (normalize(str(reference), LANGUAGE), normalize(str(hypothesis), LANGUAGE))
        for reference, hypothesis, *_ in rows
        if normalize(str(reference), LANGUAGE).strip()
    ]
    if not pairs:
        raise SystemExit(f"run has no scorable predictions: {args.run_id}")
    measures = compute_measures([ref for ref, _ in pairs], [hyp for _, hyp in pairs])
    audio_seconds = sum(float(row[2]) for row in rows)
    inference_seconds = sum(float(row[3]) for row in rows)
    errors = sum(row[4] is not None for row in rows)
    empty_hypotheses = sum(not normalize(str(row[1]), LANGUAGE).strip() for row in rows)
    rtfx = audio_seconds / inference_seconds if inference_seconds else 0.0
    adapter, model_path, benchmark_path = run_row
    print(f"run_id: {args.run_id}")
    print(f"adapter: {adapter}")
    print(f"model: {model_path}")
    print(f"benchmark: {benchmark_path}")
    print(f"rows: {len(rows)}")
    print(f"wer: {measures.wer:.2f}")
    print(f"cer: {measures.cer:.2f}")
    print(f"audio_seconds: {audio_seconds:.2f}")
    print(f"inference_seconds: {inference_seconds:.2f}")
    print(f"rtfx: {rtfx:.2f}")
    print(f"errors: {errors}")
    print(f"empty_hypotheses: {empty_hypotheses}")
    return 0


def main() -> int:
    return run(parser().parse_args())
