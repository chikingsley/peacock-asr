from __future__ import annotations

import argparse
import time
from pathlib import Path

from asr_benchmark_core.adapters import Adapter, load_adapter
from asr_benchmark_core.data import Example, examples
from asr_benchmark_core.store import BenchmarkStore, Prediction


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--adapter", choices=("whisper", "qwen", "omni"), required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--benchmark", type=Path, required=True)
    result.add_argument("--database", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--language", default="Persian")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--offset", type=int, default=0)
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=8)
    return result


def _transcribe_batch(
    adapter: Adapter, batch: list[Example]
) -> tuple[list[str], list[str | None], float]:
    inference_started = time.perf_counter()
    try:
        hypotheses = adapter.transcribe_batch(batch)
        errors: list[str | None] = [None] * len(batch)
    except Exception as exc:  # noqa: BLE001 - preserve per-row failures and resume
        hypotheses = [""] * len(batch)
        errors = [f"{type(exc).__name__}: {exc}"] * len(batch)
    inference_seconds = (time.perf_counter() - inference_started) / len(batch)
    return hypotheses, errors, inference_seconds


def _store_batch(
    store: BenchmarkStore,
    adapter: Adapter,
    run_id: str,
    batch: list[Example],
) -> list[str | None]:
    hypotheses, errors, inference_seconds = _transcribe_batch(adapter, batch)
    for item, hypothesis, error in zip(batch, hypotheses, errors, strict=True):
        store.add(
            run_id,
            Prediction(
                row_index=item.row_index,
                reference=item.reference,
                hypothesis=hypothesis,
                audio_seconds=item.audio_seconds,
                inference_seconds=inference_seconds,
                error=error,
            ),
        )
    return errors


def run(args: argparse.Namespace) -> int:
    for path_name in ("model", "benchmark"):
        path = getattr(args, path_name)
        if not path.exists():
            raise SystemExit(f"{path_name} path does not exist: {path}")

    store = BenchmarkStore(args.database)
    config = {
        "device": args.device,
        "limit": args.limit,
        "offset": args.offset,
        "batch_size": args.batch_size,
    }
    store.ensure_run(
        run_id=args.run_id,
        adapter=args.adapter,
        model_path=args.model,
        benchmark_path=args.benchmark,
        language=args.language,
        config=config,
    )
    completed = store.completed_rows(args.run_id)
    print(f"run={args.run_id} completed={len(completed)}", flush=True)
    adapter = load_adapter(
        args.adapter,
        args.model,
        language=args.language,
        device=args.device,
    )
    processed = 0
    started = time.perf_counter()
    try:
        batch: list[Example] = []
        for example in examples(args.benchmark, offset=args.offset, limit=args.limit):
            if example.row_index not in completed:
                batch.append(example)
            if len(batch) < args.batch_size:
                continue
            errors = _store_batch(store, adapter, args.run_id, batch)
            processed += len(batch)
            elapsed = time.perf_counter() - started
            print(
                f"processed={processed} row={batch[-1].row_index} "
                f"elapsed={elapsed:.1f}s error={any(errors)}",
                flush=True,
            )
            batch = []
        if batch:
            _store_batch(store, adapter, args.run_id, batch)
            processed += len(batch)
    finally:
        store.close()
    print(f"done run={args.run_id} new_rows={processed}", flush=True)
    return 0


def main() -> int:
    return run(parser().parse_args())
