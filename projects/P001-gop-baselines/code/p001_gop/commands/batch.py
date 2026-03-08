"""Batch command helpers."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import TYPE_CHECKING, Any

from p001_gop.batch_config import (
    BatchCliDefaults,
    BatchResolvedJob,
    load_batch_spec,
    resolve_batch_jobs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from p001_gop.evaluate import EvalResult

logger = logging.getLogger("p001_gop")


def register_batch_parser(subparsers: Any) -> None:
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run a YAML-defined experiment batch",
    )
    batch_parser.add_argument(
        "--config",
        required=True,
        help="Batch YAML config path",
    )
    batch_parser.add_argument(
        "--output-dir",
        default="artifacts/batches",
        help=(
            "Directory where batch run folders are written "
            "(default: artifacts/batches)"
        ),
    )
    batch_parser.add_argument(
        "--device",
        default=None,
        help="Default device override for jobs (default: use settings)",
    )
    batch_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Default utterance limit for jobs (0 = all)",
    )
    batch_parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Default workers for jobs (0 = auto)",
    )
    batch_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Default no-cache behavior for jobs",
    )
    batch_parser.add_argument(
        "--score-variant",
        choices=["gop_sf", "logit_margin", "logit_combined"],
        default="gop_sf",
        help="Default score variant for jobs (default: gop_sf)",
    )
    batch_parser.add_argument(
        "--score-alpha",
        type=float,
        default=0.5,
        help="Default score alpha for jobs (default: 0.5)",
    )


def _write_batch_summary(rows: list[dict[str, str]], path: Path) -> None:
    columns = [
        "job_id",
        "backend",
        "mode",
        "score_variant",
        "score_alpha",
        "repeat",
        "seed",
        "status",
        "pcc",
        "pcc_low",
        "pcc_high",
        "mse",
        "n_phones",
        "duration_sec",
        "log_path",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_batch_aggregates(rows: list[dict[str, str]], path: Path) -> None:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            str(row["job_id"]),
            str(row["backend"]),
            str(row["mode"]),
            str(row["score_variant"]),
            str(row["score_alpha"]),
        )
        groups.setdefault(key, []).append(row)

    columns = [
        "job_id",
        "backend",
        "mode",
        "score_variant",
        "score_alpha",
        "n_runs",
        "n_success",
        "pcc_mean",
        "pcc_std",
        "mse_mean",
        "mse_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for (
            job_id, backend, mode, score_variant, score_alpha
        ), grouped_rows in groups.items():
            ok_rows = [row for row in grouped_rows if row["status"] == "ok"]
            pcc_values = [float(row["pcc"]) for row in ok_rows]
            mse_values = [float(row["mse"]) for row in ok_rows]
            if not pcc_values:
                pcc_mean = float("nan")
                pcc_std = float("nan")
                mse_mean = float("nan")
                mse_std = float("nan")
            elif len(pcc_values) == 1:
                pcc_mean = pcc_values[0]
                pcc_std = 0.0
                mse_mean = mse_values[0]
                mse_std = 0.0
            else:
                pcc_mean = mean(pcc_values)
                pcc_std = stdev(pcc_values)
                mse_mean = mean(mse_values)
                mse_std = stdev(mse_values)
            writer.writerow(
                {
                    "job_id": job_id,
                    "backend": backend,
                    "mode": mode,
                    "score_variant": score_variant,
                    "score_alpha": score_alpha,
                    "n_runs": len(grouped_rows),
                    "n_success": len(ok_rows),
                    "pcc_mean": f"{pcc_mean:.4f}" if pcc_values else "nan",
                    "pcc_std": f"{pcc_std:.4f}" if pcc_values else "nan",
                    "mse_mean": f"{mse_mean:.4f}" if pcc_values else "nan",
                    "mse_std": f"{mse_std:.4f}" if pcc_values else "nan",
                }
            )


def _run_batch_repeat(
    *,
    run_count: int,
    run_dir: Path,
    job: BatchResolvedJob,
    repeat: int,
    seed: int | None,
    run_command: Callable[[argparse.Namespace], tuple[str, EvalResult]],
    mode_name_fn: Callable[..., str],
) -> tuple[dict[str, str], bool]:
    use_gopt = job.mode == "gopt"
    use_hmamba = job.mode == "hmamba"
    use_feats = job.mode == "feats"
    mode_name = mode_name_fn(
        use_gopt=use_gopt,
        use_hmamba=use_hmamba,
        use_feats=use_feats,
    )

    run_tag = f"{job.job_id}_r{repeat}"
    log_path = run_dir / f"{run_tag}.log"
    logger.info(
        (
            "[batch] (%d) start job=%s backend=%s mode=%s variant=%s "
            "alpha=%.4f repeat=%d seed=%s"
        ),
        run_count,
        job.job_id,
        job.backend,
        job.mode,
        job.score_variant,
        job.score_alpha,
        repeat,
        seed if seed is not None else "none",
    )

    run_args = argparse.Namespace(
        backend=job.backend,
        feats=use_feats,
        gopt=use_gopt,
        hmamba=use_hmamba,
        device=job.device,
        limit=job.limit,
        no_cache=job.no_cache,
        workers=job.workers,
        seed=seed,
        score_variant=job.score_variant,
        score_alpha=job.score_alpha,
        verbose=job.verbose,
    )

    started = perf_counter()
    status = "ok"
    error = ""
    pcc = float("nan")
    pcc_low = float("nan")
    pcc_high = float("nan")
    mse = float("nan")
    n_phones = 0

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            "batch_job="
            f"{job.job_id} repeat={repeat} backend={job.backend} mode={job.mode} "
            f"score_variant={job.score_variant} score_alpha={job.score_alpha:.4f} "
            f"seed={seed if seed is not None else 'none'}\n"
        )
        log_file.flush()
        try:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                _, eval_result = run_command(run_args)
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=log_file)
        else:
            pcc = float(eval_result.pcc)
            pcc_low = float(eval_result.pcc_low)
            pcc_high = float(eval_result.pcc_high)
            mse = float(eval_result.mse)
            n_phones = int(eval_result.n_phones)

    duration_sec = perf_counter() - started
    row = {
        "job_id": job.job_id,
        "backend": job.backend,
        "mode": mode_name,
        "score_variant": job.score_variant,
        "score_alpha": f"{job.score_alpha:.4f}",
        "repeat": str(repeat),
        "seed": str(seed) if seed is not None else "",
        "status": status,
        "pcc": f"{pcc:.4f}" if status == "ok" else "nan",
        "pcc_low": f"{pcc_low:.4f}" if status == "ok" else "nan",
        "pcc_high": f"{pcc_high:.4f}" if status == "ok" else "nan",
        "mse": f"{mse:.4f}" if status == "ok" else "nan",
        "n_phones": str(n_phones) if status == "ok" else "",
        "duration_sec": f"{duration_sec:.2f}",
        "log_path": str(log_path),
        "error": error,
    }
    logger.info(
        "[batch] done job=%s repeat=%d status=%s pcc=%s mse=%s",
        job.job_id,
        repeat,
        status,
        row["pcc"],
        row["mse"],
    )
    return row, status != "ok"


def cmd_batch(
    args: argparse.Namespace,
    *,
    run_command: Callable[[argparse.Namespace], tuple[str, EvalResult]],
    mode_name_fn: Callable[..., str],
) -> None:
    """Run a YAML-defined batch of runs."""
    spec_path = Path(args.config)
    spec = load_batch_spec(spec_path)
    cli_defaults = BatchCliDefaults(
        device=args.device,
        limit=args.limit,
        workers=args.workers,
        no_cache=args.no_cache,
        verbose=args.verbose,
        score_variant=args.score_variant,
        score_alpha=args.score_alpha,
    )
    jobs = resolve_batch_jobs(spec, cli_defaults=cli_defaults)

    safe_batch_name = re.sub(r"[^\w\-.]", "_", spec.name).strip("_") or "batch"
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{stamp}_{safe_batch_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_spec_copy = run_dir / "batch_spec.yaml"
    run_spec_copy.write_text(spec_path.read_text(encoding="utf-8"), encoding="utf-8")

    rows: list[dict[str, str]] = []
    total_failed = 0
    run_count = 0

    for job in jobs:
        for repeat, seed in enumerate(job.seeds, start=1):
            run_count += 1
            row, failed = _run_batch_repeat(
                run_count=run_count,
                run_dir=run_dir,
                job=job,
                repeat=repeat,
                seed=seed,
                run_command=run_command,
                mode_name_fn=mode_name_fn,
            )
            rows.append(row)
            if failed:
                total_failed += 1

    summary_path = run_dir / "summary.tsv"
    aggregate_path = run_dir / "aggregates.tsv"
    _write_batch_summary(rows, summary_path)
    _write_batch_aggregates(rows, aggregate_path)
    logger.info("[batch] summary: %s", summary_path)
    logger.info("[batch] aggregates: %s", aggregate_path)

    if total_failed > 0:
        logger.error("[batch] %d run(s) failed", total_failed)
        sys.exit(1)
