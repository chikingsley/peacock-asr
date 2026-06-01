"""Throughput/memory tuning sweep for the Omni CTC recipe.

Finds the largest batch budget (``max_num_elements``) — optionally across several clip-length
caps (``max_audio_len``) — that fits under a GPU-memory ceiling, ranked by throughput. The
point is to stop hand-guessing the batch knobs for a given model + GPU.

Each candidate runs in its OWN subprocess so that an OOM (or a teardown hang) in one trial
cannot poison the next or leak GPU memory: the subprocess is killed by process group on
timeout, and the GPU is released when it exits. Memory + throughput are read back from the
recipe's ``metrics/train.jsonl`` (flushed per step, so the numbers survive a teardown hang).

Library surface: :func:`sweep` runs the trials and returns :class:`TrialResult` rows;
:func:`recommend` and :func:`format_report` turn them into a decision. A consuming project
wraps these in a CLI that first points the HF/fairseq2 caches at its own tree (see
``persian_omnilingual_asr.training.tune``). ``print`` lives in that wrapper, not here.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from omni_finetune_core.train import RECIPE_MODULE

if TYPE_CHECKING:
    from collections.abc import Sequence

_GIB = 2**30
# Push validation/checkpointing past the end of a short trial so they never fire.
_NEVER = 1_000_000_000
_OOM_MARKERS = ("out of memory", "outofmemoryerror", "cuda error: out of memory")


@dataclass(frozen=True)
class TrialResult:
    """One (max_audio_len, max_num_elements) trial: did it fit, and how fast."""

    max_audio_len: int
    max_num_elements: int
    status: str  # "ok" (fit, no OOM) | "oom" | "error"
    steps_seen: int
    peak_reserved_pct: float | None  # reserved is what actually triggers OOM
    peak_active_gib: float | None
    elements_per_sec: float | None
    note: str = ""


def _find_train_jsonl(output_dir: Path) -> Path | None:
    matches = sorted(output_dir.glob("ws_*/metrics/train.jsonl"))
    return matches[0] if matches else None


def _read_metrics(jsonl: Path) -> list[dict]:
    records: list[dict] = []
    with jsonl.open(encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _run_subprocess(argv: list[str], log_path: Path, timeout_s: float) -> int | None:
    """Run argv to completion, capturing output to log_path. Returns the exit code, or
    None if it timed out (the whole process group is killed so nothing lingers on the GPU).
    """
    with log_path.open("wb") as log:
        proc = subprocess.Popen(  # noqa: S603 - argv is built from trusted constants
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            return None


def _trial(
    config_file: Path,
    output_dir: Path,
    *,
    max_audio_len: int,
    max_num_elements: int,
    steps: int,
    timeout_s: float,
    extra_overrides: Sequence[str],
) -> TrialResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "trial.log"
    argv = [
        sys.executable,
        "-m",
        RECIPE_MODULE,
        str(output_dir),
        "--config-file",
        str(config_file),
        "--config",
        f"regime.num_steps={steps}",
        f"regime.validate_after_n_steps={_NEVER}",
        f"regime.checkpoint_after_n_steps={_NEVER}",
        "regime.publish_metrics_after_n_steps=0",
        "regime.publish_metrics_every_n_steps=1",
        f"dataset.asr_task_config.max_audio_len={max_audio_len}",
        f"dataset.asr_task_config.max_num_elements={max_num_elements}",
        *extra_overrides,
    ]
    returncode = _run_subprocess(argv, log_path, timeout_s)

    # fairseq2 exits 0 even on OOM, so returncode is unreliable — the log marker is
    # authoritative. Metrics are flushed per step, so they survive a teardown hang.
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    oom = any(marker in text for marker in _OOM_MARKERS)
    jsonl = _find_train_jsonl(output_dir)
    records = _read_metrics(jsonl) if jsonl else []

    if not records:
        status = "oom" if oom else "error"
        note = "OOM before first step" if oom else "no metrics produced (see trial.log)"
        return TrialResult(max_audio_len, max_num_elements, status, 0, None, None, None, note)

    peak_pct = max(r.get("Peak Reserved Device Memory (%)", 0.0) for r in records)
    peak_active = max(r.get("Peak Active Device Memory", 0) for r in records) / _GIB
    rates = [r["Elements per Second"] for r in records if r.get("Elements per Second")]
    eps = sorted(rates)[len(rates) // 2] if rates else None
    if oom:
        return TrialResult(
            max_audio_len, max_num_elements, "oom", len(records),
            peak_pct, peak_active, eps, "OOM after partial steps (budget too tight)",
        )
    note = "" if returncode is not None else "hung at teardown (metrics complete)"
    return TrialResult(
        max_audio_len, max_num_elements, "ok", len(records), peak_pct, peak_active, eps, note
    )


def sweep(
    config_file: Path,
    candidates: Sequence[tuple[int, int]],
    *,
    output_root: Path,
    steps: int = 8,
    timeout_s: float = 600.0,
    extra_overrides: Sequence[str] = (),
) -> list[TrialResult]:
    """Run each ``(max_audio_len, max_num_elements)`` candidate as its own short trial.

    ``extra_overrides`` are bare ``key=value`` recipe overrides appended to every trial's
    ``--config`` (e.g. ``common.cluster=none``). Returns one :class:`TrialResult` per candidate.
    """
    results: list[TrialResult] = []
    for index, (max_audio_len, max_num_elements) in enumerate(candidates):
        out = Path(output_root) / f"trial_{index:02d}_mal{max_audio_len}_mne{max_num_elements}"
        results.append(
            _trial(
                config_file,
                out,
                max_audio_len=max_audio_len,
                max_num_elements=max_num_elements,
                steps=steps,
                timeout_s=timeout_s,
                extra_overrides=extra_overrides,
            )
        )
    return results


def recommend(results: Sequence[TrialResult], *, mem_ceiling: float = 0.9) -> TrialResult | None:
    """The fittest candidate: highest throughput among trials that fit under the ceiling."""
    fits = [
        r
        for r in results
        if r.status == "ok"
        and r.peak_reserved_pct is not None
        and r.peak_reserved_pct <= mem_ceiling
    ]
    return max(fits, key=lambda r: r.elements_per_sec or 0.0) if fits else None


def format_report(results: Sequence[TrialResult], *, mem_ceiling: float = 0.9) -> str:
    """Render the sweep as a table plus a recommendation line."""
    header = (
        f"{'max_audio_len':>14} {'max_num_elem':>13} {'status':>7} {'steps':>6} "
        f"{'peak_resv%':>11} {'peak_act_GiB':>13} {'elem/s':>14}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        pct = f"{r.peak_reserved_pct * 100:.1f}" if r.peak_reserved_pct is not None else "-"
        active = f"{r.peak_active_gib:.2f}" if r.peak_active_gib is not None else "-"
        rate = f"{r.elements_per_sec:,.0f}" if r.elements_per_sec else "-"
        row = (
            f"{r.max_audio_len:>14} {r.max_num_elements:>13} {r.status:>7} {r.steps_seen:>6} "
            f"{pct:>11} {active:>13} {rate:>14}"
        )
        lines.append(f"{row}  {r.note}" if r.note else row)

    best = recommend(results, mem_ceiling=mem_ceiling)
    lines.append("")
    if best is None or best.peak_reserved_pct is None or best.elements_per_sec is None:
        lines.append(
            f"RECOMMEND: nothing fit under {mem_ceiling * 100:.0f}% reserved — "
            "lower max_num_elements or max_audio_len."
        )
    else:
        resv = best.peak_reserved_pct * 100
        lines.append(
            f"RECOMMEND: max_audio_len={best.max_audio_len} "
            f"max_num_elements={best.max_num_elements} "
            f"(peak reserved {resv:.1f}% <= {mem_ceiling * 100:.0f}%, "
            f"{best.elements_per_sec:,.0f} elem/s)"
        )
    return "\n".join(lines)
