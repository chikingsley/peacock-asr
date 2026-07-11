"""Balance one Scribe-API concurrency budget across the active Scribe jobs (the factory's split).

Reads a single budget — the total concurrent Scribe calls the SuperWhisper service can take — and
divides it across whichever ``<lang>-curate verify``/``labelq`` jobs are running, writing each
one's live window file (``data/.scribe_window``). Those jobs re-read that file every batch and
retune, so this is the one knob that keeps their sum within budget and rebalances automatically as
jobs start and finish.

    uv run -m omni_curator.scribe.balance --budget 300             # one-shot
    uv run -m omni_curator.scribe.balance --budget 300 --watch 30  # rebalance every 30s
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from omni_curator.scribe.concurrency import split_budget, write_window

#: Matches a Scribe job argv, e.g. "... russian-curate verify ..." -> ("russian", "verify").
#: Both stages count as DISTINCT jobs: labelq and verify for one language run concurrently and each
#: reads its own window file, so the budget must be split per (lang, command), not per language.
_JOB_RE = re.compile(r"(\w+)-curate\s+(verify|labelq)\b")


def parse_jobs(ps_output: str) -> list[str]:
    """``lang:command`` keys of running Scribe jobs from ``ps`` output, de-duped and sorted."""
    return sorted({f"{m.group(1)}:{m.group(2)}" for m in _JOB_RE.finditer(ps_output)})


def active_scribe_jobs() -> list[str]:
    """Lang keys with a running verify/labelq job (e.g. ``['georgian', 'russian']``)."""
    proc = subprocess.run(
        ["ps", "-eo", "args"],
        capture_output=True,
        text=True,
        check=False,
    )
    return parse_jobs(proc.stdout)


def window_path(job: str, *, root: Path) -> Path:
    """Live-window file for a ``lang:command`` job (its ``data/.scribe_window.<command>``)."""
    lang, _, command = job.partition(":")
    return root / "projects" / f"{lang}-asr" / "data" / f".scribe_window.{command}"


def apply_budget(budget: int, jobs: list[str], *, root: Path) -> dict[str, int]:
    """Split ``budget`` across ``jobs`` and write each window atomically; return the map."""
    assignment = split_budget(budget, jobs)
    for job, window in assignment.items():
        write_window(window_path(job, root=root), window)
    return assignment


def _default_root() -> Path:
    """Walk up from CWD to the repo root (the dir holding ``projects/``); fall back to CWD."""
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "projects").is_dir():
            return candidate
    return here


def balance_once(budget: int, *, root: Path) -> dict[str, int]:
    """Detect active jobs, split the budget, write each window; print + return assignments."""
    jobs = active_scribe_jobs()
    if not jobs:
        print("no active scribe jobs (verify/labelq) -- nothing to balance", flush=True)
        return {}
    if budget < len(jobs):
        print(
            f"warning: budget {budget} < {len(jobs)} active jobs; each gets 1 "
            f"(their sum {len(jobs)} exceeds the budget)",
            flush=True,
        )
    assignment = apply_budget(budget, jobs, root=root)
    print(
        f"budget {budget} over {len(jobs)} job(s): "
        + ", ".join(f"{k}={v}" for k, v in assignment.items()),
        flush=True,
    )
    return assignment


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Balance the Scribe concurrency budget across jobs.")
    ap.add_argument("--budget", type=int, required=True, help="total concurrent Scribe calls")
    ap.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SECS",
        help="rebalance every SECS (0 = one-shot, the default)",
    )
    ap.add_argument("--root", type=Path, default=None, help="repo root (default: auto-detect)")
    args = ap.parse_args(argv)
    root = args.root or _default_root()
    balance_once(args.budget, root=root)
    while args.watch > 0:
        time.sleep(args.watch)
        balance_once(args.budget, root=root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
