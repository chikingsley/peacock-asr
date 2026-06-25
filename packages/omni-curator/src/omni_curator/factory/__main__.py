"""Factory CLI: ``uv run -m omni_curator.factory`` (factory_plan §7).

    uv run -m omni_curator.factory --once                 # one reconcile tick (cron/testing)
    uv run -m omni_curator.factory --once --dry-run       # log what it WOULD launch, spawn nothing
    uv run -m omni_curator.factory                         # the daemon loop (TICK=30s)
    uv run -m omni_curator.factory --projects dari,farsi   # select projects (default: discover)
    uv run -m omni_curator.factory --stages enqueue,segment  # segment-only (default: all stages)
    uv run -m omni_curator.factory --budget 300            # split 300 Scribe slots labelq/verify

``--once`` does a single pass and exits (the unit of cron / headless testing). ``--dry-run`` makes a
pass launch nothing — it only logs the commands it would run — so the reconcile logic is testable
without spawning real stages. Launches/exits are appended to ``factory.log`` (and echoed to stdout).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from omni_curator.factory.supervisor import (
    DEFAULT_ARCHIVE_ROOT,
    DEFAULT_CLIPS_ROOT,
    DEFAULT_CREATE_ROOT,
    DEFAULT_TICK_S,
    REPO_ROOT,
    Supervisor,
    discover_projects,
    have_flock_binary,
    resolve_stages,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_logger(log_path: Path) -> Callable[[str], None]:
    """A logger that timestamps each line, appends to ``factory.log``, and echoes to stdout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        stamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(stamped, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(stamped + "\n")

    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omni_curator.factory",
        description="omni-curator factory: auto-drive the curate pipeline across projects.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="run a single reconcile tick and exit (for cron / testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="log what each stage WOULD launch, but spawn nothing (headless reconcile test)",
    )
    parser.add_argument(
        "--projects", metavar="dari,farsi,...",
        help="comma-separated lang names to drive (default: discover projects/<lang>-asr)",
    )
    parser.add_argument(
        "--stages", metavar="enqueue,segment,...", default=None,
        help="comma-separated stages to drive, or 'all' (default all): "
             "enqueue,segment,labelq,harvest,archive",
    )
    parser.add_argument(
        "--budget", type=int, default=None, metavar="N",
        help="total concurrent Scribe calls to split across live labelq+verify jobs (off if unset)",
    )
    parser.add_argument(
        "--tick", type=float, default=DEFAULT_TICK_S, metavar="SECONDS",
        help=f"daemon loop interval (default {DEFAULT_TICK_S:g}s)",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT,
        help="repo root holding projects/<lang>-asr",
    )
    parser.add_argument(
        "--clips-root", type=Path, default=Path(DEFAULT_CLIPS_ROOT), metavar="DIR",
        help=f"SSD clips root; segment writes <DIR>/<lang> (default {DEFAULT_CLIPS_ROOT})",
    )
    parser.add_argument(
        "--create-root", type=Path, default=Path(DEFAULT_CREATE_ROOT), metavar="DIR",
        help=f"SSD create root; enqueue scans <DIR>/<lang> (default {DEFAULT_CREATE_ROOT})",
    )
    parser.add_argument(
        "--archive-root", type=Path, default=Path(DEFAULT_ARCHIVE_ROOT), metavar="DIR",
        help=f"archive destination; sources drain to <DIR>/<lang> (default {DEFAULT_ARCHIVE_ROOT})",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None, metavar="PATH",
        help="factory log path (default <repo-root>/factory.log)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = resolve_stages(args.stages)  # raises on an unknown stage name
    names = [n.strip() for n in args.projects.split(",") if n.strip()] if args.projects else None
    projects = discover_projects(
        names,
        repo_root=args.repo_root,
        clips_root=args.clips_root,
        create_root=args.create_root,
        archive_root=args.archive_root,
    )
    log_path = args.log_file or (args.repo_root / "factory.log")
    log = _make_logger(log_path)
    if not projects:
        log("no projects registered (discover found none; pass --projects)")
        return 1
    # The launch wrapper is flock(1); without it the per-stage lock + segment cap can't hold.
    if not args.dry_run and not have_flock_binary():
        log("FATAL: the flock(1) utility is not on PATH; stage locks/cap can't hold. Aborting.")
        return 1
    log(
        f"factory start: projects={[p.name for p in projects]} stages={list(stages)} "
        f"budget={args.budget} once={args.once} dry_run={args.dry_run} tick={args.tick:g}s"
    )
    supervisor = Supervisor(
        projects=projects, log=log, dry_run=args.dry_run, stages=stages,
        budget=args.budget, repo_root=args.repo_root,
    )
    if args.once:
        launched = supervisor.tick()
        log(f"tick done: launched={launched}")
        return 0
    supervisor.run(tick_s=args.tick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
