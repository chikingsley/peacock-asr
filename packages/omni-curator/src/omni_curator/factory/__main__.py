"""Factory CLI: ``uv run -m omni_curator.factory`` (factory_plan §7), v0.

    uv run -m omni_curator.factory --once                 # one reconcile tick (cron/testing)
    uv run -m omni_curator.factory --once --dry-run       # log what it WOULD launch, spawn nothing
    uv run -m omni_curator.factory                         # the daemon loop (TICK=30s)
    uv run -m omni_curator.factory --projects dari,farsi   # select projects (default: discover)

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
    DEFAULT_CLIPS_ROOT,
    DEFAULT_CREATE_ROOT,
    DEFAULT_TICK_S,
    REPO_ROOT,
    Supervisor,
    discover_projects,
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
        description="omni-curator factory (v0): auto-drive enqueue + segment across projects.",
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
        "--log-file", type=Path, default=None, metavar="PATH",
        help="factory log path (default <repo-root>/factory.log)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = [n.strip() for n in args.projects.split(",") if n.strip()] if args.projects else None
    projects = discover_projects(
        names,
        repo_root=args.repo_root,
        clips_root=args.clips_root,
        create_root=args.create_root,
    )
    log_path = args.log_file or (args.repo_root / "factory.log")
    log = _make_logger(log_path)
    if not projects:
        log("no projects registered (discover found none; pass --projects)")
        return 1
    log(
        f"factory v0 start: projects={[p.name for p in projects]} "
        f"once={args.once} dry_run={args.dry_run} tick={args.tick:g}s"
    )
    supervisor = Supervisor(projects=projects, log=log, dry_run=args.dry_run)
    if args.once:
        launched = supervisor.tick()
        log(f"tick done: launched={launched}")
        return 0
    supervisor.run(tick_s=args.tick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
