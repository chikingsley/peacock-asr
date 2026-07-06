"""Factory CLI: ``uv run -m omni_curator.factory``.

    uv run -m omni_curator.factory --once                 # one reconcile tick (cron/testing)
    uv run -m omni_curator.factory --once --dry-run       # log what it WOULD launch, spawn nothing
    uv run -m omni_curator.factory                         # the daemon loop (TICK=30s)
    uv run -m omni_curator.factory --projects dari,farsi   # select projects (default: discover)
    uv run -m omni_curator.factory --stages enqueue,segment  # segment-only (default: all stages)
    uv run -m omni_curator.factory --budget 300            # split 300 Scribe slots labelq/verify

``--once`` does a single pass and exits (the unit of cron / headless testing). ``--dry-run`` makes a
pass launch nothing — it only logs the commands it would run — so the reconcile logic is testable
without spawning real stages. Launches/exits are printed to stdout; when a log path is configured,
they are appended there too.
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from omni_curator.factory.supervisor import (
    DEFAULT_ARCHIVE_ROOT,
    DEFAULT_CLIPS_ROOT,
    DEFAULT_TICK_S,
    REPO_ROOT,
    FactorySettings,
    Supervisor,
    discover_projects,
    have_flock_binary,
    resolve_stages,
)

if TYPE_CHECKING:
    from collections.abc import Callable

def _make_logger(log_path: Path | None) -> Callable[[str], None]:
    """A logger that timestamps each line and optionally appends to the configured log."""
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        stamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(stamped, flush=True)
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(stamped + "\n")

    return log


def _load_config(path: Path | None) -> dict[str, object]:
    """Load a flat TOML factory config. Missing config means no overrides."""
    if path is None:
        return {}
    with path.open("rb") as fh:
        loaded = tomllib.load(fh)
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _cfg_path(cfg: dict[str, object], key: str, default: Path) -> Path:
    value = cfg.get(key)
    return Path(value) if isinstance(value, str) else default


def _cfg_optional_path(cfg: dict[str, object], key: str) -> Path | None:
    value = cfg.get(key)
    return Path(value) if isinstance(value, str) else None


def _cfg_float(cfg: dict[str, object], key: str, default: float) -> float:
    value = cfg.get(key)
    return float(value) if isinstance(value, int | float) else default


def _cfg_int(cfg: dict[str, object], key: str, default: int) -> int:
    value = cfg.get(key)
    return int(value) if isinstance(value, int) else default


def _cfg_optional_int(cfg: dict[str, object], key: str, default: int | None) -> int | None:
    value = cfg.get(key)
    if value is None:
        return default
    return int(value) if isinstance(value, int) else default


def _cfg_csv_or_list(cfg: dict[str, object], key: str) -> str | None:
    value = cfg.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return ",".join(str(item) for item in value)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omni_curator.factory",
        description="omni-curator factory: auto-drive the curate pipeline across projects.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="flat TOML config path for roots, workers, HWM, min free GB, stages, projects, budget",
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
        "--projects", metavar="dari,farsi,...", default=None,
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
        "--tick", type=float, default=None, metavar="SECONDS",
        help=f"daemon loop interval (default {DEFAULT_TICK_S:g}s)",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=None,
        help="repo root holding projects/<lang>-asr",
    )
    parser.add_argument(
        "--clips-root", type=Path, default=None, metavar="DIR",
        help=f"SSD clips root; segment writes <DIR>/<lang> (default {DEFAULT_CLIPS_ROOT})",
    )
    parser.add_argument(
        "--create-root", type=Path, default=None, metavar="DIR",
        help="optional scratch create root; enqueue scans <DIR>/<lang> (default project create)",
    )
    parser.add_argument(
        "--archive-root", type=Path, default=None, metavar="DIR",
        help=f"archive destination; sources drain to <DIR>/<lang> (default {DEFAULT_ARCHIVE_ROOT})",
    )
    parser.add_argument("--segment-gpu-procs", type=int, default=None)
    parser.add_argument("--segment-cpu-procs", type=int, default=None)
    parser.add_argument("--pending-hwm", type=int, default=None)
    parser.add_argument("--min-free-gb", type=float, default=None)
    parser.add_argument(
        "--log-file", type=Path, default=None, metavar="PATH",
        help="append factory logs here (default: stdout only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(args.config)
    repo_root = args.repo_root or _cfg_path(cfg, "repo_root", REPO_ROOT)
    clips_root = args.clips_root or _cfg_path(cfg, "clips_root", Path(DEFAULT_CLIPS_ROOT))
    create_root = args.create_root or _cfg_optional_path(cfg, "create_root")
    archive_root = args.archive_root or _cfg_path(cfg, "archive_root", Path(DEFAULT_ARCHIVE_ROOT))
    stages_spec = args.stages or _cfg_csv_or_list(cfg, "stages")
    projects_spec = args.projects or _cfg_csv_or_list(cfg, "projects")
    tick_s = args.tick if args.tick is not None else _cfg_float(cfg, "tick_s", DEFAULT_TICK_S)
    settings = FactorySettings(
        segment_gpu_procs=args.segment_gpu_procs
        if args.segment_gpu_procs is not None else _cfg_int(cfg, "segment_gpu_procs", 3),
        segment_cpu_procs=args.segment_cpu_procs
        if args.segment_cpu_procs is not None else _cfg_int(cfg, "segment_cpu_procs", 10),
        pending_hwm=args.pending_hwm
        if args.pending_hwm is not None else _cfg_int(cfg, "pending_hwm", 50_000),
        min_free_gb=args.min_free_gb
        if args.min_free_gb is not None else _cfg_float(cfg, "min_free_gb", 50.0),
        scribe_budget=args.budget
        if args.budget is not None else _cfg_optional_int(cfg, "scribe_budget", None),
    )
    stages = resolve_stages(stages_spec)  # raises on an unknown stage name
    names = [n.strip() for n in projects_spec.split(",") if n.strip()] if projects_spec else None
    projects = discover_projects(
        names,
        repo_root=repo_root,
        clips_root=clips_root,
        create_root=create_root,
        archive_root=archive_root,
    )
    log_path = args.log_file or _cfg_optional_path(cfg, "log_file")
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
        f"settings={settings} once={args.once} dry_run={args.dry_run} tick={tick_s:g}s"
    )
    supervisor = Supervisor(
        projects=projects, log=log, dry_run=args.dry_run, stages=stages,
        settings=settings, repo_root=repo_root,
    )
    if args.once:
        launched = supervisor.tick()
        log(f"tick done: launched={launched}")
        return 0
    supervisor.run(tick_s=tick_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
