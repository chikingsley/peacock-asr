"""The polling supervisor (factory_plan §2) — v0: enqueue + segment only.

One tick loop. Each ``TICK`` seconds, for every registered project it evaluates the v0 predicates
and, for each eligible stage whose ``flock`` is free, launches that stage as a subprocess of the
EXISTING curate CLI (``uv run --project <proj> <lang>-curate <stage> ...``). All work is steered
onto the fast SSD via ``--create-root`` (enqueue) and ``--clips-root`` (segment).

Liveness is the ``flock`` (factory_plan §1), not the ``Popen`` handle: a stage that dies frees its
lock, so the next tick sees the predicate still true and the lock free and relaunches it. The handle
is kept only to reap zombies and as the launch-handshake target.

**Global segment cap:** one segment job saturates the disk-read path, two contend. So v0 enforces a
GLOBAL cap of 1 concurrent segment across ALL projects (``GLOBAL_SEGMENT_CAP``). The cap is counted
from the live segment ``flock``s themselves (the source of truth), so a manually-started segment
also counts against it. enqueue is cheap and runs freely.

NOT in v0: labelq, verify, harvest, merge, archive, the Scribe balancer, overflow backpressure /
hard-halt. Those are v1.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from omni_curator.factory import flock, predicates

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

#: v0 stages, in evaluation order. enqueue first (cheap, seeds segment's input within the tick).
V0_STAGES = ("enqueue", "segment")

#: One segment across ALL projects: a single segment job saturates the disk-read path.
GLOBAL_SEGMENT_CAP = 1

#: Default factory tick.
DEFAULT_TICK_S = 30.0

#: Default SSD roots; ``<lang>`` is filled per project (the disk moved off spinning overflow).
DEFAULT_CLIPS_ROOT = "/mnt/fast-ssd-2tb/peacock-clips"
DEFAULT_CREATE_ROOT = "/mnt/fast-ssd-2tb/peacock-create"

#: Default repo root holding ``projects/<lang>-asr``.
REPO_ROOT = Path("/home/simon/github/peacock-asr")

#: After spawning a child, how long to wait for it to acquire its lock before declaring the launch
#: failed (and freeing its cap slot for a retry next tick). Segment's import path (nemo/torch) is
#: heavy, so this is generous.
LAUNCH_HANDSHAKE_TIMEOUT_S = 120.0
_HANDSHAKE_POLL_S = 0.5


@dataclass(frozen=True, kw_only=True)
class Project:
    """A registered language project the factory drives."""

    name: str  # short lang name, e.g. "dari" (the curate-script prefix)
    path: Path  # the project dir, e.g. /home/simon/github/peacock-asr/projects/dari-asr
    clips_root: Path  # SSD clips root for THIS lang (segment --clips-root)
    create_root: Path  # SSD create root for THIS lang (enqueue --create-root)

    @property
    def data_dir(self) -> Path:
        """The project's data dir (a symlink to overflow; locks/queue live under it)."""
        return self.path / "data"

    @property
    def queue_path(self) -> Path:
        return self.data_dir / "queue.sqlite"

    @property
    def curate(self) -> str:
        """The curate console-script name (``<lang>-curate``)."""
        return f"{self.name}-curate"


def discover_projects(
    names: Iterable[str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    clips_root: Path = Path(DEFAULT_CLIPS_ROOT),
    create_root: Path = Path(DEFAULT_CREATE_ROOT),
) -> list[Project]:
    """Build the registry from ``projects/<lang>-asr`` dirs (or the explicit ``names``).

    Each project's SSD roots are ``<clips_root>/<lang>`` and ``<create_root>/<lang>``.
    """
    if names is None:
        found = sorted(
            p.name.removesuffix("-asr")
            for p in (repo_root / "projects").glob("*-asr")
            if (p / "data").exists()
        )
    else:
        found = list(names)
    projects: list[Project] = []
    for lang in found:
        path = repo_root / "projects" / f"{lang}-asr"
        projects.append(
            Project(
                name=lang,
                path=path,
                clips_root=clips_root / lang,
                create_root=create_root / lang,
            )
        )
    return projects


def stage_command(project: Project, stage: str) -> list[str]:
    """The exact argv to launch ``stage`` for ``project`` via the existing curate CLI."""
    base = ["uv", "run", "--project", str(project.path), project.curate, stage]
    if stage == "enqueue":
        return [*base, "--create-root", str(project.create_root)]
    if stage == "segment":
        return [
            *base,
            "--clips-root", str(project.clips_root),
            "--gpu-procs", "3",
            "--cpu-procs", "10",
            "--hwm", "5000000",
        ]
    msg = f"unknown v0 stage: {stage}"
    raise ValueError(msg)


def stage_is_running(project: Project, stage: str) -> bool:
    """``True`` if a live owner holds ``project``'s ``stage`` lock (factory or manual)."""
    return flock.is_locked(flock.lock_path(project.data_dir, stage))


def segment_running_count(projects: Iterable[Project]) -> int:
    """How many projects currently have a live ``segment`` owner (the global-cap counter)."""
    return sum(1 for p in projects if stage_is_running(p, "segment"))


def stage_eligible(project: Project, stage: str) -> bool:
    """Does ``stage``'s predicate hold for ``project`` right now? (lock-state aside)."""
    if stage == "enqueue":
        return predicates.enqueue_needed(project.queue_path, project.create_root)
    if stage == "segment":
        return predicates.segment_needed(project.queue_path)
    msg = f"unknown v0 stage: {stage}"
    raise ValueError(msg)


@dataclass
class Supervisor:
    """The reconcile engine. Holds the registry + live child handles; ``tick`` is one pass."""

    projects: list[Project]
    log: Callable[[str], None] = print
    dry_run: bool = False
    #: live child handles keyed by ``f"{lang}:{stage}"`` (control/reaping; flock is liveness truth).
    children: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)

    def _launch(self, project: Project, stage: str) -> bool:
        """Spawn ``stage`` for ``project`` and complete the launch handshake.

        Returns ``True`` once the child has acquired its lock (counts as running), ``False`` if it
        died before acquiring (a retry happens next tick). In ``dry_run`` the command is logged and
        nothing is spawned (so ``--once`` is testable headlessly).
        """
        cmd = stage_command(project, stage)
        key = f"{project.name}:{stage}"
        if self.dry_run:
            self.log(f"DRY-RUN would launch {key}: {' '.join(cmd)}")
            return True
        # Own process group so the supervisor can signal/reap the stage and its workers as a unit.
        proc = subprocess.Popen(cmd, start_new_session=True)  # noqa: S603 — fixed curate argv
        self.log(f"LAUNCH {key} pid={proc.pid}: {' '.join(cmd)}")
        if self._await_lock(project, stage, proc):
            self.children[key] = proc
            return True
        self.log(f"LAUNCH-FAILED {key}: child exited before acquiring lock (will retry)")
        return False

    def _await_lock(
        self, project: Project, stage: str, proc: subprocess.Popen[bytes]
    ) -> bool:
        """Poll until the child holds its lock (success) or dies first (failure), with a timeout."""
        lock = flock.lock_path(project.data_dir, stage)
        deadline = time.monotonic() + LAUNCH_HANDSHAKE_TIMEOUT_S
        while time.monotonic() < deadline:
            if flock.is_locked(lock):
                return True
            if proc.poll() is not None:  # exited before acquiring
                return False
            time.sleep(_HANDSHAKE_POLL_S)
        return flock.is_locked(lock)

    def _reap(self) -> None:
        """Drop finished child handles (the flock already freed; this just clears the table)."""
        for key, proc in list(self.children.items()):
            if proc.poll() is not None:
                self.log(f"EXIT {key} rc={proc.returncode}")
                del self.children[key]

    def tick(self) -> list[str]:
        """One reconcile pass. Returns the ``lang:stage`` keys launched this tick.

        For each project, in stage order: skip if its lock is held (a live owner — adopt, don't
        duplicate); else if its predicate holds, launch it — segment additionally gated by the
        GLOBAL cap counted from the live segment locks.
        """
        self._reap()
        launched: list[str] = []
        # Count live segments once up front, then track launches within the tick against the cap.
        segments_live = segment_running_count(self.projects)
        for project in self.projects:
            for stage in V0_STAGES:
                key = f"{project.name}:{stage}"
                if stage_is_running(project, stage):
                    continue
                if not stage_eligible(project, stage):
                    continue
                if stage == "segment" and segments_live >= GLOBAL_SEGMENT_CAP:
                    self.log(
                        f"HOLD {key}: global segment cap {GLOBAL_SEGMENT_CAP} reached "
                        f"({segments_live} live)"
                    )
                    continue
                if self._launch(project, stage):
                    launched.append(key)
                    if stage == "segment":
                        segments_live += 1
        return launched

    def run(self, *, tick_s: float = DEFAULT_TICK_S, max_ticks: int | None = None) -> None:
        """The daemon loop: ``tick`` then sleep, forever (or ``max_ticks`` passes, for tests)."""
        n = 0
        while max_ticks is None or n < max_ticks:
            self.tick()
            n += 1
            if max_ticks is not None and n >= max_ticks:
                break
            time.sleep(tick_s)
