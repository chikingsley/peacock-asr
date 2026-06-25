"""The polling supervisor (factory_plan §2) — enqueue, segment, labelq, harvest, archive + balancer.

One tick loop. Each ``TICK`` seconds, for every registered project it evaluates each stage's
"claimable now" predicate and, for each eligible stage whose ``flock`` is free, launches that stage
as a subprocess of the EXISTING curate CLI (``uv run --project <proj> <lang>-curate <stage> ...``).
All create-pipeline work is steered onto the fast SSD via ``--create-root`` (enqueue) and
``--clips-root`` (segment).

**The flock is held by the spawned stage, not the factory.** The curate CLI does no locking of its
own, so the supervisor wraps every launch with the ``flock(1)`` utility::

    flock -n <project>/data/.lock.<stage>  uv run --project <proj> <lang>-curate <stage> ...

``flock -n`` takes the per-(project, stage) lock non-blocking, runs the command holding the lock,
and releases on exit/crash/``kill -9``. The supervisor's try-acquire probe (``stage_is_running``)
then correctly reads the lock as held, the handshake succeeds *before* the heavy nemo/torch import,
and the global segment cap holds. (Before this, the spawned segment never locked anything, the probe
always read "free", and three segments started at once — the cap was a no-op.)

Liveness is the ``flock``, not the ``Popen`` handle: a stage that dies frees its lock, so the next
tick sees the predicate still true and the lock free and relaunches it.

**Stage kinds.** *daemon* stages (segment, labelq) self-drain over a long run; the supervisor awaits
their lock after spawn (the launch handshake) and counts them as running. *one-shot* stages
(enqueue, harvest, archive) run and exit fast; they are still ``flock``-wrapped (so a slow one isn't
double-launched), but the supervisor does NOT await their lock — it gates them purely on the
predicate, launches, and moves on (awaiting a lock a fast exit already released would always
"fail").

**Global segment cap:** one segment job saturates the disk-read path, two contend — so a GLOBAL cap
of 1 concurrent segment across ALL projects (``GLOBAL_SEGMENT_CAP``), counted from the live segment
``flock``s (so a manual segment counts too).

**Scribe balancer (factory_plan §5):** each tick, one ``--budget`` is split across all live labelq +
verify jobs by writing their window files, so total concurrent Scribe calls stay within capacity.

NOT yet: merge, verify-as-a-stage, overflow backpressure / hard-halt.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from omni_curator.factory import flock, predicates
from omni_curator.scribe.balance import active_scribe_jobs, apply_budget

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

#: One segment across ALL projects: a single segment job saturates the disk-read path.
GLOBAL_SEGMENT_CAP = 1

#: Default factory tick.
DEFAULT_TICK_S = 30.0

#: Default SSD roots; ``<lang>`` is filled per project (the disk moved off spinning overflow).
DEFAULT_CLIPS_ROOT = "/mnt/fast-ssd-2tb/peacock-clips"
DEFAULT_CREATE_ROOT = "/mnt/fast-ssd-2tb/peacock-create"

#: Default archive destination — sources drain here off the working drive (factory_plan §3).
DEFAULT_ARCHIVE_ROOT = "/mnt/storage/peacock-archive"

#: Default repo root holding ``projects/<lang>-asr``.
REPO_ROOT = Path("/home/simon/github/peacock-asr")

#: After spawning a DAEMON, how long to wait for it to acquire its lock before declaring the launch
#: failed (and freeing its cap slot for a retry next tick). Segment's import path (nemo/torch) is
#: heavy, so this is generous. One-shots skip the handshake entirely.
LAUNCH_HANDSHAKE_TIMEOUT_S = 120.0
_HANDSHAKE_POLL_S = 0.5

#: All stages the factory can drive, in tick evaluation order (upstream first, so a stage's input
#: appears within the same tick). ``export``/``verify``/``merge`` stay out of v1.
ALL_STAGES = ("enqueue", "segment", "labelq", "harvest", "archive")


@dataclass(frozen=True, kw_only=True)
class Project:
    """A registered language project the factory drives."""

    name: str  # short lang name, e.g. "dari" (the curate-script prefix)
    path: Path  # the project dir, e.g. /home/simon/github/peacock-asr/projects/dari-asr
    clips_root: Path  # SSD clips root for THIS lang (segment --clips-root)
    create_root: Path  # SSD create root for THIS lang (enqueue --create-root)
    archive_root: Path  # archive destination ROOT (archive --archive-root; <name>/ appended by CLI)

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
    archive_root: Path = Path(DEFAULT_ARCHIVE_ROOT),
) -> list[Project]:
    """Build the registry from ``projects/<lang>-asr`` dirs (or the explicit ``names``).

    Each project's SSD roots are ``<clips_root>/<lang>`` and ``<create_root>/<lang>``; archive uses
    a shared ``archive_root`` (the curate CLI appends ``<lang>/`` itself).
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
                archive_root=archive_root,
            )
        )
    return projects


# -- stage table: kind (daemon/one-shot), predicate, and curate argv per stage -----------------

#: Stages whose launch the supervisor must AWAIT (long-running; the handshake confirms the lock is
#: held before counting them up). Everything else is a one-shot: launch, don't await, gate on the
#: predicate only.
DAEMON_STAGES = frozenset({"segment", "labelq"})


def stage_argv(project: Project, stage: str) -> list[str]:
    """The curate argv (WITHOUT the ``flock`` wrapper) to run ``stage`` for ``project``."""
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
    if stage == "labelq":
        return base  # window file (data/.scribe_window.labelq) is the balancer's knob; defaults OK
    if stage == "harvest":
        return base
    if stage == "archive":
        return [*base, "--archive-root", str(project.archive_root)]
    msg = f"unknown stage: {stage}"
    raise ValueError(msg)


def stage_command(project: Project, stage: str) -> list[str]:
    """The full launch argv: ``flock -n <lock> <curate argv>`` so the spawned stage HOLDS the lock.

    ``flock(1)`` acquires the per-(project, stage) lock non-blocking, runs the command holding it,
    and drops it on exit/crash — which is what makes the supervisor's probe + the segment cap real.
    """
    lock = flock.lock_path(project.data_dir, stage)
    return ["flock", "-n", str(lock), *stage_argv(project, stage)]


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
    if stage == "labelq":
        return predicates.labelq_needed(project.queue_path)
    if stage == "harvest":
        return predicates.harvest_needed(project.queue_path)
    if stage == "archive":
        return predicates.archive_needed(project.queue_path)
    msg = f"unknown stage: {stage}"
    raise ValueError(msg)


@dataclass
class Supervisor:
    """The reconcile engine. Holds the registry + live child handles; ``tick`` is one pass."""

    projects: list[Project]
    log: Callable[[str], None] = print
    dry_run: bool = False
    stages: tuple[str, ...] = ALL_STAGES  # which stages this supervisor drives (--stages filter)
    budget: int | None = None  # Scribe concurrency budget to split each tick (None disables)
    repo_root: Path = REPO_ROOT  # for the balancer's window-file paths
    #: live child handles keyed by ``f"{lang}:{stage}"`` (control/reaping; flock is liveness truth).
    children: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)

    def _spawn(self, cmd: list[str]) -> subprocess.Popen[bytes]:
        # Own process group so the supervisor can signal/reap the stage and its workers as a unit.
        return subprocess.Popen(cmd, start_new_session=True)  # noqa: S603 — fixed flock+curate argv

    def _launch(self, project: Project, stage: str) -> bool:
        """Spawn ``stage`` for ``project`` (flock-wrapped).

        DAEMON stages complete a launch handshake: return ``True`` once the child holds its lock,
        ``False`` if it died before acquiring (retry next tick). ONE-SHOT stages launch and return
        ``True`` without awaiting a lock — they exit fast and re-gate on the predicate next tick. In
        ``dry_run`` the command is logged and nothing is spawned (so ``--once`` is testable).
        """
        cmd = stage_command(project, stage)
        key = f"{project.name}:{stage}"
        if self.dry_run:
            self.log(f"DRY-RUN would launch {key}: {' '.join(cmd)}")
            return True
        proc = self._spawn(cmd)
        self.log(f"LAUNCH {key} pid={proc.pid}: {' '.join(cmd)}")
        if stage not in DAEMON_STAGES:
            self.children[key] = proc  # one-shot: no handshake, gate on predicate
            return True
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

    def _balance_scribe(self) -> None:
        """Split ``budget`` across live labelq + verify jobs (factory_plan §5). No-op if unset."""
        if self.budget is None or self.dry_run:
            return
        jobs = active_scribe_jobs()
        if not jobs:
            return
        assignment = apply_budget(self.budget, jobs, root=self.repo_root)
        self.log(f"BALANCE budget={self.budget} -> {assignment}")

    def tick(self) -> list[str]:
        """One reconcile pass. Returns the ``lang:stage`` keys launched this tick.

        For each project, in stage order: skip if its lock is held (a live owner — adopt, don't
        duplicate); else if its predicate holds, launch it — segment additionally gated by the
        GLOBAL cap counted from the live segment locks. Then rebalance the Scribe budget.
        """
        self._reap()
        launched: list[str] = []
        segments_live = segment_running_count(self.projects)
        for project in self.projects:
            for stage in self.stages:
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
        self._balance_scribe()
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


def resolve_stages(spec: str | None) -> tuple[str, ...]:
    """Parse a ``--stages`` value into an ordered, validated stage tuple (``None``/all = all)."""
    if spec is None or spec.strip().lower() == "all":
        return ALL_STAGES
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    unknown = wanted - set(ALL_STAGES)
    if unknown:
        msg = f"unknown stage(s): {sorted(unknown)}; valid: {list(ALL_STAGES)}"
        raise ValueError(msg)
    return tuple(s for s in ALL_STAGES if s in wanted)  # preserve canonical order


def have_flock_binary() -> bool:
    """``True`` if the ``flock(1)`` utility is on PATH (the launch wrapper depends on it)."""
    return shutil.which("flock") is not None
