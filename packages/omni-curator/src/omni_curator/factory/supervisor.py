"""The polling supervisor — enqueue, segment, labelq, harvest, archive + balancer.

One tick loop. Each ``TICK`` seconds, for every registered project it evaluates each stage's
"claimable now" predicate and, for each eligible stage whose ``flock`` is free, launches that stage
as a subprocess of the EXISTING curate CLI (``uv run --project <proj> <lang>-curate <stage> ...``).
All create-pipeline work is steered through configured hot roots via ``--create-root`` (enqueue)
and ``--clips-root`` (segment).

**The flock is held by the spawned stage, not the factory.** The curate CLI does no locking of its
own, so the supervisor wraps every launch with the ``flock(1)`` utility::

    flock -n <project>/data/.lock.<stage>  uv run --project <proj> <lang>-curate <stage> ...

``flock -n`` takes the per-(project, stage) lock non-blocking, runs the command holding the lock,
and releases on exit/crash/``kill -9``. The supervisor's try-acquire probe (``stage_is_running``)
then correctly reads the lock as held, and the handshake succeeds *before* the heavy nemo/torch
import. (Before this, the spawned segment never locked anything, and the probe always read "free".)

Liveness is the ``flock``, not the ``Popen`` handle: a stage that dies frees its lock, so the next
tick sees the predicate still true and the lock free and relaunches it.

**Stage kinds.** *daemon* stages (segment, labelq) self-drain over a long run; the supervisor awaits
their lock after spawn (the launch handshake) and counts them as running. *one-shot* stages
(enqueue, harvest, archive) run and exit fast; they are still ``flock``-wrapped (so a slow one isn't
double-launched), but the supervisor does NOT await their lock — it gates them purely on the
predicate, launches, and moves on (awaiting a lock a fast exit already released would always
"fail").

**Segment launch policy:** segment is launched per project when its own resource predicates pass:
claimable queue backlog, pending-clip high-watermark, clips-root free space, and free stage lock.
There is no cross-project global segment cap.

**Scribe balancer:** each tick, one ``--budget`` is split across all live labelq +
verify jobs by writing their window files, so total concurrent Scribe calls stay within capacity.

NOT yet: merge, verify-as-a-stage.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from omni_curator.create.segment import DEFAULT_MIN_FREE_GB, DEFAULT_PENDING_HWM
from omni_curator.factory import flock, predicates
from omni_curator.scribe.balance import active_scribe_jobs, apply_budget
from omni_curator.scribe.concurrency import split_budget

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

#: Default factory tick.
DEFAULT_TICK_S = 30.0

#: Clips may be steered to a fast scratch root; downloads default to each project's
#: ``data/create`` unless an operator explicitly passes a create root.
DEFAULT_CLIPS_ROOT = "/mnt/workerssd-2t/peacock-clips"

#: Default archive destination — sources drain here off the working drive.
DEFAULT_ARCHIVE_ROOT = "/mnt/massive-22t/peacock-asr-archive"

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
class FactorySettings:
    """Factory knobs used by stage predicates and launched curate commands."""

    segment_gpu_procs: int = 3
    segment_cpu_procs: int = 10
    pending_hwm: int = DEFAULT_PENDING_HWM
    min_free_gb: float = DEFAULT_MIN_FREE_GB
    scribe_budget: int | None = None


DEFAULT_SETTINGS = FactorySettings()


@dataclass(frozen=True, kw_only=True)
class Project:
    """A registered language project the factory drives."""

    name: str  # short lang name, e.g. "dari" (the curate-script prefix)
    path: Path  # the project dir, e.g. /home/simon/github/peacock-asr/projects/dari-asr
    clips_root: Path  # SSD clips root for THIS lang (segment --clips-root)
    create_root: Path  # source-audio root for THIS lang (enqueue --create-root)
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
    create_root: Path | None = None,
    archive_root: Path = Path(DEFAULT_ARCHIVE_ROOT),
) -> list[Project]:
    """Build the registry from ``projects/<lang>-asr`` dirs (or the explicit ``names``).

    Each project's clips root is ``<clips_root>/<lang>``. The create root defaults to the project's
    own ``data/create``; if an operator passes a scratch ``create_root``, it becomes
    ``<create_root>/<lang>``. Archive uses a shared ``archive_root`` (the curate CLI appends
    ``<lang>/`` itself).
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
        default_create = path / "data" / "create"
        project_create_root = create_root / lang if create_root is not None else default_create
        projects.append(
            Project(
                name=lang,
                path=path,
                clips_root=clips_root / lang,
                create_root=project_create_root,
                archive_root=archive_root,
            )
        )
    return projects


# -- stage table: kind (daemon/one-shot), predicate, and curate argv per stage -----------------

#: Stages whose launch the supervisor must AWAIT (long-running; the handshake confirms the lock is
#: held before counting them up). Everything else is a one-shot: launch, don't await, gate on the
#: predicate only.
DAEMON_STAGES = frozenset({"segment", "labelq"})


def _free_gb(path: Path) -> float:
    """Free GB on the filesystem holding ``path`` or its nearest existing parent."""
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1_000_000_000


def _segment_blockers(project: Project, settings: FactorySettings) -> list[str]:
    blockers: list[str] = []
    backlog = predicates.segment_backlog(project.queue_path)
    if backlog <= 0:
        blockers.append("no pending/stale videos")
    pending = predicates.pending_clip_count(project.queue_path)
    if pending >= settings.pending_hwm:
        blockers.append(f"pending clips {pending} >= HWM {settings.pending_hwm}")
    free_gb = _free_gb(project.clips_root)
    if free_gb < settings.min_free_gb:
        blockers.append(f"clips free {free_gb:.1f}GB < min {settings.min_free_gb:.1f}GB")
    return blockers


def _predicate_blockers(*, ready: bool, reason: str) -> list[str]:
    return [] if ready else [reason]


def stage_argv(
    project: Project, stage: str, settings: FactorySettings = DEFAULT_SETTINGS
) -> list[str]:
    """The curate argv (WITHOUT the ``flock`` wrapper) to run ``stage`` for ``project``."""
    base = ["uv", "run", "--project", str(project.path), project.curate, stage]
    if stage == "enqueue":
        return [*base, "--create-root", str(project.create_root)]
    if stage == "segment":
        return [
            *base,
            "--clips-root", str(project.clips_root),
            "--gpu-procs", str(settings.segment_gpu_procs),
            "--cpu-procs", str(settings.segment_cpu_procs),
            "--hwm", str(settings.pending_hwm),
            "--min-free-gb", str(settings.min_free_gb),
        ]
    if stage == "labelq":
        return base  # window file (data/.scribe_window.labelq) is the balancer's knob; defaults OK
    if stage == "harvest":
        return base
    if stage == "archive":
        return [*base, "--archive-root", str(project.archive_root)]
    msg = f"unknown stage: {stage}"
    raise ValueError(msg)


def stage_command(
    project: Project, stage: str, settings: FactorySettings = DEFAULT_SETTINGS
) -> list[str]:
    """The full launch argv: ``flock -n <lock> <curate argv>`` so the spawned stage HOLDS the lock.

    ``flock(1)`` acquires the per-(project, stage) lock non-blocking, runs the command holding it,
    and drops it on exit/crash, which is what makes the supervisor's probe real.
    """
    lock = flock.lock_path(project.data_dir, stage)
    return ["flock", "-n", str(lock), *stage_argv(project, stage, settings)]


def stage_is_running(project: Project, stage: str) -> bool:
    """``True`` if a live owner holds ``project``'s ``stage`` lock (factory or manual)."""
    return flock.is_locked(flock.lock_path(project.data_dir, stage))


def segment_running_count(projects: Iterable[Project]) -> int:
    """How many projects currently have a live ``segment`` owner."""
    return sum(1 for p in projects if stage_is_running(p, "segment"))


def stage_blockers(
    project: Project,
    stage: str,
    settings: FactorySettings = DEFAULT_SETTINGS,
) -> list[str]:
    """Reasons ``stage`` should not launch for ``project`` right now."""
    if stage_is_running(project, stage):
        return [f"{stage} lock held"]
    if stage == "enqueue":
        return _predicate_blockers(
            ready=predicates.enqueue_needed(project.queue_path, project.create_root),
            reason="no new source FLACs",
        )
    if stage == "segment":
        return _segment_blockers(project, settings)
    if stage == "labelq":
        return _predicate_blockers(
            ready=predicates.labelq_needed(project.queue_path),
            reason="no claimable clips",
        )
    if stage == "harvest":
        return _predicate_blockers(
            ready=predicates.harvest_needed(project.queue_path),
            reason="no done unharvested clips",
        )
    if stage == "archive":
        return _predicate_blockers(
            ready=predicates.archive_needed(project.queue_path),
            reason="no segmented sources still on working disk",
        )
    msg = f"unknown stage: {stage}"
    raise ValueError(msg)


def stage_eligible(
    project: Project, stage: str, settings: FactorySettings = DEFAULT_SETTINGS
) -> bool:
    """Does ``stage``'s predicate hold for ``project`` right now? (lock-state included)."""
    return not stage_blockers(project, stage, settings)


@dataclass
class Supervisor:
    """The reconcile engine. Holds the registry + live child handles; ``tick`` is one pass."""

    projects: list[Project]
    log: Callable[[str], None] = print
    dry_run: bool = False
    stages: tuple[str, ...] = ALL_STAGES  # which stages this supervisor drives (--stages filter)
    settings: FactorySettings = DEFAULT_SETTINGS
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
        cmd = stage_command(project, stage, self.settings)
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
        """Split ``budget`` across live labelq + verify jobs. No-op if unset."""
        if self.settings.scribe_budget is None:
            return
        jobs = active_scribe_jobs()
        if not jobs:
            if self.dry_run:
                self.log("BALANCE dry-run: no active scribe jobs")
            return
        if self.dry_run:
            assignment = split_budget(self.settings.scribe_budget, jobs)
            self.log(f"BALANCE dry-run budget={self.settings.scribe_budget} -> {assignment}")
            return
        assignment = apply_budget(self.settings.scribe_budget, jobs, root=self.repo_root)
        self.log(f"BALANCE budget={self.settings.scribe_budget} -> {assignment}")

    def tick(self) -> list[str]:
        """One reconcile pass. Returns the ``lang:stage`` keys launched this tick.

        For each project, in stage order: skip if its lock is held (a live owner — adopt, don't
        duplicate); else if its predicate holds, launch it. Then rebalance the Scribe budget.
        """
        self._reap()
        launched: list[str] = []
        for project in self.projects:
            for stage in self.stages:
                key = f"{project.name}:{stage}"
                blockers = stage_blockers(project, stage, self.settings)
                if blockers:
                    if self.dry_run:
                        self.log(f"HOLD {key}: {'; '.join(blockers)}")
                    continue
                if self._launch(project, stage):
                    launched.append(key)
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
