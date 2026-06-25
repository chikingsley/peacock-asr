"""Supervisor reconcile logic (v0): predicate gating, lock adoption, the global segment cap.

These run in ``dry_run`` so a tick launches nothing — it only computes which ``lang:stage`` keys it
WOULD launch — which is exactly the headless ``--once --dry-run`` contract.
"""

from __future__ import annotations

import time

from omni_curator.create.queue import QueueStore, QVideo
from omni_curator.factory import flock
from omni_curator.factory.supervisor import (
    GLOBAL_SEGMENT_CAP,
    Project,
    Supervisor,
    discover_projects,
    resolve_stages,
    stage_argv,
    stage_command,
)


def _project(tmp_path, name):
    """A Project whose data/create dirs live under tmp_path (real queue, real lock files)."""
    root = tmp_path / name
    data = root / "data"
    create = root / "create"
    data.mkdir(parents=True)
    create.mkdir(parents=True)
    return Project(
        name=name, path=root, clips_root=root / "clips", create_root=create,
        archive_root=tmp_path / "archive",
    )


def _seed_pending_video(project, video_id="chan_v1"):
    q = QueueStore(project.queue_path)
    q.enqueue_videos([QVideo(video_id, "chan", "/v1.flac", "noisy", None)])
    q.close()


def _seed_new_flac(project, channel="chan", stem="v1"):
    (project.create_root / channel).mkdir(parents=True, exist_ok=True)
    (project.create_root / channel / f"{stem}.flac").touch()


def _one_clip(video_id):
    from omni_curator.create.queue import QClip

    return [
        QClip(f"{video_id}_0000", video_id, "chan", 0, f"/clips/{video_id}/0.flac",
              0.0, 5.0, "tgk_Cyrl", "Cyrillic", None)
    ]


def _seed_done_clip(project, video_id="chan_v1"):
    """Drive a clip all the way to status='done' (harvestable, no longer pending)."""
    q = QueueStore(project.queue_path)
    q.enqueue_videos([QVideo(video_id, "chan", "/v1.flac", "noisy", None)])
    claimed = q.claim_video("w1")
    assert claimed is not None
    q.complete_video(video_id, _one_clip(video_id), claim_token=claimed.claim_token)
    token = "labeltok"
    q.claim_clips(10, token)
    q.complete_clips(token, [(f"{video_id}_0000", "label text", "")])
    q.close()


def _supervisor(projects):
    return Supervisor(projects=list(projects), log=lambda _m: None, dry_run=True)


# -- predicate gating --------------------------------------------------------------------------


def test_idle_project_launches_nothing(tmp_path):
    p = _project(tmp_path, "dari")
    assert _supervisor([p]).tick() == []


def test_new_flac_triggers_enqueue(tmp_path):
    p = _project(tmp_path, "dari")
    _seed_new_flac(p)
    assert _supervisor([p]).tick() == ["dari:enqueue"]


def test_pending_video_triggers_segment(tmp_path):
    p = _project(tmp_path, "dari")
    _seed_pending_video(p)
    assert _supervisor([p]).tick() == ["dari:segment"]


def test_both_stages_launch_when_both_eligible(tmp_path):
    p = _project(tmp_path, "dari")
    _seed_new_flac(p, stem="v2")  # a new FLAC absent from the queue -> enqueue
    _seed_pending_video(p, video_id="chan_v1")  # a pending video already queued -> segment
    assert _supervisor([p]).tick() == ["dari:enqueue", "dari:segment"]


# -- lock adoption (don't duplicate a live owner) ----------------------------------------------


def test_held_segment_lock_is_adopted_not_relaunched(tmp_path):
    p = _project(tmp_path, "dari")
    _seed_pending_video(p)
    with flock.hold(flock.lock_path(p.data_dir, "segment")):
        assert _supervisor([p]).tick() == []  # a live owner exists -> adopt, don't duplicate


def test_held_enqueue_lock_is_adopted(tmp_path):
    p = _project(tmp_path, "dari")
    _seed_new_flac(p)
    with flock.hold(flock.lock_path(p.data_dir, "enqueue")):
        assert _supervisor([p]).tick() == []


# -- the GLOBAL segment cap (one segment across ALL projects) ----------------------------------


def test_global_segment_cap_blocks_second_project(tmp_path):
    a = _project(tmp_path, "dari")
    b = _project(tmp_path, "farsi")
    _seed_pending_video(a)
    _seed_pending_video(b)
    # Both eligible, but the global cap is 1: only the first project's segment launches this tick.
    launched = _supervisor([a, b]).tick()
    assert launched == ["dari:segment"]
    assert GLOBAL_SEGMENT_CAP == 1


def test_existing_segment_lock_consumes_the_global_slot(tmp_path):
    a = _project(tmp_path, "dari")
    b = _project(tmp_path, "farsi")
    _seed_pending_video(b)
    # a already has a live segment (the cap counts from live locks) -> b is blocked this tick.
    with flock.hold(flock.lock_path(a.data_dir, "segment")):
        assert _supervisor([a, b]).tick() == []


def test_enqueue_runs_freely_alongside_a_capped_segment(tmp_path):
    a = _project(tmp_path, "dari")
    b = _project(tmp_path, "farsi")
    _seed_pending_video(a)  # a wants segment (takes the one slot)
    _seed_new_flac(b, stem="v2")  # b wants enqueue (cheap, uncapped); distinct from its queued vid
    _seed_pending_video(b, video_id="chan_v1")  # b also wants segment (blocked by the cap)
    launched = _supervisor([a, b]).tick()
    assert launched == ["dari:segment", "farsi:enqueue"]  # b:enqueue runs; b:segment held


# -- command construction ----------------------------------------------------------------------


def test_command_is_flock_wrapped_so_the_spawned_stage_holds_the_lock(tmp_path):
    # THE FIX: every launch is `flock -n <lock> <curate ...>` so the child holds the per-stage lock
    # for its lifetime — otherwise the probe always reads "free" and the segment cap is a no-op.
    p = _project(tmp_path, "dari")
    cmd = stage_command(p, "segment")
    assert cmd[:3] == ["flock", "-n", str(flock.lock_path(p.data_dir, "segment"))]
    assert cmd[3:8] == ["uv", "run", "--project", str(p.path), "dari-curate"]
    assert cmd[8] == "segment"


def test_enqueue_argv_passes_ssd_create_root(tmp_path):
    p = _project(tmp_path, "dari")
    argv = stage_argv(p, "enqueue")
    assert argv[:5] == ["uv", "run", "--project", str(p.path), "dari-curate"]
    assert argv[5] == "enqueue"
    assert argv[argv.index("--create-root") + 1] == str(p.create_root)


def test_segment_argv_passes_ssd_clips_root_and_knobs(tmp_path):
    p = _project(tmp_path, "dari")
    argv = stage_argv(p, "segment")
    assert argv[5] == "segment"
    assert argv[argv.index("--clips-root") + 1] == str(p.clips_root)
    assert argv[argv.index("--gpu-procs") + 1] == "3"
    assert argv[argv.index("--cpu-procs") + 1] == "10"
    assert argv[argv.index("--hwm") + 1] == "5000000"


def test_archive_argv_passes_archive_root(tmp_path):
    p = _project(tmp_path, "dari")
    argv = stage_argv(p, "archive")
    assert argv[5] == "archive"
    assert argv[argv.index("--archive-root") + 1] == str(p.archive_root)


# -- discovery ---------------------------------------------------------------------------------


def test_discover_explicit_names_builds_ssd_roots(tmp_path):
    projects = discover_projects(
        ["dari", "farsi"],
        repo_root=tmp_path,
        clips_root=tmp_path / "clips",
        create_root=tmp_path / "create",
    )
    assert [p.name for p in projects] == ["dari", "farsi"]
    assert projects[0].clips_root == tmp_path / "clips" / "dari"
    assert projects[0].create_root == tmp_path / "create" / "dari"
    assert projects[0].path == tmp_path / "projects" / "dari-asr"


def test_discover_scans_projects_dir(tmp_path):
    for lang in ("dari", "georgian"):
        (tmp_path / "projects" / f"{lang}-asr" / "data").mkdir(parents=True)
    (tmp_path / "projects" / "no-data-asr").mkdir(parents=True)  # no data/ -> skipped
    projects = discover_projects(None, repo_root=tmp_path)
    assert sorted(p.name for p in projects) == ["dari", "georgian"]


# -- THE FLOCK FIX: a flock(1)-held lock is seen, and the global cap blocks a 2nd segment ------


def _flock_hold(lockfile, hold_s):
    """Spawn `flock -n <lockfile> sleep <hold_s>` (mimics a spawned stage holding its lock)."""
    import subprocess

    proc = subprocess.Popen(
        ["flock", "-n", str(lockfile), "sleep", str(hold_s)], start_new_session=True
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not flock.is_locked(lockfile):
        time.sleep(0.02)
    return proc


def _kill_group(proc):
    """Kill the whole process group (flock + its `sleep` child both hold/release the lock)."""
    import os
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(5)


def test_flock_utility_held_lock_is_seen_by_the_probe(tmp_path):
    p = _project(tmp_path, "dari")
    lockfile = flock.lock_path(p.data_dir, "segment")
    assert not flock.is_locked(lockfile)
    proc = _flock_hold(lockfile, 30)
    try:
        assert flock.is_locked(lockfile)  # the probe sees the flock(1)-held lock
    finally:
        _kill_group(proc)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and flock.is_locked(lockfile):
        time.sleep(0.02)
    assert not flock.is_locked(lockfile)  # released on the holder's death


def test_global_cap_blocks_second_segment_when_flock_utility_holds_the_first(tmp_path):
    # The end-to-end cap proof: a real flock(1) holding project a's segment lock consumes the one
    # global slot, so the supervisor must NOT launch project b's segment this tick.
    a = _project(tmp_path, "dari")
    b = _project(tmp_path, "farsi")
    _seed_pending_video(b)
    proc = _flock_hold(flock.lock_path(a.data_dir, "segment"), 30)
    try:
        assert _supervisor([a, b]).tick() == []
    finally:
        _kill_group(proc)


# -- one-shots: no handshake (gate purely on the predicate) ------------------------------------


class _FakeProc:
    """A stand-in Popen so a real (non-dry-run) launch stays headless (no subprocess runs)."""

    pid = 4242
    returncode = None

    def poll(self):
        return None


def test_oneshot_launch_does_not_await_a_lock(tmp_path, monkeypatch):
    # enqueue exits immediately, so awaiting a held lock would always "fail". A real (non-dry-run)
    # launch of a one-shot must be recorded as launched without any handshake. We swap in a no-op
    # spawn so the test stays headless (no uv/flock subprocess actually runs).
    p = _project(tmp_path, "dari")
    _seed_new_flac(p)
    sup = Supervisor(projects=[p], log=lambda _m: None, dry_run=False)
    spawned = []
    monkeypatch.setattr(sup, "_spawn", lambda cmd: (spawned.append(cmd), _FakeProc())[1])
    launched = sup.tick()
    assert launched == ["dari:enqueue"]  # launched without ever awaiting a lock
    assert spawned[0][0] == "flock"  # still flock-wrapped


# -- v1 stage predicates wire through tick -----------------------------------------------------


def test_labelq_launches_on_pending_clips(tmp_path):
    p = _project(tmp_path, "dari")
    q = QueueStore(p.queue_path)
    q.enqueue_videos([QVideo("chan_v1", "chan", "/v1.flac", "noisy", None)])
    claimed = q.claim_video("w1")
    assert claimed is not None
    q.complete_video("chan_v1", _one_clip("chan_v1"), claim_token=claimed.claim_token)
    q.close()
    assert _supervisor([p]).tick() == ["dari:labelq"]


def test_harvest_launches_on_done_unharvested_clips(tmp_path):
    p = _project(tmp_path, "dari")
    _seed_done_clip(p)
    # done+unharvested -> harvest fires; no pending clips so labelq does not
    assert _supervisor([p]).tick() == ["dari:harvest"]


def test_archive_launches_when_segmented_source_still_on_disk(tmp_path):
    p = _project(tmp_path, "dari")
    src = tmp_path / "overflow" / "v1.flac"
    src.parent.mkdir(parents=True)
    src.touch()
    q = QueueStore(p.queue_path)
    q.enqueue_videos([QVideo("chan_v1", "chan", str(src), "noisy", None)])
    claimed = q.claim_video("w1")
    assert claimed is not None
    q.complete_video("chan_v1", [], claim_token=claimed.claim_token)  # status='segmented'
    q.close()
    assert _supervisor([p]).tick() == ["dari:archive"]


def test_archive_not_launched_when_source_already_drained(tmp_path):
    p = _project(tmp_path, "dari")
    q = QueueStore(p.queue_path)
    q.enqueue_videos([QVideo("chan_v1", "chan", "/gone/v1.flac", "noisy", None)])
    claimed = q.claim_video("w1")
    assert claimed is not None
    q.complete_video("chan_v1", [], claim_token=claimed.claim_token)
    q.close()
    assert _supervisor([p]).tick() == []  # source path does not exist -> nothing to archive


# -- the --stages filter -----------------------------------------------------------------------


def test_resolve_stages_default_is_all():
    from omni_curator.factory.supervisor import ALL_STAGES

    assert resolve_stages(None) == ALL_STAGES
    assert resolve_stages("all") == ALL_STAGES


def test_resolve_stages_preserves_canonical_order():
    assert resolve_stages("segment,enqueue") == ("enqueue", "segment")


def test_resolve_stages_rejects_unknown():
    import pytest

    with pytest.raises(ValueError, match="unknown stage"):
        resolve_stages("enqueue,bogus")


def test_stages_filter_runs_segment_only(tmp_path):
    p = _project(tmp_path, "dari")
    _seed_new_flac(p, stem="v2")  # would trigger enqueue, but it's filtered out
    _seed_pending_video(p, video_id="chan_v1")
    sup = Supervisor(
        projects=[p], log=lambda _m: None, dry_run=True, stages=("segment",)
    )
    assert sup.tick() == ["dari:segment"]  # enqueue not in --stages


# -- the Scribe balancer -----------------------------------------------------------------------


def test_balancer_splits_budget_across_live_jobs(tmp_path, monkeypatch):
    p = _project(tmp_path, "dari")
    logs = []
    sup = Supervisor(
        projects=[p], log=logs.append, dry_run=False, stages=("enqueue",),
        budget=300, repo_root=tmp_path,
    )
    monkeypatch.setattr(
        "omni_curator.factory.supervisor.active_scribe_jobs",
        lambda: ["dari:labelq", "farsi:verify"],
    )
    monkeypatch.setattr(sup, "_spawn", lambda _cmd: _FakeProc())  # no real children this test
    sup.tick()
    # each job's window file got half the budget (split_budget); the balancer logged it
    assert (tmp_path / "projects" / "dari-asr" / "data" / ".scribe_window.labelq").read_text(
    ).strip() == "150"
    assert (tmp_path / "projects" / "farsi-asr" / "data" / ".scribe_window.verify").read_text(
    ).strip() == "150"
    assert any("BALANCE" in m for m in logs)


def test_balancer_disabled_without_budget(tmp_path, monkeypatch):
    p = _project(tmp_path, "dari")
    called = []
    monkeypatch.setattr(
        "omni_curator.factory.supervisor.active_scribe_jobs",
        lambda: called.append(True) or [],
    )
    Supervisor(projects=[p], log=lambda _m: None, dry_run=False, budget=None).tick()
    assert called == []  # no budget -> the balancer never even probes
