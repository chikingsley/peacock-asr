"""Supervisor reconcile logic (v0): predicate gating, lock adoption, the global segment cap.

These run in ``dry_run`` so a tick launches nothing — it only computes which ``lang:stage`` keys it
WOULD launch — which is exactly the headless ``--once --dry-run`` contract.
"""

from __future__ import annotations

from omni_curator.create.queue import QueueStore, QVideo
from omni_curator.factory import flock
from omni_curator.factory.supervisor import (
    GLOBAL_SEGMENT_CAP,
    Project,
    Supervisor,
    discover_projects,
    stage_command,
)


def _project(tmp_path, name):
    """A Project whose data/create dirs live under tmp_path (real queue, real lock files)."""
    root = tmp_path / name
    data = root / "data"
    create = root / "create"
    data.mkdir(parents=True)
    create.mkdir(parents=True)
    return Project(name=name, path=root, clips_root=root / "clips", create_root=create)


def _seed_pending_video(project, video_id="chan_v1"):
    q = QueueStore(project.queue_path)
    q.enqueue_videos([QVideo(video_id, "chan", "/v1.flac", "noisy", None)])
    q.close()


def _seed_new_flac(project, channel="chan", stem="v1"):
    (project.create_root / channel).mkdir(parents=True, exist_ok=True)
    (project.create_root / channel / f"{stem}.flac").touch()


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


def test_enqueue_command_passes_ssd_create_root(tmp_path):
    p = _project(tmp_path, "dari")
    cmd = stage_command(p, "enqueue")
    assert cmd[:5] == ["uv", "run", "--project", str(p.path), "dari-curate"]
    assert cmd[5] == "enqueue"
    assert "--create-root" in cmd
    assert cmd[cmd.index("--create-root") + 1] == str(p.create_root)


def test_segment_command_passes_ssd_clips_root_and_knobs(tmp_path):
    p = _project(tmp_path, "dari")
    cmd = stage_command(p, "segment")
    assert cmd[5] == "segment"
    assert cmd[cmd.index("--clips-root") + 1] == str(p.clips_root)
    assert cmd[cmd.index("--gpu-procs") + 1] == "3"
    assert cmd[cmd.index("--cpu-procs") + 1] == "10"
    assert cmd[cmd.index("--hwm") + 1] == "5000000"


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
