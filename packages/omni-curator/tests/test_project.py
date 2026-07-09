"""CuratorProject CLI: every language gets the full, identical command set from one config."""

from __future__ import annotations

import json
import sqlite3

import pytest

from omni_curator.create.youtube import Channel, channel
from omni_curator.project import (
    CuratorProject,
    _parse_weights,
    build_parser,
    cmd_enqueue,
    cmd_harvest,
    cmd_repair_metadata,
    fleurs_source,
    huggingface_source,
)

ALL_COMMANDS = {
    "prescan", "list", "download", "cookies",
    "enqueue", "repair-metadata", "segment", "vad-pilot", "resegment", "labelq", "harvest",
    "archive",
    "merge", "ingest", "verify", "rescore", "export",
}


@pytest.fixture
def project(tmp_path):
    return CuratorProject(
        name="testlang",
        language="tgk_Cyrl",
        script="Cyrillic",
        data=tmp_path / "data",
        db=tmp_path / "data" / "curator.sqlite",
        channels=(
            channel("chan_a", "@handle_a", "clean", "test", category="news"),
            channel("chan_b", "UCxxxxxxxxxxxxxxxxxxxxxx", "noisy", "test"),
        ),
        ingests={"fleurs": fleurs_source("tg_tj")},
        mixture_weights={"fleurs": 490.0},
    )


def test_every_command_is_registered(project):
    parser = build_parser(project)
    sub = next(a for a in parser._actions if a.dest == "command")  # noqa: SLF001
    assert sub.choices is not None
    assert set(sub.choices) == ALL_COMMANDS


def test_channel_builder_expands_idents():
    assert channel("a", "@handle", "clean", "").url == "https://www.youtube.com/@handle"
    assert channel("b", "UCabc", "noisy", "").url == "https://www.youtube.com/channel/UCabc"
    full = "https://www.youtube.com/channel/UCxyz"
    assert channel("c", full, "clean", "").url == full
    assert channel("d", "@handle", "clean", "", category="lecture").category == "lecture"
    assert channel("e", "@handle", "clean", "studio news bulletins").category == "news"


def test_selected_channels_filters(project):
    parser = build_parser(project)
    args = parser.parse_args(["download", "--tier", "noisy"])
    assert [c.slug for c in project.selected_channels(args)] == ["chan_b"]
    args = parser.parse_args(["download", "--channel", "chan_a"])
    assert [c.slug for c in project.selected_channels(args)] == ["chan_a"]


def test_data_layout_is_owned_by_the_project(project):
    assert project.create_dir == project.data / "create"
    assert project.queue_path == project.data / "queue.sqlite"
    assert project.channels_by_slug["chan_a"].tier == "clean"
    assert project.channels_by_slug["chan_a"].category == "news"


def test_prescan_records_channel_decision(tmp_path):
    from omni_curator.create.youtube import prescan_channels

    ch = channel("a", "@a", "clean", "notes", category="lecture")
    seen: list[tuple[str, int | None]] = []

    def fake_lister(url: str, *, limit: int | None) -> list[str]:
        seen.append((url, limit))
        return ["vid1"]

    db = tmp_path / "prescan.sqlite"
    results = prescan_channels([ch], db_path=db, limit=1, lane="gluetun-lane1",
                               list_videos=fake_lister)

    assert seen == [("https://www.youtube.com/@a", 1)]
    assert results[0].status == "ok"
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT slug, tier, category, lane, status, video_count FROM channel_prescan"
    ).fetchone()
    conn.close()
    assert row == ("a", "clean", "lecture", "gluetun-lane1", "ok", 1)


def test_enqueue_preserves_channel_and_video_metadata(project):
    from omni_curator.create.queue import QueueStore

    source_dir = project.create_dir / "chan_a"
    source_dir.mkdir(parents=True)
    flac = source_dir / "vid1.flac"
    flac.write_bytes(b"not audio yet")
    flac.with_suffix(".info.json").write_text(
        json.dumps(
            {
                "title": "A title",
                "description": "x" * 5000,
                "upload_date": "20250102",
                "channel": "Handle A",
                "webpage_url": "https://youtu.be/vid1",
                "unneeded_blob": {"large": True},
            }
        ),
        encoding="utf-8",
    )

    args = build_parser(project).parse_args(["enqueue", "--channel", "chan_a"])
    assert cmd_enqueue(project, args) == 0

    queue = QueueStore(project.queue_path)
    video = queue.claim_video("seg-0")
    queue.close()
    assert video is not None
    assert video.category == "news"
    assert video.tier == "clean"
    assert video.meta["title"] == "A title"
    assert video.meta["upload_date"] == "20250102"
    assert len(str(video.meta["description"])) == 4000
    assert "unneeded_blob" not in video.meta


def test_repair_metadata_refreshes_existing_queue_rows(project):
    from omni_curator.create.queue import QueueStore, QVideo

    queue = QueueStore(project.queue_path)
    queue.enqueue_videos([
        QVideo("chan_a_vid1", "chan_a", "/audio/vid1.flac", "noisy", None)
    ])
    queue.close()

    args = build_parser(project).parse_args(["repair-metadata", "--channel", "chan_a"])
    assert cmd_repair_metadata(project, args) == 0

    queue = QueueStore(project.queue_path)
    video = queue.claim_video("seg-0")
    queue.close()
    assert video is not None
    assert video.tier == "clean"
    assert video.category == "news"
    assert video.citation == "https://www.youtube.com/@handle_a"
    assert video.meta["webpage_url"] == "https://www.youtube.com/watch?v=vid1"
    assert video.meta["channel_slug"] == "chan_a"


def test_harvest_writes_source_metadata_to_channel_store(project):
    from omni_curator.create.queue import QClip, QueueStore, QVideo
    from omni_curator.data.store import CuratorStore

    queue = QueueStore(project.queue_path)
    queue.enqueue_videos([
        QVideo(
            "chan_a_vid1",
            "chan_a",
            "/audio/vid1.flac",
            "clean",
            "https://www.youtube.com/@handle_a",
            category="news",
            meta={"title": "Video title", "upload_date": "20250102"},
        )
    ])
    clip = QClip(
        "chan_a_vid1_0000",
        "chan_a_vid1",
        "chan_a",
        0,
        "/clips/chan_a_vid1/seg_0000.flac",
        0.0,
        5.0,
        project.language,
        project.script,
        "https://www.youtube.com/@handle_a",
        tier="clean",
        category="news",
        meta={"title": "Video title", "upload_date": "20250102"},
    )
    queue.complete_video("chan_a_vid1", [clip])
    claimed = queue.claim_clips(1, "label-token")
    assert queue.complete_clips("label-token", [(claimed[0].clip_id, "label", "[]")]) == 1
    queue.close()

    args = build_parser(project).parse_args(["harvest"])
    assert cmd_harvest(project, args) == 0

    store = CuratorStore(project.channels_dir / "chan_a" / "store.sqlite")
    samples = list(store.iter_samples())
    store.close()
    assert len(samples) == 1
    assert samples[0].meta["category"] == "news"
    assert samples[0].meta["tier"] == "clean"
    assert samples[0].meta["title"] == "Video title"


def test_heldout_none_is_empty_but_missing_path_fails_fast(project, tmp_path):
    assert project.heldout_videos() == frozenset()  # None manifest -> no carve
    manifest = tmp_path / "heldout.json"
    manifest.write_text('{"video_ids": ["chan_a_vid1", "chan_b_vid2"]}')
    with_manifest = CuratorProject(
        name="t", language="x", script="X",
        data=tmp_path, db=tmp_path / "db.sqlite", heldout_manifest=manifest,
    )
    assert with_manifest.heldout_videos() == {"chan_a_vid1", "chan_b_vid2"}
    # A configured-but-missing manifest must raise, never silently skip the carve.
    broken = CuratorProject(
        name="t", language="x", script="X",
        data=tmp_path, db=tmp_path / "db.sqlite",
        heldout_manifest=tmp_path / "nonexistent.json",
    )
    with pytest.raises(FileNotFoundError, match="held-out manifest"):
        broken.heldout_videos()


def test_ingest_choices_derive_from_the_registry(project):
    parser = build_parser(project)
    args = parser.parse_args(["ingest", "fleurs"])
    assert args.dataset == ["fleurs"]  # nargs="*": one or more sources
    assert parser.parse_args(["ingest", "fleurs", "fleurs"]).dataset == ["fleurs", "fleurs"]
    with pytest.raises(SystemExit):  # not registered on this project
        parser.parse_args(["ingest", "commonvoice"])


def test_hf_sources_default_to_non_streaming(project, monkeypatch):
    from omni_curator.ingest import fleurs as fleurs_mod
    from omni_curator.ingest import huggingface as hf_mod

    seen: list[bool] = []

    def fake_fleurs(*_args, streaming: bool, **_kwargs):
        seen.append(streaming)
        return iter(())

    def fake_hf(*_args, streaming: bool, **_kwargs):
        seen.append(streaming)
        return iter(())

    monkeypatch.setattr(fleurs_mod, "load_fleurs", fake_fleurs)
    monkeypatch.setattr(hf_mod, "load_hf_audio", fake_hf)

    list(fleurs_source("tg_tj")(project))
    list(huggingface_source("org/dataset")(project))
    list(fleurs_source("tg_tj", streaming=True)(project))
    list(huggingface_source("org/dataset", streaming=True)(project))

    assert seen == [False, False, True, True]


def test_config_typos_fail_at_construction(tmp_path):
    with pytest.raises(ValueError, match="duplicate channel slugs"):
        CuratorProject(
            name="t", language="x", script="X", data=tmp_path, db=tmp_path / "db",
            channels=(channel("a", "@a", "clean", ""), channel("a", "@b", "noisy", "")),
        )
    with pytest.raises(ValueError, match="unknown channel tiers"):
        CuratorProject(
            name="t", language="x", script="X", data=tmp_path, db=tmp_path / "db",
            channels=(channel("a", "@a", "claen", ""),),
        )


def test_mixture_weights_default_and_override(project):
    assert _parse_weights(project, None) == {"fleurs": 490.0}
    assert _parse_weights(project, ["fleurs=100", "youtube-x=5.5"]) == {
        "fleurs": 100.0,
        "youtube-x": 5.5,
    }
    with pytest.raises(SystemExit, match="corpus=hours"):
        _parse_weights(project, ["fleurs"])


def test_channel_dataclass_is_frozen():
    ch = Channel("a", "https://x", "clean", "")
    with pytest.raises(AttributeError):
        ch.slug = "b"  # ty: ignore[invalid-assignment]  # the point of the test
