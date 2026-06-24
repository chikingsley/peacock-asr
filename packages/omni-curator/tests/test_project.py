"""CuratorProject CLI: every language gets the full, identical command set from one config."""

from __future__ import annotations

import pytest

from omni_curator.create.youtube import Channel, channel
from omni_curator.project import (
    CuratorProject,
    _parse_weights,
    build_parser,
    fleurs_source,
)

ALL_COMMANDS = {
    "list", "download", "cookies",
    "enqueue", "segment", "labelq", "harvest", "archive",
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
            channel("chan_a", "@handle_a", "clean", "test"),
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
