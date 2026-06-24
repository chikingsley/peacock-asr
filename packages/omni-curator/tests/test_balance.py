"""Tests for the Scribe budget balancer (omni_curator.scribe.balance)."""

from __future__ import annotations

from pathlib import Path

from omni_curator.scribe.balance import apply_budget, parse_jobs, window_path


def test_parse_jobs_keys_are_lang_and_command():
    ps = (
        "uv run --project projects/russian-asr russian-curate verify --workers 100\n"
        "uv run --project projects/georgian-asr georgian-curate labelq --workers 50\n"
        "/some/python russian-curate verify  # a second russian verify worker\n"
        "/usr/bin/yt-dlp https://youtube.com/unrelated\n"
    )
    assert parse_jobs(ps) == ["georgian:labelq", "russian:verify"]


def test_parse_jobs_same_lang_both_stages_are_distinct_jobs():
    # labelq + verify for one language must count as TWO jobs (each reads its own window file),
    # so the budget is split per (lang, command), not collapsed to one per language.
    ps = (
        "tajik-curate labelq --workers 100\n"
        "tajik-curate verify --workers 100\n"
    )
    assert parse_jobs(ps) == ["tajik:labelq", "tajik:verify"]


def test_parse_jobs_ignores_non_scribe_stages():
    ps = (
        "russian-curate download --lane gluetun-lane1\n"
        "tajik-curate segment --procs 2\n"
        "python -m omni_curator.scribe.balance --budget 300\n"
    )
    assert parse_jobs(ps) == []


def test_window_path_matches_project_layout():
    assert window_path("russian:verify", root=Path("/x")) == Path(
        "/x/projects/russian-asr/data/.scribe_window.verify"
    )
    assert window_path("georgian:labelq", root=Path("/x")) == Path(
        "/x/projects/georgian-asr/data/.scribe_window.labelq"
    )


def test_apply_budget_splits_and_writes(tmp_path):
    out = apply_budget(300, ["russian:verify", "georgian:labelq"], root=tmp_path)
    assert out == {"russian:verify": 150, "georgian:labelq": 150}
    assert (
        tmp_path / "projects/russian-asr/data/.scribe_window.verify"
    ).read_text().strip() == "150"
    assert (
        tmp_path / "projects/georgian-asr/data/.scribe_window.labelq"
    ).read_text().strip() == "150"


def test_apply_budget_same_lang_two_stages_each_get_half(tmp_path):
    out = apply_budget(300, ["tajik:labelq", "tajik:verify"], root=tmp_path)
    assert out == {"tajik:labelq": 150, "tajik:verify": 150}
    assert (tmp_path / "projects/tajik-asr/data/.scribe_window.labelq").exists()
    assert (tmp_path / "projects/tajik-asr/data/.scribe_window.verify").exists()


def test_apply_budget_single_job_gets_full(tmp_path):
    out = apply_budget(300, ["russian:verify"], root=tmp_path)
    assert out == {"russian:verify": 300}
    assert (
        tmp_path / "projects/russian-asr/data/.scribe_window.verify"
    ).read_text().strip() == "300"


def test_apply_budget_no_jobs_writes_nothing(tmp_path):
    assert apply_budget(300, [], root=tmp_path) == {}
    assert not (tmp_path / "projects").exists()
