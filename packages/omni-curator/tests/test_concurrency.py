"""Tests for live window control + budget splitting (omni_curator.scribe.concurrency)."""

from __future__ import annotations

import pytest

from omni_curator.scribe.concurrency import read_window, split_budget, write_window


def test_read_window_missing_file_uses_default(tmp_path):
    assert read_window(tmp_path / "nope", default=100, cap=300) == 100


def test_read_window_none_uses_default():
    assert read_window(None, default=120, cap=300) == 120


def test_read_window_reads_value(tmp_path):
    f = tmp_path / "win"
    f.write_text("150\n", encoding="utf-8")
    assert read_window(f, default=100, cap=300) == 150


def test_read_window_clamps_to_cap(tmp_path):
    f = tmp_path / "win"
    f.write_text("500", encoding="utf-8")
    assert read_window(f, default=100, cap=300) == 300


def test_read_window_floor_is_one(tmp_path):
    f = tmp_path / "win"
    f.write_text("0", encoding="utf-8")
    assert read_window(f, default=100, cap=300) == 1
    f.write_text("-9", encoding="utf-8")
    assert read_window(f, default=100, cap=300) == 1


@pytest.mark.parametrize("junk", ["", "   ", "abc", "1.5", "12x"])
def test_read_window_garbage_uses_default(tmp_path, junk):
    f = tmp_path / "win"
    f.write_text(junk, encoding="utf-8")
    assert read_window(f, default=77, cap=300) == 77


def test_read_window_default_above_cap_is_clamped(tmp_path):
    # a missing file with a default larger than the pool ceiling must still clamp
    assert read_window(tmp_path / "nope", default=999, cap=300) == 300


def test_write_window_roundtrips(tmp_path):
    f = tmp_path / ".scribe_window.verify"
    write_window(f, 175)
    assert read_window(f, default=100, cap=300) == 175


def test_write_window_creates_parent_and_leaves_no_temp(tmp_path):
    f = tmp_path / "data" / ".scribe_window.verify"
    write_window(f, 42)
    assert f.read_text().strip() == "42"
    assert list(f.parent.glob("*.tmp")) == []  # the atomic temp must not linger


def test_write_window_overwrites(tmp_path):
    f = tmp_path / ".scribe_window.verify"
    write_window(f, 300)
    write_window(f, 150)
    assert read_window(f, default=100, cap=300) == 150


def test_split_budget_no_jobs():
    assert split_budget(300, []) == {}


def test_split_budget_single_job_gets_all():
    assert split_budget(300, ["russian"]) == {"russian": 300}


def test_split_budget_even():
    assert split_budget(300, ["russian", "georgian"]) == {"russian": 150, "georgian": 150}


def test_split_budget_remainder_to_first():
    assert split_budget(301, ["a", "b", "c"]) == {"a": 101, "b": 100, "c": 100}


def test_split_budget_below_job_count_floors_at_one():
    # budget smaller than #jobs: each still gets 1 (sum may exceed budget by design)
    assert split_budget(2, ["a", "b", "c"]) == {"a": 1, "b": 1, "c": 1}


def test_split_budget_sum_equals_budget_when_above_job_count():
    out = split_budget(300, ["a", "b", "c", "d", "e", "f", "g"])
    assert sum(out.values()) == 300
    assert all(v >= 1 for v in out.values())
