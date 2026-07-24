from __future__ import annotations

import json

import pytest

from english_asr.arm import parse_evaluation, wait_for


def test_parse_evaluation_requires_named_existing_manifest(tmp_path) -> None:
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")

    evaluation = parse_evaluation(f"cv26={manifest}")

    assert evaluation.name == "cv26"
    assert evaluation.manifest == manifest.resolve()
    with pytest.raises(ValueError, match="expected NAME=MANIFEST"):
        parse_evaluation(str(manifest))


def test_wait_for_returns_when_all_markers_exist(tmp_path) -> None:
    first = tmp_path / "first.complete"
    second = tmp_path / "second.complete"
    first.touch()
    second.touch()

    wait_for([first, second], 0.001)
