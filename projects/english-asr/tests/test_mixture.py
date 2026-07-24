from __future__ import annotations

import json

import pytest

from english_asr.mixture import (
    Source,
    build_mixture,
    normalize_training_text,
    parse_source,
    parse_source_weights,
)


def _write_manifest(path, source, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(texts):
            row = {
                "audio_filepath": f"/{source}/{index}.flac",
                "duration": 1.0,
                "sample_id": f"{source}-{index}",
                "text": text,
            }
            handle.write(json.dumps(row) + "\n")


def _source(tmp_path, name, train, dev):
    directory = tmp_path / name
    _write_manifest(directory / "train.jsonl", name, train)
    _write_manifest(directory / "dev.jsonl", name, dev)
    return Source(name, directory)


def test_normalize_training_text_harmonizes_case_and_punctuation() -> None:
    assert normalize_training_text("NO, We CAN\u2019T--stop.") == "no we can't stop"
    assert normalize_training_text("LifeOfTheLaw.org") == "lifeofthelaw org"


def test_parse_source_requires_stable_name_and_directory(tmp_path) -> None:
    assert parse_source(f"common-voice={tmp_path}") == Source("common-voice", tmp_path.resolve())
    with pytest.raises(ValueError, match="NAME=DIRECTORY"):
        parse_source(str(tmp_path))
    with pytest.raises(ValueError, match="lowercase"):
        parse_source(f"CommonVoice={tmp_path}")


def test_parse_source_weights_rejects_ambiguous_values() -> None:
    assert parse_source_weights([]) is None
    assert parse_source_weights(["primary=0.9", "replay=0.1"]) == {
        "primary": 0.9,
        "replay": 0.1,
    }
    with pytest.raises(ValueError, match="duplicate"):
        parse_source_weights(["primary=0.9", "primary=0.1"])
    with pytest.raises(ValueError, match="positive"):
        parse_source_weights(["primary=0"])


def test_build_mixture_preserves_sources_and_balances_dev(tmp_path) -> None:
    first = _source(tmp_path, "first", ["HELLO."], ["A!", "B!", "C!"])
    second = _source(tmp_path, "second", ["World?"], ["D?", "E?", "F?"])
    output = tmp_path / "mixture"

    summary = build_mixture([first, second], output, validation_per_source=2, seed=7)

    assert summary["label_profile"] == "lexical-lower-v1"
    assert [source["sampling_weight"] for source in summary["sources"]] == [0.5, 0.5]
    assert summary["balanced_validation"]["rows"] == 4
    assert len((output / "balanced-dev.jsonl").read_text().splitlines()) == 4
    assert json.loads((output / "first" / "train.jsonl").read_text())["text"] == "hello"
    assert json.loads((output / "second" / "train.jsonl").read_text())["text"] == "world"
    with pytest.raises(FileExistsError, match="immutable output"):
        build_mixture([first, second], output)


def test_build_mixture_preserves_explicit_sampling_weights(tmp_path) -> None:
    first = _source(tmp_path, "first", ["HELLO."], ["A!"])
    second = _source(tmp_path, "second", ["World?"], ["B?"])
    output = tmp_path / "weighted-mixture"

    summary = build_mixture(
        [first, second],
        output,
        sampling_weights={"first": 0.9, "second": 0.1},
    )

    assert {source["name"]: source["sampling_weight"] for source in summary["sources"]} == {
        "first": 0.9,
        "second": 0.1,
    }
    with pytest.raises(ValueError, match="match sources exactly"):
        build_mixture(
            [first, second],
            tmp_path / "missing-weight",
            sampling_weights={"first": 1.0},
        )
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        build_mixture(
            [first, second],
            tmp_path / "bad-total",
            sampling_weights={"first": 0.8, "second": 0.1},
        )
