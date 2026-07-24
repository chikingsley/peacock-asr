from __future__ import annotations

import json

import pytest

from english_asr.pnc import (
    build_restored_mixture,
    prepare_pilot,
    prepare_restoration_pool,
    score_manifest,
)


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_and_score_pilot(tmp_path) -> None:
    lexical = tmp_path / "lexical.jsonl"
    reference = tmp_path / "reference.jsonl"
    output = tmp_path / "pilot.jsonl"
    _write(
        lexical,
        [
            {"sample_id": "b", "text": "how are you"},
            {"sample_id": "a", "text": "hello world"},
        ],
    )
    _write(
        reference,
        [
            {"sample_id": "a", "text": "Hello, world!"},
            {"sample_id": "b", "text": "How are you?"},
        ],
    )

    summary = prepare_pilot(lexical, reference, output, limit=1, seed=4)
    row = json.loads(output.read_text())
    row["prediction"] = row["reference_text"]
    _write(output, [row])
    report = score_manifest(output, prediction_field="prediction")

    assert summary["rows"] == 1
    assert report["word_preservation"]["rate"] == 1.0
    assert report["capitalization_accuracy"] == 1.0
    assert report["exact_pnc_row_rate"] == 1.0


def test_prepare_rejects_lexical_reference_mismatch(tmp_path) -> None:
    lexical = tmp_path / "lexical.jsonl"
    reference = tmp_path / "reference.jsonl"
    _write(lexical, [{"sample_id": "a", "text": "hello world"}])
    _write(reference, [{"sample_id": "a", "text": "Hello there!"}])

    with pytest.raises(ValueError, match="lexical/reference mismatch"):
        prepare_pilot(lexical, reference, tmp_path / "pilot.jsonl")


def test_prepare_and_build_restored_mixture(tmp_path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    source_summaries = []
    balanced_rows = []
    for source_name in ("alpha", "beta"):
        source_dir = template / source_name
        source_dir.mkdir()
        train_rows = [{"sample_id": f"{source_name}-train", "text": "hello world"}]
        dev_rows = [{"sample_id": f"{source_name}-dev", "text": "how are you"}]
        _write(source_dir / "train.jsonl", train_rows)
        _write(source_dir / "dev.jsonl", dev_rows)
        balanced_rows.extend(dev_rows)
        source_summaries.append(
            {
                "name": source_name,
                "sampling_weight": 0.5,
                "output": {
                    "train": f"{source_name}/train.jsonl",
                    "train_rows": 1,
                    "train_sha256": "template-train",
                    "dev": f"{source_name}/dev.jsonl",
                    "dev_rows": 1,
                    "dev_sha256": "template-dev",
                },
            }
        )
    _write(template / "balanced-dev.jsonl", balanced_rows)
    (template / "mixture_summary.json").write_text(
        json.dumps(
            {
                "label_profile": "lexical-lower-v1",
                "seed": 0,
                "sources": source_summaries,
                "balanced_validation": {
                    "path": "balanced-dev.jsonl",
                    "rows_per_source": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    pool = tmp_path / "pool.jsonl"
    prepared = prepare_restoration_pool(template, pool)
    restored_rows = []
    for row in [json.loads(line) for line in pool.read_text().splitlines()]:
        row["prediction_text"] = "Hello, world!" if row["_pnc_split"] == "train" else "How are you?"
        restored_rows.append(row)
    restored_manifest = tmp_path / "restored.jsonl"
    _write(restored_manifest, restored_rows)

    output = tmp_path / "restored-mixture"
    summary = build_restored_mixture(
        restored_manifest,
        template,
        output,
        model_name="test-pnc",
    )

    assert prepared["rows"] == 4
    assert summary["restoration"]["word_preservation_rate"] == 1.0
    assert summary["balanced_validation"]["rows"] == 2
    assert json.loads((output / "alpha" / "train.jsonl").read_text())["text"] == ("Hello, world!")
    balanced_texts = [
        json.loads(line)["text"]
        for line in (output / "balanced-dev.jsonl").read_text().splitlines()
    ]
    assert balanced_texts == [
        "How are you?",
        "How are you?",
    ]


def test_build_restored_mixture_rejects_changed_words(tmp_path) -> None:
    template = tmp_path / "template"
    source = template / "alpha"
    source.mkdir(parents=True)
    row = {"sample_id": "a", "text": "hello world"}
    _write(source / "train.jsonl", [row])
    _write(source / "dev.jsonl", [row])
    _write(template / "balanced-dev.jsonl", [row])
    (template / "mixture_summary.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "alpha",
                        "sampling_weight": 1.0,
                        "output": {
                            "train": "alpha/train.jsonl",
                            "train_rows": 1,
                            "train_sha256": "train",
                            "dev": "alpha/dev.jsonl",
                            "dev_rows": 1,
                            "dev_sha256": "dev",
                        },
                    }
                ],
                "balanced_validation": {"path": "balanced-dev.jsonl", "rows_per_source": 1},
            }
        ),
        encoding="utf-8",
    )
    restored = tmp_path / "restored.jsonl"
    _write(
        restored,
        [
            {
                "sample_id": "a",
                "text": "hello world",
                "lexical_text": "hello world",
                "prediction_text": "Hello there!",
                "_pnc_source": "alpha",
                "_pnc_split": "train",
            }
        ],
    )

    with pytest.raises(ValueError, match="changed words"):
        build_restored_mixture(restored, template, tmp_path / "output", model_name="bad")
