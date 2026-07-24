from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from english_asr.cv26 import build_identity_ledger, load_identity_ledger, main, prepare


def _write_ledger(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_source(path, rows):
    schema = pa.schema(
        [
            pa.field(
                "audio", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
            ),
            pa.field("path", pa.string()),
            pa.field("upstream_split", pa.string()),
            pa.field("duration_ms", pa.int64()),
            pa.field("client_id", pa.string()),
            pa.field("sentence_id", pa.string()),
            pa.field("source_dataset_id", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _row(clip_id, audio, *, split="train"):
    return {
        "audio": {"bytes": audio, "path": clip_id},
        "path": clip_id,
        "upstream_split": split,
        "duration_ms": 3600,
        "client_id": f"speaker-{clip_id}",
        "sentence_id": f"sentence-{clip_id}",
        "source_dataset_id": "cv26-en",
    }


def test_prepare_classifies_replay_novel_and_benchmark(tmp_path):
    source = tmp_path / "train.parquet"
    benchmark_audio = b"benchmark-by-audio"
    _write_source(
        source,
        [
            _row("benchmark-by-clip.mp3", b"one"),
            _row("benchmark-by-audio.mp3", benchmark_audio),
            _row("cv7.mp3", b"three"),
            _row("novel.mp3", b"four"),
        ],
    )
    cv7 = tmp_path / "cv7.jsonl"
    benchmark = tmp_path / "cv9.jsonl"
    _write_ledger(cv7, [{"clip_id": "cv7.mp3"}])
    _write_ledger(
        benchmark,
        [
            {"clip_id": "benchmark-by-clip.mp3"},
            {"audio_sha256": hashlib.sha256(benchmark_audio).hexdigest()},
        ],
    )
    output = tmp_path / "prepared"

    stats = prepare(
        source_parquets=[source],
        source_revision="hub-commit",
        cv7_ledger=cv7,
        benchmark_ledgers={"cv9-test": benchmark},
        output_dir=output,
    )

    assert stats.rows == 4
    assert stats.train_candidate == 2
    assert stats.post_cv7_candidate == 1
    assert stats.cv7_replay == 1
    assert stats.excluded_benchmark == 2
    assert len((output / "post_cv7_candidate.jsonl").read_text().splitlines()) == 1
    assert len((output / "cv7_replay.jsonl").read_text().splitlines()) == 1
    assert len((output / "train_candidate.jsonl").read_text().splitlines()) == 2
    excluded = [
        json.loads(line) for line in (output / "excluded_benchmark.jsonl").read_text().splitlines()
    ]
    assert {row["clip_id"] for row in excluded} == {
        "benchmark-by-audio.mp3",
        "benchmark-by-clip.mp3",
    }
    summary = json.loads((output / "summary.json").read_text())
    assert summary["source_revision"] == "hub-commit"
    assert summary["benchmark_ledgers"]["cv9-test"]["sha256"]


def test_prepare_without_cv7_records_unknown_replay_and_keeps_candidates(tmp_path):
    source = tmp_path / "train.parquet"
    _write_source(source, [_row("benchmark.mp3", b"one"), _row("candidate.mp3", b"two")])
    benchmark = tmp_path / "cv9.jsonl"
    _write_ledger(benchmark, [{"clip_id": "benchmark.mp3"}])
    output = tmp_path / "prepared"

    stats = prepare(
        source_parquets=[source],
        source_revision="hub-commit",
        cv7_ledger=None,
        benchmark_ledgers={"cv9-test": benchmark},
        output_dir=output,
    )

    assert stats.train_candidate == 1
    assert stats.base_replay_unknown == 1
    assert stats.excluded_benchmark == 1
    assert len((output / "train_candidate.jsonl").read_text().splitlines()) == 1
    assert len((output / "base_replay_unknown.jsonl").read_text().splitlines()) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["cv7_ledger"] is None
    assert summary["base_replay_classification"] == "unknown"


def test_prepare_refuses_non_train_source(tmp_path):
    source = tmp_path / "test.parquet"
    _write_source(source, [_row("test.mp3", b"audio", split="test")])
    cv7 = tmp_path / "cv7.jsonl"
    benchmark = tmp_path / "cv9.jsonl"
    _write_ledger(cv7, [{"clip_id": "cv7.mp3"}])
    _write_ledger(benchmark, [{"clip_id": "cv9.mp3"}])

    with pytest.raises(ValueError, match="upstream train only"):
        prepare(
            source_parquets=[source],
            source_revision="hub-commit",
            cv7_ledger=cv7,
            benchmark_ledgers={"cv9-test": benchmark},
            output_dir=tmp_path / "prepared",
        )


def test_prepare_refuses_partial_source(tmp_path):
    source = tmp_path / "train.parquet.tmp"
    _write_source(source, [_row("train.mp3", b"audio")])
    cv7 = tmp_path / "cv7.jsonl"
    benchmark = tmp_path / "cv9.jsonl"
    _write_ledger(cv7, [{"clip_id": "cv7.mp3"}])
    _write_ledger(benchmark, [{"clip_id": "cv9.mp3"}])

    with pytest.raises(ValueError, match="refuses partial"):
        prepare(
            source_parquets=[source],
            source_revision="hub-commit",
            cv7_ledger=cv7,
            benchmark_ledgers={"cv9-test": benchmark},
            output_dir=tmp_path / "prepared",
        )


def test_build_identity_ledger_pins_sources(tmp_path):
    source = tmp_path / "cv9-test.parquet"
    audio = b"benchmark-audio"
    _write_source(source, [_row("benchmark.mp3", audio)])
    ledger = tmp_path / "cv9-test.jsonl"

    rows = build_identity_ledger(
        source_parquets=[source],
        source_revision="benchmark-commit",
        source_name="open-asr-cv9-test",
        output_ledger=ledger,
    )

    assert rows == 1
    identity = json.loads(ledger.read_text())
    assert identity["clip_id"] == "benchmark.mp3"
    assert identity["audio_sha256"] == hashlib.sha256(audio).hexdigest()
    summary = json.loads(ledger.with_suffix(".jsonl.summary.json").read_text())
    assert summary["source_revision"] == "benchmark-commit"
    assert summary["ledger_sha256"]


def test_identity_ledgers_fail_closed(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_identity_ledger(empty)
    with pytest.raises(FileNotFoundError):
        load_identity_ledger(tmp_path / "missing.jsonl")


def test_cli_requires_named_benchmark_ledger(tmp_path):
    with pytest.raises(SystemExit, match="at least one --benchmark-ledger"):
        main(
            [
                "prepare",
                "--source-parquet",
                str(tmp_path / "missing.parquet"),
                "--source-revision",
                "hub-commit",
                "--output-dir",
                str(tmp_path / "prepared"),
            ]
        )
