"""Tests for additive ASR boundary and CTC-alignment quality signals."""

from __future__ import annotations

import json

from omni_curator.audit.quality import asr_edge_mismatch
from omni_curator.audit.quality_cli import main


def test_asr_edge_mismatch_ignores_middle_only_change():
    mismatch = asr_edge_mismatch("alpha beta gamma", "alpha zeta gamma")

    assert mismatch.beginning_error_chars == 0
    assert mismatch.end_error_chars == 0


def test_asr_edge_mismatch_measures_boundary_insert_delete():
    mismatch = asr_edge_mismatch("alpha beta", "extra alpha")

    assert mismatch.beginning_operation == "insert"
    assert mismatch.beginning_error_chars == len("extra ")
    assert mismatch.end_operation == "delete"
    assert mismatch.end_error_chars == len(" beta")


def test_sample_is_bounded_and_resolves_audio_paths(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"audio_filepath": f"audio/{index}.wav", "text": str(index)}) + "\n"
            for index in range(20)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pilot.jsonl"

    assert main(["sample", "--input", str(source), "--output", str(output), "--limit", "5"]) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert all(row["sample_id"].startswith("pilot-0-") for row in rows)
    assert all(row["audio_filepath"].startswith("/") for row in rows)


def test_edge_cli_preserves_rows_and_adds_signals(tmp_path):
    source = tmp_path / "predictions.jsonl"
    source.write_text(
        json.dumps(
            {
                "audio_filepath": "/audio/a.wav",
                "text": "alpha beta",
                "hypothesis": "extra alpha",
                "duration": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "scored.jsonl"
    summary = tmp_path / "summary.json"

    assert (
        main(
            [
                "edge",
                "--input",
                str(source),
                "--output",
                str(output),
                "--summary",
                str(summary),
                "--beginning-threshold",
                "3",
                "--end-threshold",
                "3",
                "--model-id",
                "draft.nemo",
                "--model-sha256",
                "abc123",
            ]
        )
        == 0
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert row["text"] == "alpha beta"
    assert row["quality"]["asr_edge"]["would_flag"] is True
    assert row["quality"]["asr_edge"]["draft_model_sha256"] == "abc123"
    assert report["would_flag"] == 1


def test_nfa_prepare_normalizes_without_mutating_source(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"audio_filepath": "/audio/a.wav", "text": "یکی|دو"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "prepared.jsonl"
    summary = tmp_path / "summary.json"

    assert (
        main(
            [
                "nfa-prepare",
                "--input",
                str(source),
                "--output",
                str(output),
                "--summary",
                str(summary),
                "--language",
                "fas_Arab",
            ]
        )
        == 0
    )

    assert json.loads(source.read_text(encoding="utf-8"))["text"] == "یکی|دو"
    assert json.loads(output.read_text(encoding="utf-8"))["text"] == "یکی دو"
    assert json.loads(summary.read_text(encoding="utf-8"))["normalization_changed"] == 1


def test_nfa_summarize_adds_alignment_margins(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    word_ctm = tmp_path / "a.ctm"
    word_ctm.write_text("a 1 0.25 0.50 alpha\na 1 0.75 0.50 beta\n", encoding="utf-8")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"audio_filepath": str(audio), "text": "alpha beta", "duration": 2.0})
        + "\n",
        encoding="utf-8",
    )
    aligned = tmp_path / "aligned.jsonl"
    aligned.write_text(
        json.dumps({"audio_filepath": str(audio), "words_level_ctm_filepath": str(word_ctm)})
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "scored.jsonl"
    summary = tmp_path / "summary.json"

    assert (
        main(
            [
                "nfa-summarize",
                "--input",
                str(source),
                "--aligned-manifest",
                str(aligned),
                "--output",
                str(output),
                "--summary",
                str(summary),
            ]
        )
        == 0
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    signal = row["quality"]["ctc_alignment"]
    assert signal["status"] == "aligned"
    assert signal["word_coverage"] == 1.0
    assert signal["leading_margin_seconds"] == 0.25
    assert signal["trailing_margin_seconds"] == 0.75
