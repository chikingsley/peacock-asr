from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf

from english_asr.exams import _duration, _encoded_audio, _write_exact
from english_asr.external_matrix import (
    Binding,
    _aggregate_matrix,
    _prepare_output,
    _valid_json,
    parse_binding,
)


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(8_000, dtype=np.float32), 16_000, format="WAV")
    return buffer.getvalue()


def test_encoded_audio_and_duration() -> None:
    payload = _wav_bytes()
    encoded, suffix = _encoded_audio({"bytes": payload, "path": "clip.wav"})

    assert encoded == payload
    assert suffix == ".wav"
    assert _duration(payload) == pytest.approx(0.5)


def test_write_exact_reuses_identical_and_refuses_drift(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    _write_exact(path, b"same")
    _write_exact(path, b"same")

    assert path.read_bytes() == b"same"
    with pytest.raises(RuntimeError, match="differs"):
        _write_exact(path, b"different")


def test_manifest_shape_is_json_serializable() -> None:
    row = {
        "audio_filepath": "/exam/audio.wav",
        "text": "Reference.",
        "duration": 1.0,
        "sample_id": "exam:test:0",
        "audio_sha256": "0" * 64,
    }

    assert json.loads(json.dumps(row)) == row


def test_parse_external_matrix_binding_requires_existing_named_path(tmp_path) -> None:
    manifest = tmp_path / "exam.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")

    binding = parse_binding(f"common_voice={manifest}")

    assert binding.name == "common_voice"
    assert binding.path == manifest.resolve()


def test_external_matrix_output_resumes_only_exact_valid_configuration(tmp_path) -> None:
    output = tmp_path / "matrix"
    matrix = {"models": [{"name": "base"}], "batch_size": 16}

    _prepare_output(output, matrix)
    _prepare_output(output, matrix)

    assert _valid_json(output / "matrix.json")
    with pytest.raises(RuntimeError, match="configuration drift"):
        _prepare_output(output, {"models": [{"name": "candidate"}], "batch_size": 16})


def test_external_matrix_aggregate_reports_macro_pooled_and_rtfx(tmp_path) -> None:
    model = Binding("base", tmp_path / "model.nemo")
    exams = [
        Binding("first", tmp_path / "first.jsonl"),
        Binding("second", tmp_path / "second.jsonl"),
    ]
    model_dir = tmp_path / "matrix" / "base"
    model_dir.mkdir(parents=True)
    for exam, wer, errors, words, audio, elapsed in (
        (exams[0], 10.0, 10, 100, 200.0, 2.0),
        (exams[1], 20.0, 40, 200, 300.0, 3.0),
    ):
        (model_dir / f"{exam.name}.summary.json").write_text(
            json.dumps(
                {
                    "wer_percent": wer,
                    "rows": 2,
                    "deletions": errors,
                    "insertions": 0,
                    "substitutions": 0,
                    "reference_words": words,
                }
            ),
            encoding="utf-8",
        )
        (model_dir / f"{exam.name}.runtime.json").write_text(
            json.dumps(
                {"rtfx": audio / elapsed, "audio_seconds": audio, "elapsed_seconds": elapsed}
            ),
            encoding="utf-8",
        )

    result = _aggregate_matrix(tmp_path / "matrix", [model], exams)

    assert result["models"]["base"]["macro_wer_percent"] == 15.0
    assert result["models"]["base"]["pooled_wer_percent"] == pytest.approx(100 / 6)
    assert result["models"]["base"]["aggregate_rtfx"] == 100.0
