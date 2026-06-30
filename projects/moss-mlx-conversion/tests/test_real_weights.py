from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from transformers import AutoTokenizer

from moss_mlx_conversion.paths import MLX_DIR
from moss_mlx_conversion.processor import MossProcessor
from moss_mlx_conversion.runtime.audio import load_waveform
from moss_mlx_conversion.runtime.eval import iter_hf_rows
from moss_mlx_conversion.runtime.streaming_eval import evaluate_one
from moss_mlx_conversion.runtime.transcribe import load_converted_model, transcribe_waveform


def _real_model_dir() -> Path:
    if os.environ.get("MOSS_MLX_RUN_REAL_WEIGHTS") != "1":
        pytest.skip("set MOSS_MLX_RUN_REAL_WEIGHTS=1 to run real MOSS MLX weight tests")
    model_dir = Path(
        os.environ.get(
            "MOSS_MLX_MODEL_DIR",
            str(MLX_DIR / "MOSS-Transcribe-preview-2B-bf16"),
        )
    )
    if not (model_dir / "weights.safetensors").exists():
        pytest.skip(f"missing converted MLX weights at {model_dir}")
    return model_dir


def _load_real_runtime() -> tuple[Any, Any, MossProcessor, Any, Path]:
    model_dir = _real_model_dir()
    try:
        model, config = load_converted_model(model_dir)
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    processor = MossProcessor(
        tokenizer,
        template_path=model_dir / "chat_template_default.py",
        enable_time_marker=False,
    )
    return model, config, processor, tokenizer, model_dir


@pytest.mark.real_weights
def test_loads_real_converted_weights() -> None:
    model, config, _processor, _tokenizer, _model_dir = _load_real_runtime()
    assert config.text_config.hidden_size == 2048
    assert len(model.layers) == 28


@pytest.mark.real_weights
def test_real_fixture_smoke_matches_reference() -> None:
    model, config, processor, tokenizer, _model_dir = _load_real_runtime()
    waveform, _audio_path = load_waveform(None, sample_rate=config.sample_rate)
    result = transcribe_waveform(
        model=model,
        config=config,
        processor=processor,
        tokenizer=tokenizer,
        waveform=waveform,
        max_new_tokens=128,
        prefill_step_size=512,
    )
    assert result.generated_ids[:5] == [4197, 1059, 4158, 6177, 323]
    assert result.transcript.startswith("with her white paint and her scarlet smokestack")


@pytest.mark.real_weights
def test_real_streamed_one_row_eval() -> None:
    if os.environ.get("MOSS_MLX_RUN_STREAMING") != "1":
        pytest.skip("set MOSS_MLX_RUN_STREAMING=1 to run the network streamed eval gate")
    model, config, processor, tokenizer, _model_dir = _load_real_runtime()
    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        example = next(
            iter_hf_rows(
                client,
                dataset="openslr/librispeech_asr",
                config="clean",
                split="test",
                offset=0,
                limit=1,
                page_size=1,
                text_column="text",
                audio_column="audio",
                id_column="id",
            )
        )
        report = evaluate_one(
            client=client,
            example=example,
            model=model,
            config=config,
            processor=processor,
            tokenizer=tokenizer,
            max_new_tokens=256,
            prefill_step_size=512,
            generation_mode="mlx-lm",
        )
    assert report["reference_normalized"]
    assert report["hypothesis_normalized"]
    assert float(report["wer"]) <= 0.2
