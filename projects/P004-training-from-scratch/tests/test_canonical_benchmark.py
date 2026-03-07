from __future__ import annotations

from pathlib import Path

from p004_training_from_scratch.canonical.benchmark import (
    _build_variant_command,
    _build_variant_comparison,
)


def test_build_variant_command_toggles_compile_flag(tmp_path: Path) -> None:
    compile_off = _build_variant_command(
        output_dir=tmp_path / "compile_off",
        train_manifest=tmp_path / "train.jsonl.gz",
        dev_manifest=tmp_path / "dev.jsonl.gz",
        tokens_path=tmp_path / "tokens.txt",
        train_limit=24,
        dev_limit=8,
        epochs=4,
        batch_size=4,
        model_type="conformer_like",
        hidden_dim=192,
        encoder_layers=3,
        attention_heads=4,
        conv_kernel_size=15,
        dropout=0.1,
        learning_rate=3e-4,
        seed=42,
        enable_compile=False,
    )
    compile_on = _build_variant_command(
        output_dir=tmp_path / "compile_on",
        train_manifest=tmp_path / "train.jsonl.gz",
        dev_manifest=tmp_path / "dev.jsonl.gz",
        tokens_path=tmp_path / "tokens.txt",
        train_limit=24,
        dev_limit=8,
        epochs=4,
        batch_size=4,
        model_type="conformer_like",
        hidden_dim=192,
        encoder_layers=3,
        attention_heads=4,
        conv_kernel_size=15,
        dropout=0.1,
        learning_rate=3e-4,
        seed=42,
        enable_compile=True,
    )

    assert "--disable-compile" in compile_off
    assert "--disable-compile" not in compile_on
    assert "--disable-wandb" in compile_off


def test_build_variant_comparison_reports_ratios() -> None:
    comparison = _build_variant_comparison(
        compile_off={
            "elapsed_seconds": 12.0,
            "benchmark": {
                "steady_state_mean_step_seconds": 0.5,
                "peak_memory_reserved_mb": 1000.0,
            },
        },
        compile_on={
            "elapsed_seconds": 9.0,
            "benchmark": {
                "steady_state_mean_step_seconds": 0.25,
                "peak_memory_reserved_mb": 1200.0,
            },
        },
    )

    assert comparison["compile_on_vs_off_total_elapsed_ratio"] == 0.75
    assert comparison["compile_on_vs_off_steady_step_ratio"] == 0.5
    assert comparison["compile_on_vs_off_peak_reserved_ratio"] == 1.2
    assert comparison["compile_on_faster_total"] is True
    assert comparison["compile_on_faster_steady_state"] is True
