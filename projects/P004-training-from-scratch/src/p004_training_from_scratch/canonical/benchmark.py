"""Stable-lane canonical benchmark for compile and SDPA baselines.

References
- PyTorch `torch.compile`: https://pytorch.org/docs/stable/generated/torch.compile.html
- PyTorch `scaled_dot_product_attention`:
  https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from p004_training_from_scratch.canonical.train_smoke import _capture_sdpa_state
from p004_training_from_scratch.machine_manifest import capture_machine_manifest
from p004_training_from_scratch.settings import PROJECT_ROOT

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "experiments" / "benchmarks" / "canonical_phone_ctc"
)


def run_canonical_benchmark(
    *,
    output_dir: Path,
    train_manifest: Path,
    dev_manifest: Path,
    tokens_path: Path,
    train_limit: int = 24,
    dev_limit: int = 8,
    epochs: int = 4,
    batch_size: int = 4,
    model_type: str = "conformer",
    hidden_dim: int = 192,
    encoder_layers: int = 3,
    attention_heads: int = 4,
    conv_kernel_size: int = 15,
    dropout: float = 0.1,
    learning_rate: float = 3e-4,
    seed: int = 42,
    sdpa_batch_size: int = 4,
    sdpa_seq_len: int = 1024,
    sdpa_warmup_iters: int = 5,
    sdpa_timed_iters: int = 20,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    machine_manifest_path = output_dir / "machine_manifest.json"
    payload: dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(output_dir),
        "machine_manifest_path": str(machine_manifest_path),
        "config": {
            "train_limit": train_limit,
            "dev_limit": dev_limit,
            "epochs": epochs,
            "batch_size": batch_size,
            "model_type": model_type,
            "hidden_dim": hidden_dim,
            "encoder_layers": encoder_layers,
            "attention_heads": attention_heads,
            "conv_kernel_size": conv_kernel_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "seed": seed,
            "sdpa_batch_size": sdpa_batch_size,
            "sdpa_seq_len": sdpa_seq_len,
            "sdpa_warmup_iters": sdpa_warmup_iters,
            "sdpa_timed_iters": sdpa_timed_iters,
        },
        "success": False,
    }
    payload["machine_manifest"] = capture_machine_manifest(output=machine_manifest_path)

    try:
        payload["sdpa_microbenchmark"] = _run_sdpa_microbenchmark(
            batch_size=sdpa_batch_size,
            seq_len=sdpa_seq_len,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            warmup_iters=sdpa_warmup_iters,
            timed_iters=sdpa_timed_iters,
        )
    except Exception as exc:  # pragma: no cover - exercised by real runtime
        payload["sdpa_microbenchmark"] = {
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        _write_report(payload, report_path)
        return payload

    variants: list[dict[str, Any]] = []
    for name, enable_compile in (("compile_off", False), ("compile_on", True)):
        variant_output_dir = output_dir / name
        variant = _run_variant(
            name=name,
            output_dir=variant_output_dir,
            train_manifest=train_manifest,
            dev_manifest=dev_manifest,
            tokens_path=tokens_path,
            train_limit=train_limit,
            dev_limit=dev_limit,
            epochs=epochs,
            batch_size=batch_size,
            model_type=model_type,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            attention_heads=attention_heads,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            learning_rate=learning_rate,
            seed=seed,
            enable_compile=enable_compile,
        )
        variants.append(variant)

    payload["variants"] = variants
    if len(variants) == 2 and all(variant["success"] for variant in variants):
        payload["comparison"] = _build_variant_comparison(
            compile_off=_find_variant(variants, "compile_off")["report"],
            compile_on=_find_variant(variants, "compile_on")["report"],
        )
        payload["success"] = True
    else:
        payload["comparison"] = None

    return _write_report(payload, report_path)


def _run_variant(
    *,
    name: str,
    output_dir: Path,
    train_manifest: Path,
    dev_manifest: Path,
    tokens_path: Path,
    train_limit: int,
    dev_limit: int,
    epochs: int,
    batch_size: int,
    model_type: str,
    hidden_dim: int,
    encoder_layers: int,
    attention_heads: int,
    conv_kernel_size: int,
    dropout: float,
    learning_rate: float,
    seed: int,
    enable_compile: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "benchmark_stdout.log"
    command = _build_variant_command(
        output_dir=output_dir,
        train_manifest=train_manifest,
        dev_manifest=dev_manifest,
        tokens_path=tokens_path,
        train_limit=train_limit,
        dev_limit=dev_limit,
        epochs=epochs,
        batch_size=batch_size,
        model_type=model_type,
        hidden_dim=hidden_dim,
        encoder_layers=encoder_layers,
        attention_heads=attention_heads,
        conv_kernel_size=conv_kernel_size,
        dropout=dropout,
        learning_rate=learning_rate,
        seed=seed,
        enable_compile=enable_compile,
    )
    env = os.environ.copy()
    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    process_elapsed_seconds = round(time.perf_counter() - started_at, 3)
    stdout_payload = completed.stdout
    if completed.stderr:
        stdout_payload = f"{stdout_payload}\n--- stderr ---\n{completed.stderr}"
    stdout_path.write_text(stdout_payload, encoding="utf-8")

    report_path = output_dir / "report.json"
    report: dict[str, Any] | None = None
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    return {
        "name": name,
        "enable_compile": enable_compile,
        "command": command,
        "stdout_path": str(stdout_path),
        "process_elapsed_seconds": process_elapsed_seconds,
        "returncode": completed.returncode,
        "report": report,
        "success": completed.returncode == 0 and bool(report and report["success"]),
    }


def _build_variant_command(
    *,
    output_dir: Path,
    train_manifest: Path,
    dev_manifest: Path,
    tokens_path: Path,
    train_limit: int,
    dev_limit: int,
    epochs: int,
    batch_size: int,
    model_type: str,
    hidden_dim: int,
    encoder_layers: int,
    attention_heads: int,
    conv_kernel_size: int,
    dropout: float,
    learning_rate: float,
    seed: int,
    enable_compile: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "p004_training_from_scratch.cli.canonical_train_smoke",
        "--output-dir",
        str(output_dir),
        "--train-manifest",
        str(train_manifest),
        "--dev-manifest",
        str(dev_manifest),
        "--tokens",
        str(tokens_path),
        "--train-limit",
        str(train_limit),
        "--dev-limit",
        str(dev_limit),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--model-type",
        model_type,
        "--hidden-dim",
        str(hidden_dim),
        "--encoder-layers",
        str(encoder_layers),
        "--attention-heads",
        str(attention_heads),
        "--conv-kernel-size",
        str(conv_kernel_size),
        "--dropout",
        str(dropout),
        "--learning-rate",
        str(learning_rate),
        "--seed",
        str(seed),
        "--disable-wandb",
    ]
    if not enable_compile:
        command.append("--disable-compile")
    return command


def _run_sdpa_microbenchmark(
    *,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    attention_heads: int,
    warmup_iters: int,
    timed_iters: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        msg = "torch.cuda.is_available() returned false"
        raise RuntimeError(msg)

    device = torch.device("cuda")
    head_dim = hidden_dim // attention_heads
    if hidden_dim % attention_heads != 0:
        msg = "hidden_dim must be divisible by attention_heads"
        raise ValueError(msg)

    q = torch.randn(
        batch_size,
        attention_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for _ in range(warmup_iters):
        functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    torch.cuda.synchronize(device)

    durations_ms: list[float] = []
    for _ in range(timed_iters):
        start = time.perf_counter()
        out = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        torch.cuda.synchronize(device)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    checksum = float(out.float().mean().item())
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "attention_heads": attention_heads,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "backend_state": _capture_sdpa_state(torch),
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "mean_duration_ms": round(_mean(durations_ms), 6),
        "median_duration_ms": round(_median(durations_ms), 6),
        "checksum_mean": round(checksum, 6),
    }


def _build_variant_comparison(
    *,
    compile_off: dict[str, Any],
    compile_on: dict[str, Any],
) -> dict[str, Any]:
    off_benchmark = compile_off["benchmark"]
    on_benchmark = compile_on["benchmark"]
    return {
        "compile_on_vs_off_total_elapsed_ratio": _safe_ratio(
            compile_on["elapsed_seconds"],
            compile_off["elapsed_seconds"],
        ),
        "compile_on_vs_off_steady_step_ratio": _safe_ratio(
            on_benchmark["steady_state_mean_step_seconds"],
            off_benchmark["steady_state_mean_step_seconds"],
        ),
        "compile_on_vs_off_peak_reserved_ratio": _safe_ratio(
            on_benchmark["peak_memory_reserved_mb"],
            off_benchmark["peak_memory_reserved_mb"],
        ),
        "compile_on_faster_total": compile_on["elapsed_seconds"]
        < compile_off["elapsed_seconds"],
        "compile_on_faster_steady_state": (
            on_benchmark["steady_state_mean_step_seconds"]
            < off_benchmark["steady_state_mean_step_seconds"]
        ),
        "compile_off_elapsed_seconds": compile_off["elapsed_seconds"],
        "compile_on_elapsed_seconds": compile_on["elapsed_seconds"],
        "compile_off_steady_state_mean_step_seconds": off_benchmark[
            "steady_state_mean_step_seconds"
        ],
        "compile_on_steady_state_mean_step_seconds": on_benchmark[
            "steady_state_mean_step_seconds"
        ],
        "compile_off_peak_memory_reserved_mb": off_benchmark[
            "peak_memory_reserved_mb"
        ],
        "compile_on_peak_memory_reserved_mb": on_benchmark[
            "peak_memory_reserved_mb"
        ],
    }


def _find_variant(variants: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for variant in variants:
        if variant["name"] == name:
            return variant
    msg = f"variant not found: {name}"
    raise ValueError(msg)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 6)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[middle])
    return float((sorted_values[middle - 1] + sorted_values[middle]) / 2)


def _write_report(payload: dict[str, Any], report_path: Path) -> dict[str, Any]:
    report_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "_build_variant_command",
    "_build_variant_comparison",
    "run_canonical_benchmark",
]
