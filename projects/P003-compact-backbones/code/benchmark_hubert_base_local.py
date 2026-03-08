#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchaudio",
#     "transformers>=4.40",
#     "datasets",
#     "accelerate",
#     "numpy",
#     "evaluate",
#     "jiwer",
#     "pydantic",
#     "pydantic-settings",
#     "soundfile",
#     "scipy",
# ]
# ///
"""Run the canonical local HuBERT-base micro-benchmark matrix."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _load_module(script_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module from {script_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_script(script_path: Path, module_name: str, argv: list[str]) -> int:
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *argv]
        module = _load_module(script_path, module_name)
        module.main()
    finally:
        sys.argv = previous_argv
    return 0


def run_case(
    repo_root: Path,
    *,
    name: str,
    extra_args: list[str],
) -> int:
    benchmark_script = (
        repo_root
        / "projects"
        / "P003-compact-backbones"
        / "code"
        / "training"
        / "benchmark_phoneme_head.py"
    )
    output_json = (
        repo_root
        / "projects"
        / "P003-compact-backbones"
        / "experiments"
        / "benchmarks"
        / f"{name}.json"
    )
    argv = [
        "--model-name",
        "facebook/hubert-base-ls960",
        "--max-train-samples",
        "32",
        "--batch-size",
        "4",
        "--dataloader-workers",
        "0",
        "--num-batches",
        "2",
        "--step-batches",
        "1",
        "--output-json",
        str(output_json),
        *extra_args,
    ]
    _emit(f"Running benchmark: {benchmark_script} {' '.join(argv)}".rstrip())
    return _run_script(
        benchmark_script,
        "p003_benchmark_phoneme_head",
        argv,
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    cases = [
        ("hubert_base_cpu_gc_on", ["--device", "cpu", "--gradient-checkpointing"]),
        ("hubert_base_cpu_gc_off", ["--device", "cpu", "--no-gradient-checkpointing"]),
    ]
    if len(sys.argv) > 1:
        case_name = sys.argv[1]
        matching = [case for case in cases if case[0] == case_name]
        if not matching:
            sys.stderr.write(f"Unknown case: {case_name}\n")
            return 2
        cases = matching
    for name, extra_args in cases:
        code = run_case(repo_root, name=name, extra_args=extra_args)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
