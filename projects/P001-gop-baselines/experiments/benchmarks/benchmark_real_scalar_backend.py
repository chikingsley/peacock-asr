from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from p001_gop.backend_loader import get_backend
from p001_gop.dataset import DATASET_REVISION, Utterance, load_speechocean762
from p001_gop.gop import ScalarGOPResult, compute_gop_scalar_scores_only
from p001_gop.settings import settings

if TYPE_CHECKING:
    import numpy as np


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    msg = "Could not locate the P001 project root."
    raise RuntimeError(msg)


def _default_prepared_cache_path(backend: str, split: str, limit: int) -> Path:
    cache_dir = _repo_root() / "experiments" / "benchmarks" / "prepared"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{backend}_{split}_limit{limit}.pt"


def _collect_prepared_cases(
    *,
    utterances: list[Utterance],
    backend_name: str,
) -> tuple[list[tuple[str, np.ndarray, list[int]]], float, str, int]:
    backend = get_backend(backend_name)()
    load_start = perf_counter()
    backend.load()
    load_s = perf_counter() - load_start
    prepared: list[tuple[str, np.ndarray, list[int]]] = []
    for utt in tqdm(utterances, desc=f"prepare:{backend_name}", unit="utt"):
        phone_indices: list[int] = []
        for phone in utt.phones:
            mapped = backend.map_phone(phone)
            if mapped is not None:
                phone_indices.extend(mapped)
        if len(phone_indices) < 2:  # noqa: PLR2004
            continue
        posteriors = backend.get_posteriors(utt.audio, utt.sample_rate)
        prepared.append((utt.utterance_id, posteriors, phone_indices))
    return prepared, load_s, backend.name, backend.blank_index


def _load_or_prepare_cases(
    *,
    backend_name: str,
    split: str,
    limit: int,
    refresh: bool,
) -> tuple[list[tuple[str, np.ndarray, list[int]]], dict[str, Any]]:
    cache_path = _default_prepared_cache_path(backend_name, split, limit)
    if cache_path.exists() and not refresh:
        payload = torch.load(cache_path, weights_only=False)
        metadata = dict(payload["metadata"])
        metadata["prepared_cache"] = str(cache_path)
        metadata["prepared_cache_hit"] = True
        return payload["prepared"], metadata

    ds = load_speechocean762(limit=limit)
    utterances = ds.train if split == "train" else ds.test
    prepared, backend_load_s, resolved_backend_name, blank_index = _collect_prepared_cases(
        utterances=utterances,
        backend_name=backend_name,
    )
    metadata: dict[str, Any] = {
        "dataset_revision": DATASET_REVISION,
        "split": split,
        "limit": limit,
        "backend": backend_name,
        "resolved_backend_name": resolved_backend_name,
        "blank_index": blank_index,
        "backend_load_s": backend_load_s,
        "prepared_cases": len(prepared),
        "prepared_cache": str(cache_path),
        "prepared_cache_hit": False,
    }
    torch.save({"prepared": prepared, "metadata": metadata}, cache_path)
    return prepared, metadata


def _run_scalar_benchmark(
    *,
    prepared: list[tuple[str, np.ndarray, list[int]]],
    blank_index: int,
    scalar_backend: str,
    scalar_device: str,
    warmup_cases: int,
) -> tuple[list[ScalarGOPResult], float]:
    prev_backend = settings.ctc_scalar_backend
    prev_device = settings.ctc_scalar_device
    settings.ctc_scalar_backend = scalar_backend
    settings.ctc_scalar_device = scalar_device
    try:
        warmup = min(warmup_cases, len(prepared))
        for _, posteriors, phone_indices in prepared[:warmup]:
            compute_gop_scalar_scores_only(
                posteriors,
                phone_indices,
                blank=blank_index,
            )

        if scalar_backend == "k2" and settings.scalar_torch_device.type == "cuda":
            torch.cuda.synchronize(settings.scalar_torch_device)

        start = perf_counter()
        results = [
            compute_gop_scalar_scores_only(
                posteriors,
                phone_indices,
                blank=blank_index,
            )
            for _, posteriors, phone_indices in prepared
        ]
        if scalar_backend == "k2" and settings.scalar_torch_device.type == "cuda":
            torch.cuda.synchronize(settings.scalar_torch_device)
        duration_s = perf_counter() - start
    finally:
        settings.ctc_scalar_backend = prev_backend
        settings.ctc_scalar_device = prev_device
    return results, duration_s


def _summarize_results(
    *,
    prepared: list[tuple[str, np.ndarray, list[int]]],
    results: list[ScalarGOPResult],
    duration_s: float,
    label: str,
) -> dict[str, Any]:
    n_utts = len(prepared)
    n_phones = sum(len(phone_indices) for _, _, phone_indices in prepared)
    return {
        "label": label,
        "duration_s": duration_s,
        "n_utts": n_utts,
        "n_phones": n_phones,
        "utt_per_s": n_utts / duration_s if duration_s else None,
        "phones_per_s": n_phones / duration_s if duration_s else None,
    }


def _parity_against_python(
    *,
    python_results: list[ScalarGOPResult],
    other_results: list[ScalarGOPResult],
) -> dict[str, float]:
    score_max_abs = 0.0
    occ_max_abs = 0.0
    ll_self_max_abs = 0.0
    for py, other in zip(python_results, other_results, strict=True):
        ll_self_max_abs = max(ll_self_max_abs, abs(py.ll_self - other.ll_self))
        score_max_abs = max(
            score_max_abs,
            max(
                (abs(a - b) for a, b in zip(py.scores, other.scores, strict=True)),
                default=0.0,
            ),
        )
        occ_max_abs = max(
            occ_max_abs,
            max(
                (
                    abs(a - b)
                    for a, b in zip(py.occupancies, other.occupancies, strict=True)
                ),
                default=0.0,
            ),
        )
    return {
        "ll_self_max_abs_diff": ll_self_max_abs,
        "score_max_abs_diff": score_max_abs,
        "occupancy_max_abs_diff": occ_max_abs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark P001 scalar GOP backends on real SpeechOcean data.",
    )
    parser.add_argument(
        "--backend",
        choices=["original", "xlsr-espeak"],
        default="xlsr-espeak",
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--warmup-cases", type=int, default=2)
    parser.add_argument(
        "--refresh-prepared",
        action="store_true",
        help="Ignore any saved posterior cache and recollect from the backend.",
    )
    parser.add_argument(
        "--k2-device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults under experiments/benchmarks/results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared, metadata = _load_or_prepare_cases(
        backend_name=args.backend,
        split=args.split,
        limit=args.limit,
        refresh=args.refresh_prepared,
    )
    if not prepared:
        msg = "No valid prepared cases collected."
        raise RuntimeError(msg)

    python_results, python_duration_s = _run_scalar_benchmark(
        prepared=prepared,
        blank_index=int(metadata["blank_index"]),
        scalar_backend="python",
        scalar_device="cpu",
        warmup_cases=args.warmup_cases,
    )
    benchmarks: list[dict[str, Any]] = [
        _summarize_results(
            prepared=prepared,
            results=python_results,
            duration_s=python_duration_s,
            label="python/cpu",
        ),
    ]

    k2_cpu_results, k2_cpu_duration_s = _run_scalar_benchmark(
        prepared=prepared,
        blank_index=int(metadata["blank_index"]),
        scalar_backend="k2",
        scalar_device="cpu",
        warmup_cases=args.warmup_cases,
    )
    k2_cpu_summary = _summarize_results(
        prepared=prepared,
        results=k2_cpu_results,
        duration_s=k2_cpu_duration_s,
        label="k2/cpu",
    )
    k2_cpu_summary["parity_vs_python"] = _parity_against_python(
        python_results=python_results,
        other_results=k2_cpu_results,
    )
    benchmarks.append(k2_cpu_summary)

    if args.k2_device in {"auto", "cuda"} and torch.cuda.is_available():
        k2_cuda_results, k2_cuda_duration_s = _run_scalar_benchmark(
            prepared=prepared,
            blank_index=int(metadata["blank_index"]),
            scalar_backend="k2",
            scalar_device="cuda",
            warmup_cases=args.warmup_cases,
        )
        k2_cuda_summary = _summarize_results(
            prepared=prepared,
            results=k2_cuda_results,
            duration_s=k2_cuda_duration_s,
            label="k2/cuda",
        )
        k2_cuda_summary["parity_vs_python"] = _parity_against_python(
            python_results=python_results,
            other_results=k2_cuda_results,
        )
        benchmarks.append(k2_cuda_summary)

    payload = {
        "metadata": metadata,
        "benchmarks": benchmarks,
    }
    if args.output is None:
        results_dir = _repo_root() / "experiments" / "benchmarks" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        args.output = results_dir / (
            f"real_scalar_{args.backend}_{args.split}_limit{args.limit}.json"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nSaved benchmark to {args.output}")


if __name__ == "__main__":
    main()
