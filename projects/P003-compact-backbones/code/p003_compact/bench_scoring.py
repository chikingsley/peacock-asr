"""Project-local scoring benchmark harness for P003.

This is for fast optimization checkpoints, not production evaluation runs.
It times the real scoring phases on a small fixed subset so we can compare
changes without committing to a full sweep.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from p003_compact.backend_loader import get_backend
from p003_compact.dataset import DATASET_REVISION, Utterance, load_speechocean762
from p003_compact.scoring.prepared_bundle import (
    PreparedBundle,
    PreparedItem,
    load_prepared_bundle,
    save_prepared_bundle,
)
from p003_compact.scoring.runtime import (
    _collect_split_outputs,
    _compute_scalar_gop_phase,
    _compute_scalar_gop_phase_k2_batched,
    _prepare_split_inputs,
    _run_evaluation,
)
from p003_compact.settings import settings

if TYPE_CHECKING:
    from p003_compact.evaluate import EvalResult
    from p003_compact.gop import GOPResult, ScalarGOPResult

logger = logging.getLogger("p003_compact.bench")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = PROJECT_ROOT / "experiments" / "benchmarks" / "scoring"
PREPARED_ROOT = BENCH_ROOT / "prepared"
REPORTS_ROOT = BENCH_ROOT / "reports"
PROFILES_ROOT = BENCH_ROOT / "profiles"


@dataclass(frozen=True)
class SplitPrepReport:
    split: str
    utterances_seen: int
    utterances_prepared: int
    skipped: int
    posterior_seconds: float
    posterior_bytes: int
    transport_dtype: str


@dataclass(frozen=True)
class SplitScalarReport:
    split: str
    prepared_utterances: int
    workers: int
    scalar_seconds: float


@dataclass(frozen=True)
class SplitCollectReport:
    split: str
    scalar_rows: int
    utterance_features: int
    collect_seconds: float


@dataclass(frozen=True)
class BenchReport:
    benchmark_name: str
    backend: str
    backend_name: str
    limit: int
    workers: int
    extract_features: bool
    use_gopt: bool
    score_variant: str
    score_alpha: float
    label: str | None
    posterior_batch_size: int
    transport_dtype: str
    dataset_revision: str
    device: str
    backend_load_seconds: float
    dataset_load_seconds: float
    prepared_path: str | None
    scalar_results_path: str | None
    prep_reports: list[SplitPrepReport]
    scalar_reports: list[SplitScalarReport]
    collect_reports: list[SplitCollectReport]
    total_seconds: float
    eval_name: str | None = None
    eval_result: dict[str, Any] | None = None
    extras: dict[str, Any] | None = None


def _ensure_dirs() -> None:
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    PROFILES_ROOT.mkdir(parents=True, exist_ok=True)


def _split_items(
    train: list[Utterance], test: list[Utterance], split: str
) -> list[tuple[str, list[Utterance]]]:
    match split:
        case "train":
            return [("train", train)]
        case "test":
            return [("test", test)]
        case "both":
            return [("train", train), ("test", test)]
        case _:
            msg = f"Unsupported split: {split}"
            raise ValueError(msg)


def _build_stem(args: argparse.Namespace) -> str:
    backend_slug = args.backend[3:].replace("/", "__")
    feat_slug = (
        "gopt"
        if getattr(args, "gopt", False)
        else "feats" if args.extract_features else "scalar"
    )
    stem = (
        f"{backend_slug}_{args.split}_limit{args.limit}_"
        f"{args.transport_dtype}_{feat_slug}_{args.score_variant}"
    )
    if args.label:
        stem += f"_{args.label}"
    return stem


def _prepared_path(args: argparse.Namespace) -> Path:
    return PREPARED_ROOT / f"{_build_stem(args)}.bundle"


def _scalar_results_path(args: argparse.Namespace) -> Path:
    return PREPARED_ROOT / f"{_build_stem(args)}.scalar.pt"


def _report_path(args: argparse.Namespace, mode: str) -> Path:
    worker_suffix = ""
    if mode in {"scalar", "full"}:
        worker_suffix = f"_w{args.workers}"
    return REPORTS_ROOT / f"{mode}_{_build_stem(args)}{worker_suffix}.json"


def _profile_trace_path(args: argparse.Namespace, mode: str) -> Path:
    return PROFILES_ROOT / f"{mode}_{_build_stem(args)}.trace.json"


def _profile_summary_path(args: argparse.Namespace, mode: str) -> Path:
    return PROFILES_ROOT / f"{mode}_{_build_stem(args)}.summary.txt"


def _prepared_subset(
    bundle: PreparedBundle,
    split: str,
    max_utts: int,
) -> list[PreparedItem]:
    prepared = bundle[split]
    return prepared[: min(max_utts, len(prepared))]


def _batch_scalar_k2_phase(
    *,
    prepared: list[PreparedItem],
    batch_utts: int,
    blank: int,
    use_ctc_self: bool,
) -> list[ScalarGOPResult]:
    from p003_compact.gop import ScalarGOPResult  # noqa: PLC0415
    from p003_compact.k2_scalar import compute_scalar_terms_k2_batch  # noqa: PLC0415

    if use_ctc_self:
        results: list[ScalarGOPResult] = []
        for start in range(0, len(prepared), batch_utts):
            chunk = prepared[start : start + batch_utts]
            chunk_cases = [(post, pidx) for post, pidx, _, _ in chunk]
            chunk_results = compute_scalar_terms_k2_batch(
                chunk_cases,
                blank=blank,
                device=settings.scalar_torch_device,
                use_ctc_self=True,
            )
            results.extend(
                ScalarGOPResult(
                    ll_self=ll_self,
                    scores=scores,
                    occupancies=occupancies,
                )
                for ll_self, scores, occupancies in chunk_results
            )
        return results

    return _compute_scalar_gop_phase_k2_batched(
        prepared=prepared,
        split_name=f"{len(prepared)}utt-prototype",
        blank=blank,
        batch_utts=batch_utts,
        compute_scalar_terms_k2_batch=compute_scalar_terms_k2_batch,
    )

def _save_scalar_bundle(
    path: Path,
    scalar_bundle: dict[str, list[ScalarGOPResult | GOPResult]],
    meta: dict[str, Any],
) -> None:
    import torch  # noqa: PLC0415

    torch.save({"meta": meta, "splits": scalar_bundle}, path)


def _load_scalar_bundle(
    path: Path,
) -> tuple[dict[str, Any], dict[str, list[ScalarGOPResult | GOPResult]]]:
    import torch  # noqa: PLC0415

    data = torch.load(path, weights_only=False)
    return data["meta"], data["splits"]


def _posterior_bytes(prepared: list[PreparedItem]) -> int:
    return sum(post.nbytes for post, _, _, _ in prepared)


def _cast_prepared_transport(
    prepared: list[PreparedItem],
    transport_dtype: Literal["float32", "float64"],
) -> list[PreparedItem]:
    if transport_dtype == "float64":
        return [
            (post.astype(np.float64, copy=False), pidx, phones, scores)
            for post, pidx, phones, scores in prepared
        ]
    return [
        (post.astype(np.float32, copy=False), pidx, phones, scores)
        for post, pidx, phones, scores in prepared
    ]


def _serialize_eval_result(result: EvalResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["per_phone_pcc"] = dict(sorted(result.per_phone_pcc.items()))
    return payload


def _write_report(path: Path, report: BenchReport) -> None:
    path.write_text(json.dumps(asdict(report), indent=2) + "\n")
    logger.info("Wrote benchmark report to %s", path)


def _prepare_command(args: argparse.Namespace) -> BenchReport:
    total_start = perf_counter()
    settings.ctc_posterior_batch_size = args.posterior_batch_size
    settings.ctc_posterior_transport_dtype = args.transport_dtype
    backend_ctor = get_backend(args.backend)
    backend = backend_ctor()

    backend_load_start = perf_counter()
    backend.load()
    backend_load_seconds = perf_counter() - backend_load_start

    dataset_load_start = perf_counter()
    data = load_speechocean762(limit=args.limit)
    dataset_load_seconds = perf_counter() - dataset_load_start

    bundle: PreparedBundle = {}
    prep_reports: list[SplitPrepReport] = []
    split_items = _split_items(data.train, data.test, args.split)

    for split_name, utterances in split_items:
        phase_start = perf_counter()
        prepared, skipped = _prepare_split_inputs(
            utterances=utterances,
            split_name=split_name,
            backend=backend,
        )
        prepared = _cast_prepared_transport(prepared, args.transport_dtype)
        posterior_seconds = perf_counter() - phase_start
        bundle[split_name] = prepared
        prep_reports.append(SplitPrepReport(
            split=split_name,
            utterances_seen=len(utterances),
            utterances_prepared=len(prepared),
            skipped=skipped,
            posterior_seconds=posterior_seconds,
            posterior_bytes=_posterior_bytes(prepared),
            transport_dtype=args.transport_dtype,
        ))

    prepared_path = _prepared_path(args)
    save_prepared_bundle(
        prepared_path,
        bundle,
        {
            "backend": args.backend,
            "backend_name": backend.name,
            "blank_index": backend.blank_index,
            "limit": args.limit,
            "split": args.split,
            "dataset_revision": DATASET_REVISION,
            "transport_dtype": args.transport_dtype,
        },
    )

    report = BenchReport(
        benchmark_name="prepare",
        backend=args.backend,
        backend_name=backend.name,
        limit=args.limit,
        workers=args.workers,
        extract_features=args.extract_features,
        use_gopt=False,
        score_variant=args.score_variant,
        score_alpha=args.score_alpha,
        label=args.label,
        posterior_batch_size=args.posterior_batch_size,
        transport_dtype=args.transport_dtype,
        dataset_revision=DATASET_REVISION,
        device=str(settings.torch_device),
        backend_load_seconds=backend_load_seconds,
        dataset_load_seconds=dataset_load_seconds,
        prepared_path=str(prepared_path),
        scalar_results_path=None,
        prep_reports=prep_reports,
        scalar_reports=[],
        collect_reports=[],
        total_seconds=perf_counter() - total_start,
    )
    _write_report(_report_path(args, "prepare"), report)
    return report


def _scalar_command(args: argparse.Namespace) -> BenchReport:
    total_start = perf_counter()
    settings.ctc_posterior_batch_size = args.posterior_batch_size
    settings.ctc_posterior_transport_dtype = args.transport_dtype
    loaded = load_prepared_bundle(_prepared_path(args))
    if loaded is None:
        msg = (
            "Missing prepared bundle. Run the prepare command first: "
            f"{_prepared_path(args)}"
        )
        raise FileNotFoundError(msg)
    prepared_meta, bundle = loaded
    backend_load_seconds = 0.0
    dataset_load_seconds = 0.0

    device = settings.torch_device
    scalar_bundle: dict[str, list[ScalarGOPResult | GOPResult]] = {}
    scalar_reports: list[SplitScalarReport] = []
    blank_value = prepared_meta.get("blank_index", 0)
    blank = int(blank_value) if isinstance(blank_value, int | float | str) else 0

    for split_name in sorted(bundle):
        prepared = bundle[split_name]
        phase_start = perf_counter()
        scalar_results = _compute_scalar_gop_phase(
            prepared=prepared,
            split_name=split_name,
            blank=blank,
            extract_features=args.extract_features,
            device=device,
            n_workers=args.workers,
        )
        scalar_seconds = perf_counter() - phase_start
        scalar_bundle[split_name] = scalar_results
        scalar_reports.append(SplitScalarReport(
            split=split_name,
            prepared_utterances=len(prepared),
            workers=args.workers,
            scalar_seconds=scalar_seconds,
        ))

    scalar_path = _scalar_results_path(args)
    _save_scalar_bundle(
        scalar_path,
        scalar_bundle,
        {
            **prepared_meta,
            "workers": args.workers,
            "extract_features": args.extract_features,
        },
    )

    report = BenchReport(
        benchmark_name="scalar",
        backend=args.backend,
        backend_name=str(prepared_meta["backend_name"]),
        limit=args.limit,
        workers=args.workers,
        extract_features=args.extract_features,
        use_gopt=False,
        score_variant=args.score_variant,
        score_alpha=args.score_alpha,
        label=args.label,
        posterior_batch_size=args.posterior_batch_size,
        transport_dtype=str(prepared_meta["transport_dtype"]),
        dataset_revision=DATASET_REVISION,
        device=str(device),
        backend_load_seconds=backend_load_seconds,
        dataset_load_seconds=dataset_load_seconds,
        prepared_path=str(_prepared_path(args)),
        scalar_results_path=str(scalar_path),
        prep_reports=[],
        scalar_reports=scalar_reports,
        collect_reports=[],
        total_seconds=perf_counter() - total_start,
    )
    _write_report(_report_path(args, "scalar"), report)
    return report


def _prototype_batch_command(args: argparse.Namespace) -> BenchReport:
    total_start = perf_counter()
    loaded = load_prepared_bundle(_prepared_path(args))
    if loaded is None:
        msg = (
            "Missing prepared bundle. Run the prepare command first: "
            f"{_prepared_path(args)}"
        )
        raise FileNotFoundError(msg)
    prepared_meta, bundle = loaded
    if args.split == "both":
        msg = "prototype-batch requires --split train or --split test"
        raise ValueError(msg)

    prepared = _prepared_subset(bundle, args.split, args.max_utts)
    blank_value = prepared_meta.get("blank_index", 0)
    blank = int(blank_value) if isinstance(blank_value, int | float | str) else 0

    scalar_start = perf_counter()
    baseline = _compute_scalar_gop_phase(
        prepared=prepared,
        split_name=f"{args.split}-baseline",
        blank=blank,
        extract_features=False,
        device=settings.torch_device,
        n_workers=1,
    )
    baseline_seconds = perf_counter() - scalar_start

    batch_start = perf_counter()
    candidate = _batch_scalar_k2_phase(
        prepared=prepared,
        batch_utts=args.batch_utts,
        blank=blank,
        use_ctc_self=args.batch_self_method == "ctc",
    )
    batch_seconds = perf_counter() - batch_start

    max_score_diff = 0.0
    max_occ_diff: float | None = None
    for left, right in zip(baseline, candidate, strict=True):
        max_score_diff = max(
            max_score_diff,
            float(np.max(np.abs(np.asarray(left.scores) - np.asarray(right.scores)))),
        )
        if left.occupancies and right.occupancies:
            occ_diff = float(
                np.max(
                    np.abs(
                        np.asarray(left.occupancies) - np.asarray(right.occupancies),
                    ),
                ),
            )
            if max_occ_diff is None:
                max_occ_diff = occ_diff
            else:
                max_occ_diff = max(max_occ_diff, occ_diff)

    report = BenchReport(
        benchmark_name="prototype_batch",
        backend=args.backend,
        backend_name=str(prepared_meta["backend_name"]),
        limit=args.limit,
        workers=1,
        extract_features=False,
        use_gopt=False,
        score_variant=args.score_variant,
        score_alpha=args.score_alpha,
        label=args.label,
        posterior_batch_size=args.posterior_batch_size,
        transport_dtype=str(prepared_meta["transport_dtype"]),
        dataset_revision=DATASET_REVISION,
        device=str(settings.scalar_torch_device),
        backend_load_seconds=0.0,
        dataset_load_seconds=0.0,
        prepared_path=str(_prepared_path(args)),
        scalar_results_path=None,
        prep_reports=[],
        scalar_reports=[
            SplitScalarReport(
                split=args.split,
                prepared_utterances=len(prepared),
                workers=1,
                scalar_seconds=baseline_seconds,
            ),
            SplitScalarReport(
                split=f"{args.split}-batch-k2",
                prepared_utterances=len(prepared),
                workers=args.batch_utts,
                scalar_seconds=batch_seconds,
            ),
        ],
        collect_reports=[],
        total_seconds=perf_counter() - total_start,
        extras={
            "max_utts": args.max_utts,
            "batch_utts": args.batch_utts,
            "baseline_seconds": baseline_seconds,
            "batch_seconds": batch_seconds,
            "speedup_vs_baseline": (
                None if batch_seconds == 0 else baseline_seconds / batch_seconds
            ),
            "max_score_abs_diff": max_score_diff,
            "max_occupancy_abs_diff": max_occ_diff,
            "scalar_backend": settings.ctc_scalar_backend,
            "scalar_device": settings.ctc_scalar_device,
            "batch_self_method": args.batch_self_method,
        },
    )
    _write_report(_report_path(args, "prototype_batch"), report)
    return report


def _profile_command(args: argparse.Namespace) -> BenchReport:
    loaded = load_prepared_bundle(_prepared_path(args))
    if loaded is None:
        msg = (
            "Missing prepared bundle. Run the prepare command first: "
            f"{_prepared_path(args)}"
        )
        raise FileNotFoundError(msg)
    prepared_meta, bundle = loaded
    if args.split == "both":
        msg = "profile requires --split train or --split test"
        raise ValueError(msg)

    prepared = _prepared_subset(bundle, args.split, args.max_utts)
    blank_value = prepared_meta.get("blank_index", 0)
    blank = int(blank_value) if isinstance(blank_value, int | float | str) else 0
    summary_path = _profile_summary_path(args, args.profile_mode)

    if args.profile_mode == "collect-features":
        return _profile_collect_features(
            args=args,
            prepared=prepared,
            prepared_meta=prepared_meta,
            blank=blank,
            summary_path=summary_path,
        )

    import torch  # noqa: PLC0415

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_path = (
        _profile_trace_path(args, args.profile_mode)
        if args.profile_export_trace
        else None
    )

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        if args.profile_mode == "baseline":
            _compute_scalar_gop_phase(
                prepared=prepared,
                split_name=f"{args.split}-profile-baseline",
                blank=blank,
                extract_features=False,
                device=settings.torch_device,
                n_workers=1,
            )
        elif args.profile_mode == "batch-k2":
            _batch_scalar_k2_phase(
                prepared=prepared,
                batch_utts=args.batch_utts,
                blank=blank,
                use_ctc_self=args.batch_self_method == "ctc",
            )
        else:
            msg = f"Unsupported profiler mode: {args.profile_mode}"
            raise ValueError(msg)

    if trace_path is not None:
        prof.export_chrome_trace(str(trace_path))
    summary = prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=args.profile_row_limit,
    )
    summary_path.write_text(summary + "\n", encoding="utf-8")

    report = BenchReport(
        benchmark_name="profile",
        backend=args.backend,
        backend_name=str(prepared_meta["backend_name"]),
        limit=args.limit,
        workers=1,
        extract_features=False,
        use_gopt=False,
        score_variant=args.score_variant,
        score_alpha=args.score_alpha,
        label=args.label,
        posterior_batch_size=args.posterior_batch_size,
        transport_dtype=str(prepared_meta["transport_dtype"]),
        dataset_revision=DATASET_REVISION,
        device=str(settings.scalar_torch_device),
        backend_load_seconds=0.0,
        dataset_load_seconds=0.0,
        prepared_path=str(_prepared_path(args)),
        scalar_results_path=None,
        prep_reports=[],
        scalar_reports=[],
        collect_reports=[],
        total_seconds=0.0,
        extras={
            "max_utts": args.max_utts,
            "batch_utts": args.batch_utts,
            "profile_mode": args.profile_mode,
            "trace_path": None if trace_path is None else str(trace_path),
            "summary_path": str(summary_path),
            "scalar_backend": settings.ctc_scalar_backend,
            "scalar_device": settings.ctc_scalar_device,
            "batch_self_method": args.batch_self_method,
        },
    )
    _write_report(_report_path(args, f"profile_{args.profile_mode}"), report)
    if trace_path is not None:
        logger.info("Wrote profiler trace to %s", trace_path)
    logger.info("Wrote profiler summary to %s", summary_path)
    return report


def _profile_collect_features(
    *,
    args: argparse.Namespace,
    prepared: list[PreparedItem],
    prepared_meta: dict[str, Any],
    blank: int,
    summary_path: Path,
) -> BenchReport:
    import cProfile  # noqa: PLC0415
    import io  # noqa: PLC0415
    import pstats  # noqa: PLC0415

    from p003_compact.gop import compute_gop_scalar  # noqa: PLC0415

    scalar_gop_results = [
        compute_gop_scalar(post, pidx, blank)
        for post, pidx, _, _ in prepared
    ]

    profiler = cProfile.Profile()
    profiler.enable()
    _collect_split_outputs(
        prepared=prepared,
        scalar_gop_results=cast(
            "list[ScalarGOPResult | GOPResult]",
            scalar_gop_results,
        ),
        extract_features=True,
        n_workers=1,
        blank=blank,
        device=settings.torch_device,
        split_name=f"{args.split}-profile-collect",
        score_config=(args.score_variant, args.score_alpha),
    )
    profiler.disable()

    summary_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=summary_stream)
    stats.sort_stats("cumulative")
    stats.print_stats(args.profile_row_limit)
    summary_path.write_text(summary_stream.getvalue(), encoding="utf-8")

    report = BenchReport(
        benchmark_name="profile",
        backend=args.backend,
        backend_name=str(prepared_meta["backend_name"]),
        limit=args.limit,
        workers=1,
        extract_features=True,
        use_gopt=False,
        score_variant=args.score_variant,
        score_alpha=args.score_alpha,
        label=args.label,
        posterior_batch_size=args.posterior_batch_size,
        transport_dtype=str(prepared_meta["transport_dtype"]),
        dataset_revision=DATASET_REVISION,
        device=str(settings.scalar_torch_device),
        backend_load_seconds=0.0,
        dataset_load_seconds=0.0,
        prepared_path=str(_prepared_path(args)),
        scalar_results_path=None,
        prep_reports=[],
        scalar_reports=[],
        collect_reports=[],
        total_seconds=0.0,
        extras={
            "max_utts": args.max_utts,
            "profile_mode": args.profile_mode,
            "trace_path": None,
            "summary_path": str(summary_path),
            "scalar_backend": settings.ctc_scalar_backend,
            "scalar_device": settings.ctc_scalar_device,
            "profiler": "cProfile",
        },
    )
    _write_report(_report_path(args, f"profile_{args.profile_mode}"), report)
    logger.info("Wrote profiler summary to %s", summary_path)
    return report


def _full_command(args: argparse.Namespace) -> BenchReport:
    total_start = perf_counter()
    settings.ctc_posterior_batch_size = args.posterior_batch_size
    settings.ctc_posterior_transport_dtype = args.transport_dtype
    backend_ctor = get_backend(args.backend)
    backend = backend_ctor()

    backend_load_start = perf_counter()
    backend.load()
    backend_load_seconds = perf_counter() - backend_load_start

    dataset_load_start = perf_counter()
    data = load_speechocean762(limit=args.limit)
    dataset_load_seconds = perf_counter() - dataset_load_start

    device = settings.torch_device
    split_items = _split_items(data.train, data.test, args.split)
    extract_features = args.extract_features or args.gopt

    prep_reports: list[SplitPrepReport] = []
    scalar_reports: list[SplitScalarReport] = []
    collect_reports: list[SplitCollectReport] = []

    split_scalar_rows: dict[str, list[tuple[str, float, float]]] = {}
    split_utts: dict[str, list[Any]] = {}

    for split_name, utterances in split_items:
        prep_start = perf_counter()
        prepared, skipped = _prepare_split_inputs(
            utterances=utterances,
            split_name=split_name,
            backend=backend,
        )
        prepared = _cast_prepared_transport(prepared, args.transport_dtype)
        prep_seconds = perf_counter() - prep_start
        prep_reports.append(SplitPrepReport(
            split=split_name,
            utterances_seen=len(utterances),
            utterances_prepared=len(prepared),
            skipped=skipped,
            posterior_seconds=prep_seconds,
            posterior_bytes=_posterior_bytes(prepared),
            transport_dtype=args.transport_dtype,
        ))

        scalar_start = perf_counter()
        scalar_gop_results = _compute_scalar_gop_phase(
            prepared=prepared,
            split_name=split_name,
            blank=backend.blank_index,
            extract_features=extract_features,
            device=device,
            n_workers=args.workers,
        )
        scalar_seconds = perf_counter() - scalar_start
        scalar_reports.append(SplitScalarReport(
            split=split_name,
            prepared_utterances=len(prepared),
            workers=args.workers,
            scalar_seconds=scalar_seconds,
        ))

        collect_start = perf_counter()
        scalar_rows, utt_feats = _collect_split_outputs(
            prepared=prepared,
            scalar_gop_results=scalar_gop_results,
            extract_features=extract_features,
            n_workers=args.workers,
            blank=backend.blank_index,
            device=device,
            split_name=split_name,
            score_config=(args.score_variant, args.score_alpha),
        )
        collect_seconds = perf_counter() - collect_start
        collect_reports.append(SplitCollectReport(
            split=split_name,
            scalar_rows=len(scalar_rows),
            utterance_features=len(utt_feats),
            collect_seconds=collect_seconds,
        ))
        split_scalar_rows[split_name] = scalar_rows
        split_utts[split_name] = utt_feats

    eval_name: str | None = None
    eval_result_payload: dict[str, Any] | None = None
    if args.split == "both":
        eval_name, eval_result, _ = _run_evaluation(
            use_gopt=args.gopt,
            use_feats=extract_features,
            backend_name=backend.name,
            feat_dim=len(backend.vocab) + 2,
            device=device,
            train_scalar=split_scalar_rows["train"],
            test_scalar=split_scalar_rows["test"],
            train_utts=split_utts["train"],
            test_utts=split_utts["test"],
            verbose=args.verbose,
        )
        eval_result_payload = _serialize_eval_result(eval_result)

    report = BenchReport(
        benchmark_name="full",
        backend=args.backend,
        backend_name=backend.name,
        limit=args.limit,
        workers=args.workers,
        extract_features=extract_features,
        use_gopt=args.gopt,
        score_variant=args.score_variant,
        score_alpha=args.score_alpha,
        label=args.label,
        posterior_batch_size=args.posterior_batch_size,
        transport_dtype=args.transport_dtype,
        dataset_revision=DATASET_REVISION,
        device=str(device),
        backend_load_seconds=backend_load_seconds,
        dataset_load_seconds=dataset_load_seconds,
        prepared_path=None,
        scalar_results_path=None,
        prep_reports=prep_reports,
        scalar_reports=scalar_reports,
        collect_reports=collect_reports,
        total_seconds=perf_counter() - total_start,
        eval_name=eval_name,
        eval_result=eval_result_payload,
    )
    _write_report(_report_path(args, "full"), report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="peacock-benchmark-scoring",
        description="Benchmark P003 scoring phases on a small subset.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--backend",
        required=True,
        help="Backend spec, e.g. hf:Peacockery/w2v-bert-phoneme-en",
    )
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--posterior-batch-size",
        type=int,
        default=settings.ctc_posterior_batch_size,
    )
    parser.add_argument(
        "--transport-dtype",
        choices=["float32", "float64"],
        default=settings.ctc_posterior_transport_dtype,
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--gopt", action="store_true")
    parser.add_argument(
        "--score-variant",
        choices=["gop_sf", "logit_margin", "logit_combined"],
        default="gop_sf",
    )
    parser.add_argument("--score-alpha", type=float, default=0.5)
    parser.add_argument(
        "--max-utts",
        type=int,
        default=16,
        help="Max prepared utterances to use for prototype/profile commands.",
    )
    parser.add_argument(
        "--batch-utts",
        type=int,
        default=8,
        help="Utterances per k2 prototype batch.",
    )
    parser.add_argument(
        "--batch-self-method",
        choices=["exact", "ctc"],
        default="exact",
        help="How to compute ll_self inside the batched k2 prototype.",
    )
    parser.add_argument(
        "--profile-mode",
        choices=["baseline", "batch-k2", "collect-features"],
        default="baseline",
        help="Which scalar path to profile.",
    )
    parser.add_argument(
        "--profile-row-limit",
        type=int,
        default=30,
        help="Rows to include in profiler summary tables.",
    )
    parser.add_argument(
        "--profile-export-trace",
        action="store_true",
        help="Also export a Chrome trace. Off by default because traces get huge.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "prepare",
        help="Time only posterior extraction and save prepared matrices.",
    )
    subparsers.add_parser(
        "scalar",
        help="Time only scalar GOP from a prepared bundle.",
    )
    subparsers.add_parser(
        "full",
        help="Run prepare + scalar + collect, and eval when split=both.",
    )
    subparsers.add_parser(
        "prototype-batch",
        help="Compare current per-utterance scalar GOP with batched k2 scalar GOP.",
    )
    subparsers.add_parser(
        "profile",
        help="Profile the scalar GOP stage on a prepared bundle.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _ensure_dirs()

    match args.command:
        case "prepare":
            _prepare_command(args)
        case "scalar":
            _scalar_command(args)
        case "full":
            _full_command(args)
        case "prototype-batch":
            _prototype_batch_command(args)
        case "profile":
            _profile_command(args)
        case _:
            parser.print_help()
            raise SystemExit(1)


if __name__ == "__main__":
    main()
