"""CLI entry points for peacock-asr."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
import traceback
from contextlib import (
    AbstractContextManager,
    nullcontext,
    redirect_stderr,
    redirect_stdout,
)
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import torch

    from peacock_asr.backends.base import PhonemeBackend
    from peacock_asr.batch_config import BatchResolvedJob
    from peacock_asr.dataset import Utterance
    from peacock_asr.evaluate import EvalResult
    from peacock_asr.gop import GOPResult, ScalarGOPResult
    from peacock_asr.gopt_model import UtteranceFeats
    from peacock_asr.score_variants import ScoreVariant

logger = logging.getLogger("peacock_asr")


def _cuda_device_count() -> int:
    import torch  # noqa: PLC0415

    return torch.cuda.device_count()


def cmd_download(_args: argparse.Namespace) -> None:
    """Download the SpeechOcean762 dataset."""
    from peacock_asr.dataset import load_speechocean762  # noqa: PLC0415

    data = load_speechocean762()
    print(f"Train: {len(data.train)} utterances")  # noqa: T201
    print(f"Test:  {len(data.test)} utterances")  # noqa: T201
    print("Dataset cached by HuggingFace datasets.")  # noqa: T201


def _cache_path(
    features_dir: Path,
    backend_name: str,
    split: str,
    score_variant: ScoreVariant,
    score_alpha: float,
) -> Path:
    """Build the cache file path for a given backend and split."""
    safe_name = re.sub(r"[^\w\-.]", "_", backend_name)
    alpha_key = f"{score_alpha:.4f}".replace(".", "p")
    d = features_dir / safe_name / f"{score_variant}_a{alpha_key}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{split}.pt"


def _load_cache(
    path: Path,
    backend_name: str,
    dataset_revision: str,
    extract_features: bool,  # noqa: FBT001
) -> tuple[list[tuple[str, float, float]], list[UtteranceFeats]] | None:
    """Load cached features if valid. Returns None on miss."""
    import torch  # noqa: PLC0415

    if not path.exists():
        return None

    data = torch.load(path, weights_only=False)
    if data.get("backend") != backend_name:
        return None
    if data.get("dataset_revision") != dataset_revision:
        return None
    # If we need features but cache was saved without them, miss
    if extract_features and not data.get("extract_features"):
        return None

    return data["scalar_results"], data.get("utt_feats", [])


def _save_cache(
    *,
    path: Path,
    backend_name: str,
    dataset_revision: str,
    extract_features: bool,
    scalar_results: list[tuple[str, float, float]],
    utt_feats: list[UtteranceFeats],
) -> None:
    """Save extraction results to cache."""
    import torch  # noqa: PLC0415

    torch.save(
        {
            "backend": backend_name,
            "dataset_revision": dataset_revision,
            "extract_features": extract_features,
            "scalar_results": scalar_results,
            "utt_feats": utt_feats,
        },
        path,
    )
    logger.info("Saved cache to %s", path)


def _scalar_gop_worker(
    args: tuple[np.ndarray, list[int], int],
) -> ScalarGOPResult:
    """Worker for Phase 2: CPU-only scalar GOP (no features, no GPU)."""
    from peacock_asr.gop import compute_gop_scalar  # noqa: PLC0415

    posteriors, phone_indices, blank = args
    return compute_gop_scalar(posteriors, phone_indices, blank)


def _collect_split_outputs(
    *,
    prepared: list[tuple[np.ndarray, list[int], list[str], list[float]]],
    scalar_gop_results: list[ScalarGOPResult | GOPResult],
    extract_features: bool,
    n_workers: int,
    blank: int,
    device: torch.device,
    split_name: str,
    score_config: tuple[ScoreVariant, float],
) -> tuple[list[tuple[str, float, float]], list[UtteranceFeats]]:
    from tqdm import tqdm  # noqa: PLC0415

    from peacock_asr.gop import (  # noqa: PLC0415
        GOPResult,
        ScalarGOPResult,
        compute_gop_features,
    )
    from peacock_asr.gopt_model import UtteranceFeats  # noqa: PLC0415
    from peacock_asr.score_variants import apply_score_variant  # noqa: PLC0415

    scalar_results: list[tuple[str, float, float]] = []
    utt_feats: list[UtteranceFeats] = []
    score_variant, score_alpha = score_config
    iterator = (
        tqdm(prepared, desc=f"{split_name} [collect]", unit="utt")
        if extract_features and n_workers > 1
        else prepared
    )

    for idx, (post, pidx, valid_phones, valid_scores) in enumerate(iterator):
        gop = scalar_gop_results[idx]
        adjusted_scores = apply_score_variant(
            variant=score_variant,
            score_alpha=score_alpha,
            posteriors=post,
            phone_indices=pidx,
            baseline_scores=gop.scores,
        )

        for phone, gop_score, human_score in zip(
            valid_phones, adjusted_scores, valid_scores, strict=True,
        ):
            scalar_results.append((phone, gop_score, human_score))

        if not extract_features:
            continue

        if n_workers <= 1:
            if not isinstance(gop, GOPResult):
                msg = "Expected GOPResult in sequential mode."
                raise TypeError(msg)
            if gop.features is None:
                msg = "Expected feature vectors in sequential mode."
                raise RuntimeError(msg)
            features = gop.features
        else:
            if not isinstance(gop, ScalarGOPResult):
                msg = "Expected ScalarGOPResult in parallel mode."
                raise TypeError(msg)
            features = compute_gop_features(
                post, pidx, gop.ll_self, blank, device=device,
            )

        feat_vecs = [
            [*features[i].tolist(), gop.occupancies[i]]
            for i in range(len(valid_phones))
        ]
        utt_feats.append(UtteranceFeats(
            phones=valid_phones,
            feat_vecs=feat_vecs,
            scores=valid_scores,
        ))

    return scalar_results, utt_feats


def _process_split(
    *,
    utterances: list[Utterance],
    split_name: str,
    backend: PhonemeBackend,
    extract_features: bool,
    device: torch.device,
    score_variant: ScoreVariant,
    score_alpha: float,
    n_workers: int = 0,
) -> tuple[list[tuple[str, float, float]], list[UtteranceFeats]]:
    """Process a split, returning (scalar_results, utt_feats).

    Three phases:
      1. Collect posteriors sequentially (GPU, fast)
      2. Scalar GOP in parallel across CPU cores (the bottleneck)
      3. Feature extraction sequentially on GPU (fast)

    Args:
        n_workers: Number of parallel workers. 0 = auto (cpu_count - 2).
                   1 = sequential (no multiprocessing).
    """
    import os  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from tqdm import tqdm  # noqa: PLC0415

    from peacock_asr.gop import compute_gop  # noqa: PLC0415

    if n_workers == 0:
        n_workers = max(1, (os.cpu_count() or 2) - 2)

    # -- Phase 1: Collect posteriors (GPU, sequential) --
    prepared: list[tuple[np.ndarray, list[int], list[str], list[float]]] = []
    skipped = 0

    for utt in tqdm(utterances, desc=f"{split_name} [posteriors]", unit="utt"):
        phone_indices: list[int] = []
        valid_phones: list[str] = []
        valid_scores: list[float] = []

        for phone, score in zip(
            utt.phones, utt.phone_scores, strict=True,
        ):
            indices = backend.map_phone(phone)
            if indices is not None:
                phone_indices.extend(indices)
                valid_phones.append(phone)
                valid_scores.append(score)

        if len(phone_indices) < 2:  # noqa: PLR2004
            skipped += 1
            continue

        posteriors = backend.get_posteriors(utt.audio, utt.sample_rate)
        prepared.append((posteriors, phone_indices, valid_phones, valid_scores))

    # -- Phase 2: Scalar GOP (CPU, parallel) --
    blank = backend.blank_index

    if n_workers > 1 and len(prepared) > 1:
        logger.info(
            "[%s] Computing scalar GOP with %d workers...",
            split_name, n_workers,
        )
        worker_args: list[tuple[np.ndarray, list[int], int]] = [
            (post, pidx, blank) for post, pidx, _, _ in prepared
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            scalar_gop_results: list[ScalarGOPResult | GOPResult] = list(tqdm(
                pool.map(_scalar_gop_worker, worker_args, chunksize=4),
                total=len(worker_args),
                desc=f"{split_name} [GOP]",
                unit="utt",
            ))
    else:
        scalar_gop_results = []
        for post, pidx, _, _ in tqdm(
            prepared, desc=f"{split_name} [GOP]", unit="utt",
        ):
            scalar_gop_results.append(compute_gop(
                posteriors=post,
                phone_indices=pidx,
                blank=blank,
                extract_features=extract_features,
                device=device if extract_features else None,
            ))

    scalar_results, utt_feats = _collect_split_outputs(
        prepared=prepared,
        scalar_gop_results=scalar_gop_results,
        extract_features=extract_features,
        n_workers=n_workers,
        blank=blank,
        device=device,
        split_name=split_name,
        score_config=(score_variant, score_alpha),
    )

    logger.info(
        "[%s] %d utts, %d phones, %d skipped, %d workers",
        split_name, len(utterances),
        len(scalar_results), skipped, n_workers,
    )
    return scalar_results, utt_feats


def _get_run_mode(*, use_gopt: bool, use_feats: bool) -> str:
    if use_gopt:
        return "gopt"
    if use_feats:
        return "svr-feats"
    return "scalar"


def _safe_slug(value: str) -> str:
    return re.sub(r"[^\w\-.]", "_", value).strip("_") or "run"


def _prepare_gopt_checkpoint_dir(
    *,
    checkpoints_root: Path,
    backend_name: str,
    seed: int | None,
) -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S")
    backend_slug = _safe_slug(backend_name)
    seed_slug = f"seed{seed}" if seed is not None else "seednone"
    run_dir = checkpoints_root / f"{stamp}_{backend_slug}_{seed_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_checkpoint_run_info(
    *,
    checkpoint_dir: Path,
    backend_name: str,
    mode: str,
    seed: int | None,
    eval_result: EvalResult,
    training_history: list[dict[str, float]] | None,
) -> None:
    payload = {
        "backend": backend_name,
        "mode": mode,
        "seed": seed,
        "metrics": {
            "pcc": eval_result.pcc,
            "pcc_low": eval_result.pcc_low,
            "pcc_high": eval_result.pcc_high,
            "mse": eval_result.mse,
            "n_phones": eval_result.n_phones,
        },
        "training_history": training_history,
    }
    run_info_path = checkpoint_dir / "run_info.json"
    run_info_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _maybe_upload_checkpoint_to_hf(checkpoint_dir: Path) -> None:
    from peacock_asr.hf_checkpoints import upload_checkpoint_folder  # noqa: PLC0415
    from peacock_asr.settings import settings  # noqa: PLC0415

    if not settings.hf_checkpoint_upload:
        return
    repo_id = settings.hf_checkpoint_repo
    if repo_id is None or repo_id.strip() == "":
        logger.warning(
            "HF checkpoint upload enabled but PEACOCK_ASR_HF_CHECKPOINT_REPO is unset."
        )
        return

    subdir = settings.hf_checkpoint_repo_subdir.strip("/")
    path_in_repo = f"{subdir}/{checkpoint_dir.name}" if subdir else checkpoint_dir.name
    try:
        upload_info = upload_checkpoint_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            token=settings.hf_token,
            private_repo=True,
        )
    except Exception:
        logger.exception("Failed to upload checkpoint folder to HF Hub.")
    else:
        logger.info(
            "Uploaded checkpoint folder to HF Hub: repo=%s path=%s (%s)",
            repo_id,
            path_in_repo,
            upload_info,
        )


def _log_to_mlflow(  # noqa: PLR0913
    *,
    eval_name: str,
    mode: str,
    backend_name: str,
    device: torch.device,
    use_cache: bool,
    extract_features: bool,
    dataset_revision: str,
    workers: int,
    limit: int,
    score_variant: str,
    score_alpha: float,
    train_utterances: int,
    test_utterances: int,
    train_phones: int,
    test_phones: int,
    eval_result: EvalResult,
    seed: int | None = None,
    training_history: list[dict[str, float]] | None = None,
    reuse_active_run: bool = False,
    compute_wall_time_sec: float | None = None,
    compute_gpu_device_count: int | None = None,
) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return

    try:
        from mlflow.tracking import set_tracking_uri  # noqa: PLC0415
        from mlflow.tracking.fluent import (  # noqa: PLC0415
            log_artifact,
            log_metric,
            log_metrics,
            log_params,
            set_experiment,
            set_tags,
            start_run,
        )
    except ImportError:
        logger.warning(
            "MLFLOW_TRACKING_URI is set but mlflow is not installed. "
            "Install with: uv add mlflow",
        )
        return

    set_tracking_uri(tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "peacock-asr")
    set_experiment(experiment_name)

    run_name = f"{backend_name}-{mode}"
    run_context = nullcontext() if reuse_active_run else start_run(run_name=run_name)
    try:
        with run_context:
            set_tags(
                {
                    "project": "peacock-asr",
                    "backend": backend_name,
                    "mode": mode,
                    "eval_name": eval_name,
                    "score_variant": score_variant,
                },
            )
            log_params(
                {
                    "backend": backend_name,
                    "mode": mode,
                    "device": str(device),
                    "use_cache": use_cache,
                    "extract_features": extract_features,
                    "dataset_revision": dataset_revision,
                    "workers": workers,
                    "limit": limit,
                    "score_variant": score_variant,
                    "score_alpha": score_alpha,
                    "train_utterances": train_utterances,
                    "test_utterances": test_utterances,
                    "train_phones": train_phones,
                    "test_phones": test_phones,
                },
            )
            if seed is not None:
                log_param = {"seed": seed}
                log_params(log_param)
            log_metrics(
                {
                    "pcc": float(eval_result.pcc),
                    "pcc_low": float(eval_result.pcc_low),
                    "pcc_high": float(eval_result.pcc_high),
                    "mse": float(eval_result.mse),
                    "n_phones_eval": float(eval_result.n_phones),
                },
            )
            if compute_wall_time_sec is not None:
                wall_hours = compute_wall_time_sec / 3600.0
                gpu_count = max(compute_gpu_device_count or 0, 0)
                log_metrics(
                    {
                        "compute_wall_time_sec": float(compute_wall_time_sec),
                        "compute_wall_time_hours": float(wall_hours),
                        "compute_machine_hours": float(wall_hours),
                        "compute_gpu_hours": float(wall_hours * gpu_count),
                    },
                )

            for phone, pcc in sorted(eval_result.per_phone_pcc.items()):
                log_metric(f"per_phone_pcc.{phone}", float(pcc))

            if training_history is not None:
                for point in training_history:
                    step = int(point["epoch"])
                    log_metric("train_loss", point["loss"], step=step)
                    log_metric("train_lr", point["lr"], step=step)
                    log_metric(
                        "train_epoch_time_sec",
                        point["epoch_time_sec"],
                        step=step,
                    )

            report = {
                "backend": backend_name,
                "mode": mode,
                "dataset_revision": dataset_revision,
                "results": {
                    "pcc": eval_result.pcc,
                    "pcc_low": eval_result.pcc_low,
                    "pcc_high": eval_result.pcc_high,
                    "mse": eval_result.mse,
                    "n_phones": eval_result.n_phones,
                    "per_phone_pcc": eval_result.per_phone_pcc,
                },
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", encoding="utf-8", delete=False,
            ) as tmp:
                json.dump(report, tmp, indent=2, sort_keys=True)
                report_path = tmp.name
            try:
                log_artifact(report_path, artifact_path="reports")
            finally:
                Path(report_path).unlink(missing_ok=True)
    except Exception:
        logger.exception("Failed to log run to MLflow.")
    else:
        logger.info(
            "Logged run to MLflow: uri=%s experiment=%s run=%s",
            tracking_uri,
            experiment_name,
            run_name,
        )


def _setup_live_mlflow_run(
    *,
    tracking_uri: str | None,
    experiment_name: str,
    run_name: str,
) -> tuple[
    AbstractContextManager[object],
    bool,
    Callable[[int, int, float, float, float], None] | None,
]:
    live_mlflow_enabled = False
    live_epoch_elapsed_sec = 0.0
    on_gopt_epoch: Callable[[int, int, float, float, float], None] | None = None
    run_context: AbstractContextManager[object] = nullcontext()

    if not tracking_uri:
        return run_context, live_mlflow_enabled, on_gopt_epoch

    try:
        from mlflow.tracking import fluent, set_tracking_uri  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "MLFLOW_TRACKING_URI is set but mlflow is not installed. "
            "Install with: uv add mlflow",
        )
        return run_context, live_mlflow_enabled, on_gopt_epoch

    mlflow_log_metric = fluent.log_metric
    set_experiment = fluent.set_experiment
    start_run = fluent.start_run

    set_tracking_uri(tracking_uri)
    set_experiment(experiment_name)
    run_context = start_run(run_name=run_name)
    live_mlflow_enabled = True

    def _on_gopt_epoch(
        epoch: int,
        total_epochs: int,
        loss: float,
        lr: float,
        epoch_time_sec: float,
    ) -> None:
        nonlocal live_epoch_elapsed_sec
        live_epoch_elapsed_sec += epoch_time_sec
        avg_epoch_sec = live_epoch_elapsed_sec / max(epoch, 1)
        eta_sec = max(total_epochs - epoch, 0) * avg_epoch_sec

        mlflow_log_metric("train_loss", loss, step=epoch)
        mlflow_log_metric("train_lr", lr, step=epoch)
        mlflow_log_metric("train_epoch_time_sec", epoch_time_sec, step=epoch)
        mlflow_log_metric(
            "train_elapsed_sec", live_epoch_elapsed_sec, step=epoch,
        )
        mlflow_log_metric("train_eta_sec", eta_sec, step=epoch)
        mlflow_log_metric("train_epoch", float(epoch), step=epoch)

    on_gopt_epoch = _on_gopt_epoch
    return run_context, live_mlflow_enabled, on_gopt_epoch


def cmd_run(args: argparse.Namespace) -> tuple[str, EvalResult]:  # noqa: PLR0915
    """Run GOP-SF with a specified backend and evaluate."""
    from peacock_asr.backends import get_backend  # noqa: PLC0415
    from peacock_asr.dataset import (  # noqa: PLC0415
        DATASET_REVISION,
        load_speechocean762,
    )
    from peacock_asr.settings import settings  # noqa: PLC0415

    use_feats = getattr(args, "feats", False)
    use_gopt = getattr(args, "gopt", False)
    score_variant = getattr(args, "score_variant", "gop_sf")
    score_alpha = float(getattr(args, "score_alpha", 0.5))
    if not (0.0 <= score_alpha <= 1.0):
        msg = "score_alpha must be in [0, 1]"
        raise ValueError(msg)
    if score_variant != "gop_sf" and (use_feats or use_gopt):
        msg = (
            "score variants other than 'gop_sf' currently support scalar mode "
            "only (no --feats/--gopt)."
        )
        raise ValueError(msg)
    extract_features = use_feats or use_gopt
    use_cache = not getattr(args, "no_cache", False) and args.limit == 0
    if getattr(args, "device", None) is not None:
        settings.device = args.device
    device = settings.torch_device

    run_start = perf_counter()
    backend = get_backend(args.backend)()
    logger.info("Backend: %s", backend.name)
    backend.load()
    logger.info("Vocab size: %d  Device: %s", len(backend.vocab), device)

    if extract_features:
        logger.info(
            "Feature extraction: ON (batched ctc_loss, %d tokens)",
            len(backend.vocab),
        )

    data = load_speechocean762(limit=args.limit)

    results: dict[str, tuple[list, list]] = {}
    for split_name, utterances in [("train", data.train), ("test", data.test)]:
        split_cache_path = _cache_path(
            settings.features_dir,
            backend.name,
            split_name,
            score_variant,
            score_alpha,
        )
        cached = (
            _load_cache(
                split_cache_path,
                backend.name,
                DATASET_REVISION,
                extract_features,
            )
            if use_cache
            else None
        )
        if cached is not None:
            logger.info("[%s] Loaded from cache", split_name)
            results[split_name] = cached
        else:
            result = _process_split(
                utterances=utterances,
                split_name=split_name,
                backend=backend,
                extract_features=extract_features,
                device=device,
                score_variant=score_variant,
                score_alpha=score_alpha,
                n_workers=getattr(args, "workers", 0),
            )
            if use_cache:
                _save_cache(
                    path=split_cache_path,
                    backend_name=backend.name,
                    dataset_revision=DATASET_REVISION,
                    extract_features=extract_features,
                    scalar_results=result[0],
                    utt_feats=result[1],
                )
            results[split_name] = result

    train_scalar, train_utts = results["train"]
    test_scalar, test_utts = results["test"]

    mode_name = _get_run_mode(use_gopt=use_gopt, use_feats=use_feats)
    run_seed = getattr(args, "seed", None)
    checkpoint_dir: Path | None = None
    if use_gopt:
        checkpoint_dir = _prepare_gopt_checkpoint_dir(
            checkpoints_root=settings.checkpoints_dir,
            backend_name=backend.name,
            seed=run_seed,
        )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "peacock-asr")
    mlflow_run_context, live_mlflow_enabled, on_gopt_epoch = _setup_live_mlflow_run(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=f"{backend.name}-{mode_name}",
    )

    with mlflow_run_context:
        eval_name, eval_result, training_history = _run_evaluation(
            use_gopt=use_gopt,
            use_feats=use_feats,
            backend_name=backend.name,
            feat_dim=len(backend.vocab) + 2,
            device=device,
            train_scalar=train_scalar,
            test_scalar=test_scalar,
            train_utts=train_utts,
            test_utts=test_utts,
            verbose=args.verbose,
            seed=run_seed,
            checkpoint_dir=checkpoint_dir,
            on_gopt_epoch=on_gopt_epoch,
        )

        _log_to_mlflow(
            eval_name=eval_name,
            mode=mode_name,
            backend_name=backend.name,
            device=device,
            use_cache=use_cache,
            extract_features=extract_features,
            dataset_revision=DATASET_REVISION,
            workers=getattr(args, "workers", 0),
            limit=getattr(args, "limit", 0),
            score_variant=score_variant,
            score_alpha=score_alpha,
            train_utterances=len(data.train),
            test_utterances=len(data.test),
            train_phones=len(train_scalar),
            test_phones=len(test_scalar),
            eval_result=eval_result,
            seed=run_seed,
            training_history=None if live_mlflow_enabled else training_history,
            reuse_active_run=live_mlflow_enabled,
            compute_wall_time_sec=perf_counter() - run_start,
            compute_gpu_device_count=(
                _cuda_device_count() if device.type == "cuda" else 0
            ),
        )
    if checkpoint_dir is not None:
        _write_checkpoint_run_info(
            checkpoint_dir=checkpoint_dir,
            backend_name=backend.name,
            mode=mode_name,
            seed=run_seed,
            eval_result=eval_result,
            training_history=training_history,
        )
        _maybe_upload_checkpoint_to_hf(checkpoint_dir)
    return eval_name, eval_result


def _run_evaluation(  # noqa: PLR0913
    *,
    use_gopt: bool,
    use_feats: bool,
    backend_name: str,
    feat_dim: int,
    device: torch.device,
    train_scalar: list[tuple[str, float, float]],
    test_scalar: list[tuple[str, float, float]],
    train_utts: list[UtteranceFeats],
    test_utts: list[UtteranceFeats],
    verbose: bool,
    seed: int | None = None,
    checkpoint_dir: Path | None = None,
    on_gopt_epoch: Callable[[int, int, float, float, float], None] | None = None,
) -> tuple[str, EvalResult, list[dict[str, float]] | None]:
    """Dispatch to the appropriate evaluation method."""
    from peacock_asr.evaluate import (  # noqa: PLC0415
        EvalResult,
        evaluate_gop,
        evaluate_gop_feats,
    )

    training_history: list[dict[str, float]] | None = None
    if use_gopt:
        from peacock_asr.gopt_model import (  # noqa: PLC0415
            train_and_evaluate_gopt,
        )

        observed_feat_dims = {
            len(vec)
            for utt in [*train_utts, *test_utts]
            for vec in utt.feat_vecs
        }
        if len(observed_feat_dims) != 1:
            msg = (
                "Expected exactly one feature width across train/test utterances, "
                f"got {sorted(observed_feat_dims)}"
            )
            raise RuntimeError(msg)
        observed_feat_dim = next(iter(observed_feat_dims))
        if observed_feat_dim != feat_dim:
            logger.warning(
                "Feature width mismatch: configured feat_dim=%d, observed=%d. "
                "Using observed width.",
                feat_dim, observed_feat_dim,
            )
        feat_dim = observed_feat_dim

        logger.info(
            "Training GOPT transformer (feat_dim=%d)...", feat_dim,
        )
        training_history = []
        eval_result = train_and_evaluate_gopt(
            train_utts, test_utts,
            input_dim=feat_dim,
            device=device,
            seed=seed,
            history_out=training_history,
            checkpoint_dir=checkpoint_dir,
            on_epoch_end=on_gopt_epoch,
        )
        eval_name = backend_name + " (GOPT)"
        _print_results(eval_name, eval_result, verbose)
    elif use_feats:
        # Flatten utterance records into per-phone tuples for SVR
        train_flat = [
            (p, f, s)
            for u in train_utts
            for p, f, s in zip(
                u.phones, u.feat_vecs, u.scores, strict=True,
            )
        ]
        test_flat = [
            (p, f, s)
            for u in test_utts
            for p, f, s in zip(
                u.phones, u.feat_vecs, u.scores, strict=True,
            )
        ]
        logger.info("Evaluating with SVR on feature vectors...")
        eval_result = evaluate_gop_feats(train_flat, test_flat)
        eval_name = backend_name + " (SVR+feats)"
        _print_results(eval_name, eval_result, verbose)
    else:
        logger.info(
            "Evaluating with poly regression on scalar scores...",
        )
        eval_result = evaluate_gop(train_scalar, test_scalar)
        eval_name = backend_name
        _print_results(eval_name, eval_result, verbose)

    if not isinstance(eval_result, EvalResult):
        msg = "Unexpected evaluation result type."
        raise TypeError(msg)
    return eval_name, eval_result, training_history


def _print_results(
    name: str,
    result: object,
    verbose: bool,  # noqa: FBT001
) -> None:
    from peacock_asr.evaluate import EvalResult  # noqa: PLC0415

    if not isinstance(result, EvalResult):
        return
    sep = "=" * 50
    print(f"\n{sep}")  # noqa: T201
    print(f"Backend:    {name}")  # noqa: T201
    print(f"Phones:     {result.n_phones}")  # noqa: T201
    print(f"PCC:        {result.pcc:.4f}")  # noqa: T201
    lo, hi = result.pcc_low, result.pcc_high
    print(f"PCC 95% CI: [{lo:.4f}, {hi:.4f}]")  # noqa: T201
    print(f"MSE:        {result.mse:.4f}")  # noqa: T201
    print(sep)  # noqa: T201

    if verbose:
        print("\nPer-phone PCC:")  # noqa: T201
        items = sorted(result.per_phone_pcc.items(), key=lambda x: -x[1])
        for phone, pcc in items:
            print(f"  {phone:4s}  {pcc:.4f}")  # noqa: T201


def cmd_compare(args: argparse.Namespace) -> None:
    """Run all available backends and compare."""
    from peacock_asr.backends import BACKEND_REGISTRY  # noqa: PLC0415

    available = list(BACKEND_REGISTRY.keys())
    print(f"Available backends: {', '.join(available)}")  # noqa: T201
    print()  # noqa: T201

    for name in available:
        sep = "=" * 50
        print(sep)  # noqa: T201
        print(f"Running: {name}")  # noqa: T201
        print(sep)  # noqa: T201
        run_args = argparse.Namespace(backend=name, limit=args.limit, verbose=False)
        try:
            cmd_run(run_args)
        except Exception as e:  # noqa: BLE001
            print(f"  Skipped {name}: {type(e).__name__}: {e}")  # noqa: T201
        print()  # noqa: T201


def _write_batch_summary(rows: list[dict[str, str]], path: Path) -> None:
    cols = [
        "job_id",
        "backend",
        "mode",
        "score_variant",
        "score_alpha",
        "repeat",
        "seed",
        "status",
        "pcc",
        "pcc_low",
        "pcc_high",
        "mse",
        "n_phones",
        "duration_sec",
        "log_path",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_batch_aggregates(rows: list[dict[str, str]], path: Path) -> None:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            str(row["job_id"]),
            str(row["backend"]),
            str(row["mode"]),
            str(row["score_variant"]),
            str(row["score_alpha"]),
        )
        groups.setdefault(key, []).append(row)

    cols = [
        "job_id",
        "backend",
        "mode",
        "score_variant",
        "score_alpha",
        "n_runs",
        "n_success",
        "pcc_mean",
        "pcc_std",
        "mse_mean",
        "mse_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for (
            job_id, backend, mode, score_variant, score_alpha
        ), grouped_rows in groups.items():
            ok_rows = [r for r in grouped_rows if r["status"] == "ok"]
            pcc_vals = [float(r["pcc"]) for r in ok_rows]
            mse_vals = [float(r["mse"]) for r in ok_rows]
            if len(pcc_vals) == 0:
                pcc_mean = float("nan")
                pcc_std = float("nan")
                mse_mean = float("nan")
                mse_std = float("nan")
            elif len(pcc_vals) == 1:
                pcc_mean = pcc_vals[0]
                pcc_std = 0.0
                mse_mean = mse_vals[0]
                mse_std = 0.0
            else:
                pcc_mean = mean(pcc_vals)
                pcc_std = stdev(pcc_vals)
                mse_mean = mean(mse_vals)
                mse_std = stdev(mse_vals)
            writer.writerow(
                {
                    "job_id": job_id,
                    "backend": backend,
                    "mode": mode,
                    "score_variant": score_variant,
                    "score_alpha": score_alpha,
                    "n_runs": len(grouped_rows),
                    "n_success": len(ok_rows),
                    "pcc_mean": f"{pcc_mean:.4f}" if pcc_vals else "nan",
                    "pcc_std": f"{pcc_std:.4f}" if pcc_vals else "nan",
                    "mse_mean": f"{mse_mean:.4f}" if pcc_vals else "nan",
                    "mse_std": f"{mse_std:.4f}" if pcc_vals else "nan",
                }
            )


def _run_batch_repeat(
    *,
    run_count: int,
    run_dir: Path,
    job: BatchResolvedJob,
    repeat: int,
    seed: int | None,
) -> tuple[dict[str, str], bool]:
    use_gopt = job.mode == "gopt"
    use_feats = job.mode == "feats"
    mode_name = _get_run_mode(use_gopt=use_gopt, use_feats=use_feats)

    run_tag = f"{job.job_id}_r{repeat}"
    log_path = run_dir / f"{run_tag}.log"
    logger.info(
        "[batch] (%d) start job=%s backend=%s mode=%s variant=%s alpha=%.4f "
        "repeat=%d seed=%s",
        run_count,
        job.job_id,
        job.backend,
        job.mode,
        job.score_variant,
        job.score_alpha,
        repeat,
        seed if seed is not None else "none",
    )

    run_args = argparse.Namespace(
        backend=job.backend,
        feats=use_feats,
        gopt=use_gopt,
        device=job.device,
        limit=job.limit,
        no_cache=job.no_cache,
        workers=job.workers,
        seed=seed,
        score_variant=job.score_variant,
        score_alpha=job.score_alpha,
        verbose=job.verbose,
    )

    started = perf_counter()
    status = "ok"
    error = ""
    pcc = float("nan")
    pcc_low = float("nan")
    pcc_high = float("nan")
    mse = float("nan")
    n_phones = 0

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            "batch_job="
            f"{job.job_id} repeat={repeat} backend={job.backend} mode={job.mode} "
            f"score_variant={job.score_variant} "
            f"score_alpha={job.score_alpha:.4f} "
            f"seed={seed if seed is not None else 'none'}\n"
        )
        log_file.flush()
        try:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                _, eval_result = cmd_run(run_args)
        except Exception as e:  # noqa: BLE001
            status = "failed"
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc(file=log_file)
        else:
            pcc = float(eval_result.pcc)
            pcc_low = float(eval_result.pcc_low)
            pcc_high = float(eval_result.pcc_high)
            mse = float(eval_result.mse)
            n_phones = int(eval_result.n_phones)

    duration_sec = perf_counter() - started
    row = {
        "job_id": job.job_id,
        "backend": job.backend,
        "mode": mode_name,
        "score_variant": job.score_variant,
        "score_alpha": f"{job.score_alpha:.4f}",
        "repeat": str(repeat),
        "seed": str(seed) if seed is not None else "",
        "status": status,
        "pcc": f"{pcc:.4f}" if status == "ok" else "nan",
        "pcc_low": f"{pcc_low:.4f}" if status == "ok" else "nan",
        "pcc_high": f"{pcc_high:.4f}" if status == "ok" else "nan",
        "mse": f"{mse:.4f}" if status == "ok" else "nan",
        "n_phones": str(n_phones) if status == "ok" else "",
        "duration_sec": f"{duration_sec:.2f}",
        "log_path": str(log_path),
        "error": error,
    }

    logger.info(
        "[batch] done job=%s repeat=%d status=%s pcc=%s mse=%s",
        job.job_id,
        repeat,
        status,
        row["pcc"],
        row["mse"],
    )
    return row, status != "ok"


def cmd_batch(args: argparse.Namespace) -> None:
    """Run a YAML-defined batch of runs."""
    from peacock_asr.batch_config import (  # noqa: PLC0415
        BatchCliDefaults,
        load_batch_spec,
        resolve_batch_jobs,
    )

    spec_path = Path(args.config)
    spec = load_batch_spec(spec_path)
    cli_defaults = BatchCliDefaults(
        device=args.device,
        limit=args.limit,
        workers=args.workers,
        no_cache=args.no_cache,
        verbose=args.verbose,
        score_variant=args.score_variant,
        score_alpha=args.score_alpha,
    )
    jobs = resolve_batch_jobs(spec, cli_defaults=cli_defaults)

    batch_name = spec.name
    safe_batch_name = re.sub(r"[^\w\-.]", "_", batch_name).strip("_") or "batch"
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S")
    output_root = Path(args.output_dir)
    run_dir = output_root / f"{stamp}_{safe_batch_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_spec_copy = run_dir / "batch_spec.yaml"
    run_spec_copy.write_text(spec_path.read_text(encoding="utf-8"), encoding="utf-8")

    rows: list[dict[str, str]] = []
    total_failed = 0
    run_count = 0

    for job in jobs:
        for rep, seed in enumerate(job.seeds, start=1):
            run_count += 1
            row, failed = _run_batch_repeat(
                run_count=run_count,
                run_dir=run_dir,
                job=job,
                repeat=rep,
                seed=seed,
            )
            rows.append(row)
            if failed:
                total_failed += 1

    summary_path = run_dir / "summary.tsv"
    aggregate_path = run_dir / "aggregates.tsv"
    _write_batch_summary(rows, summary_path)
    _write_batch_aggregates(rows, aggregate_path)
    logger.info("[batch] summary: %s", summary_path)
    logger.info("[batch] aggregates: %s", aggregate_path)

    if total_failed > 0:
        logger.error("[batch] %d run(s) failed", total_failed)
        sys.exit(1)


def _run_train_profile(profile_name: str) -> None:
    import subprocess  # noqa: PLC0415

    from peacock_asr.settings import settings  # noqa: PLC0415
    from peacock_asr.train_profile import load_train_profile  # noqa: PLC0415

    profile = load_train_profile(profile_name)
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "training" / "train_phoneme_head.py"
    if not script_path.exists():
        msg = f"Training script not found: {script_path}"
        raise FileNotFoundError(msg)

    cmd = [
        "uv",
        "run",
        "python",
        str(script_path),
        "--output-dir",
        profile.output_dir,
        "--num-epochs",
        str(profile.num_epochs),
        "--batch-size",
        str(profile.batch_size),
        "--gradient-accumulation",
        str(profile.gradient_accumulation),
        "--learning-rate",
        str(profile.learning_rate),
        "--eval-split",
        profile.eval_split,
        "--dataloader-workers",
        str(profile.dataloader_workers),
        "--train-splits",
        *profile.train_splits,
    ]
    if profile.max_train_samples is not None:
        cmd.extend(["--max-train-samples", str(profile.max_train_samples)])
    if profile.max_eval_samples is not None:
        cmd.extend(["--max-eval-samples", str(profile.max_eval_samples)])
    if profile.push_to_hub:
        cmd.extend(["--hub-repo", settings.hf_train_repo])
    else:
        cmd.append("--no-push")

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri
    env["MLFLOW_EXPERIMENT_NAME"] = settings.mlflow_experiment_name
    env["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = str(
        settings.mlflow_enable_system_metrics_logging,
    ).lower()
    env["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = str(
        settings.mlflow_system_metrics_sampling_interval,
    )
    env["MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING"] = str(
        settings.mlflow_system_metrics_samples_before_logging,
    )
    env.setdefault("UV_CACHE_DIR", str(repo_root / ".cache" / "uv"))
    src_path = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
    )
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token

    logger.info(
        "[train] profile=%s mlflow=%s experiment=%s",
        profile.name,
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment_name,
    )
    logger.info("[train] cmd=%s", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)  # noqa: S603


def cmd_train_preflight(_args: argparse.Namespace) -> None:
    _run_train_profile("preflight")


def cmd_train_main(_args: argparse.Namespace) -> None:
    _run_train_profile("main")


def cmd_papers_convert(args: argparse.Namespace) -> None:
    from peacock_asr.papers.convert import (  # noqa: PLC0415
        ConvertConfig,
        convert_papers,
    )

    config = ConvertConfig(
        papers_root=Path(args.root),
        folder=args.folder,
        force=args.force,
        strict=args.strict,
        min_words=args.min_words,
        timeout_seconds=args.timeout,
        report_path=Path(args.report) if args.report else None,
    )
    convert_papers(config)


def main() -> None:  # noqa: PLR0915
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        prog="peacock-asr",
        description="Pronunciation assessment benchmark",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download SpeechOcean762 dataset")

    run_p = sub.add_parser("run", help="Run GOP-SF with a backend")
    run_p.add_argument(
        "--backend",
        "-b",
        required=True,
        help="Backend: original, xlsr-espeak, zipa",
    )
    run_p.add_argument(
        "--feats",
        action="store_true",
        help="Extract full feature vectors (LPP+LPR) and evaluate with SVR",
    )
    run_p.add_argument(
        "--gopt",
        action="store_true",
        help="Train GOPT transformer on feature vectors and evaluate",
    )
    run_p.add_argument(
        "--device",
        default=None,
        help="Compute device: auto, cpu, cuda (default: auto)",
    )
    run_p.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Limit utterances per split (0 = all)",
    )
    run_p.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip feature cache (always re-extract)",
    )
    run_p.add_argument(
        "--workers",
        "-w",
        type=int,
        default=0,
        help="Parallel workers for GOP computation (0 = auto, 1 = sequential)",
    )
    run_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed (used by GOPT training path)",
    )
    run_p.add_argument(
        "--score-variant",
        choices=["gop_sf", "logit_margin", "logit_combined"],
        default="gop_sf",
        help=(
            "Scalar score variant. Non-default variants currently support "
            "scalar mode only."
        ),
    )
    run_p.add_argument(
        "--score-alpha",
        type=float,
        default=0.5,
        help="Mixture weight for logit_combined in [0, 1] (default: 0.5).",
    )

    cmp_p = sub.add_parser("compare", help="Run all backends and compare")
    cmp_p.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Limit utterances per split (0 = all)",
    )

    batch_p = sub.add_parser(
        "batch", help="Run a YAML-defined experiment batch"
    )
    batch_p.add_argument(
        "--config",
        default="runs/batch.yaml",
        help="Batch YAML config path (default: runs/batch.yaml)",
    )
    batch_p.add_argument(
        "--output-dir",
        default="runs",
        help="Directory where batch run folders are written (default: runs)",
    )
    batch_p.add_argument(
        "--device",
        default=None,
        help="Default device override for jobs (default: use settings)",
    )
    batch_p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Default utterance limit for jobs (0 = all)",
    )
    batch_p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Default workers for jobs (0 = auto)",
    )
    batch_p.add_argument(
        "--no-cache",
        action="store_true",
        help="Default no-cache behavior for jobs",
    )
    batch_p.add_argument(
        "--score-variant",
        choices=["gop_sf", "logit_margin", "logit_combined"],
        default="gop_sf",
        help="Default score variant for jobs (default: gop_sf)",
    )
    batch_p.add_argument(
        "--score-alpha",
        type=float,
        default=0.5,
        help="Default score alpha for jobs (default: 0.5)",
    )

    sub.add_parser(
        "train-preflight",
        help="Run fixed preflight training profile (tiny smoke run)",
    )
    sub.add_parser(
        "train-main",
        help="Run fixed main training profile",
    )

    papers_p = sub.add_parser("papers", help="Paper ingestion tools")
    papers_sub = papers_p.add_subparsers(dest="papers_command", required=True)

    papers_convert_p = papers_sub.add_parser(
        "convert",
        help="Convert papers to markdown with quality checks",
    )
    papers_convert_p.add_argument(
        "--root",
        default="docs/papers",
        help="Root papers directory (default: docs/papers)",
    )
    papers_convert_p.add_argument(
        "--folder",
        default=None,
        help="Optional subfolder under --root to process",
    )
    papers_convert_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .md files",
    )
    papers_convert_p.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail command when quality checks report warnings (default: true)",
    )
    papers_convert_p.add_argument(
        "--min-words",
        type=int,
        default=700,
        help="Minimum word count threshold for quality checks",
    )
    papers_convert_p.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout for arXiv HTML fetches, in seconds",
    )
    papers_convert_p.add_argument(
        "--report",
        default=None,
        help="Optional path for conversion JSON report",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Keep third-party HTTP libraries quiet even in -v mode; debug headers
    # can include temporary auth details from remote services.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    match args.command:
        case "download":
            cmd_download(args)
        case "run":
            cmd_run(args)
        case "compare":
            cmd_compare(args)
        case "batch":
            cmd_batch(args)
        case "train-preflight":
            cmd_train_preflight(args)
        case "train-main":
            cmd_train_main(args)
        case "papers":
            if args.papers_command == "convert":
                cmd_papers_convert(args)
            else:
                parser.print_help()
                sys.exit(1)
        case _:
            parser.print_help()
            sys.exit(1)
