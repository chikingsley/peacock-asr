"""CLI entry points for peacock-asr."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

    from peacock_asr.backends.base import PhonemeBackend
    from peacock_asr.dataset import Utterance
    from peacock_asr.evaluate import EvalResult
    from peacock_asr.gop import GOPResult, ScalarGOPResult
    from peacock_asr.gopt_model import UtteranceFeats

logger = logging.getLogger("peacock_asr")


def cmd_download(_args: argparse.Namespace) -> None:
    """Download the SpeechOcean762 dataset."""
    from peacock_asr.dataset import load_speechocean762  # noqa: PLC0415

    data = load_speechocean762()
    print(f"Train: {len(data.train)} utterances")  # noqa: T201
    print(f"Test:  {len(data.test)} utterances")  # noqa: T201
    print("Dataset cached by HuggingFace datasets.")  # noqa: T201


def _cache_path(
    features_dir: Path, backend_name: str, split: str,
) -> Path:
    """Build the cache file path for a given backend and split."""
    safe_name = re.sub(r"[^\w\-.]", "_", backend_name)
    d = features_dir / safe_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{split}.pt"


def _load_cache(
    features_dir: Path,
    backend_name: str,
    split: str,
    dataset_revision: str,
    extract_features: bool,  # noqa: FBT001
) -> tuple[list[tuple[str, float, float]], list[UtteranceFeats]] | None:
    """Load cached features if valid. Returns None on miss."""
    import torch  # noqa: PLC0415

    path = _cache_path(features_dir, backend_name, split)
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
    features_dir: Path,
    backend_name: str,
    split: str,
    dataset_revision: str,
    extract_features: bool,
    scalar_results: list[tuple[str, float, float]],
    utt_feats: list[UtteranceFeats],
) -> None:
    """Save extraction results to cache."""
    import torch  # noqa: PLC0415

    path = _cache_path(features_dir, backend_name, split)
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
    logger.info("[%s] Saved cache to %s", split, path)


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
) -> tuple[list[tuple[str, float, float]], list[UtteranceFeats]]:
    from tqdm import tqdm  # noqa: PLC0415

    from peacock_asr.gop import (  # noqa: PLC0415
        GOPResult,
        ScalarGOPResult,
        compute_gop_features,
    )
    from peacock_asr.gopt_model import UtteranceFeats  # noqa: PLC0415

    scalar_results: list[tuple[str, float, float]] = []
    utt_feats: list[UtteranceFeats] = []
    iterator = (
        tqdm(prepared, desc=f"{split_name} [collect]", unit="utt")
        if extract_features and n_workers > 1
        else prepared
    )

    for idx, (post, pidx, valid_phones, valid_scores) in enumerate(iterator):
        gop = scalar_gop_results[idx]

        for phone, gop_score, human_score in zip(
            valid_phones, gop.scores, valid_scores, strict=True,
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
    train_utterances: int,
    test_utterances: int,
    train_phones: int,
    test_phones: int,
    eval_result: EvalResult,
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
    try:
        with start_run(run_name=run_name):
            set_tags(
                {
                    "project": "peacock-asr",
                    "backend": backend_name,
                    "mode": mode,
                    "eval_name": eval_name,
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
                    "train_utterances": train_utterances,
                    "test_utterances": test_utterances,
                    "train_phones": train_phones,
                    "test_phones": test_phones,
                },
            )
            log_metrics(
                {
                    "pcc": float(eval_result.pcc),
                    "pcc_low": float(eval_result.pcc_low),
                    "pcc_high": float(eval_result.pcc_high),
                    "mse": float(eval_result.mse),
                    "n_phones_eval": float(eval_result.n_phones),
                },
            )

            for phone, pcc in sorted(eval_result.per_phone_pcc.items()):
                log_metric(f"per_phone_pcc.{phone}", float(pcc))

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


def cmd_run(args: argparse.Namespace) -> None:
    """Run GOP-SF with a specified backend and evaluate."""
    from peacock_asr.backends import get_backend  # noqa: PLC0415
    from peacock_asr.dataset import (  # noqa: PLC0415
        DATASET_REVISION,
        load_speechocean762,
    )
    from peacock_asr.settings import settings  # noqa: PLC0415

    use_feats = getattr(args, "feats", False)
    use_gopt = getattr(args, "gopt", False)
    extract_features = use_feats or use_gopt
    use_cache = not getattr(args, "no_cache", False) and args.limit == 0
    if getattr(args, "device", None) is not None:
        settings.device = args.device
    device = settings.torch_device

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
        cached = (
            _load_cache(
                settings.features_dir, backend.name, split_name,
                DATASET_REVISION, extract_features,
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
                n_workers=getattr(args, "workers", 0),
            )
            if use_cache:
                _save_cache(
                    features_dir=settings.features_dir,
                    backend_name=backend.name,
                    split=split_name,
                    dataset_revision=DATASET_REVISION,
                    extract_features=extract_features,
                    scalar_results=result[0],
                    utt_feats=result[1],
                )
            results[split_name] = result

    train_scalar, train_utts = results["train"]
    test_scalar, test_utts = results["test"]

    eval_name, eval_result = _run_evaluation(
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
    )

    _log_to_mlflow(
        eval_name=eval_name,
        mode=_get_run_mode(use_gopt=use_gopt, use_feats=use_feats),
        backend_name=backend.name,
        device=device,
        use_cache=use_cache,
        extract_features=extract_features,
        dataset_revision=DATASET_REVISION,
        workers=getattr(args, "workers", 0),
        limit=getattr(args, "limit", 0),
        train_utterances=len(data.train),
        test_utterances=len(data.test),
        train_phones=len(train_scalar),
        test_phones=len(test_scalar),
        eval_result=eval_result,
    )


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
) -> tuple[str, EvalResult]:
    """Dispatch to the appropriate evaluation method."""
    from peacock_asr.evaluate import (  # noqa: PLC0415
        EvalResult,
        evaluate_gop,
        evaluate_gop_feats,
    )

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
        eval_result = train_and_evaluate_gopt(
            train_utts, test_utts,
            input_dim=feat_dim,
            device=device,
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
    return eval_name, eval_result


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


def main() -> None:
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

    cmp_p = sub.add_parser("compare", help="Run all backends and compare")
    cmp_p.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Limit utterances per split (0 = all)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    match args.command:
        case "download":
            cmd_download(args)
        case "run":
            cmd_run(args)
        case "compare":
            cmd_compare(args)
        case _:
            parser.print_help()
            sys.exit(1)
