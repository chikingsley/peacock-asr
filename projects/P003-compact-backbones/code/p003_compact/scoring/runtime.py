"""Scoring runtime for pronunciation evaluation commands."""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypedDict, cast

from p003_compact.scoring.prepared_bundle import (
    PreparedItem,
    load_prepared_bundle,
    save_prepared_bundle,
)
from p003_compact.settings import settings

if TYPE_CHECKING:
    import argparse

    import numpy as np
    import torch

    from p003_compact.backend_protocol import PhonemeBackend
    from p003_compact.dataset import Utterance
    from p003_compact.evaluate import EvalResult
    from p003_compact.gop import GOPResult, ScalarGOPResult
    from p003_compact.gopt_model import UtteranceFeats
    from p003_compact.score_variants import ScoreVariant
    from wandb import Run

logger = logging.getLogger("p003_compact")
WANDB_TAG_MAX_LEN = 64
try:
    UTC = dt.UTC
except AttributeError:
    UTC = dt.timezone(dt.timedelta())
_POSTERIOR_NDIM = 2
_BatchScalarK2Fn = Callable[
    ...,
    list[tuple[float, list[float], list[float]]],
]


class _WandbInitKwargs(TypedDict, total=False):
    entity: str
    project: str
    name: str
    tags: list[str]
    config: dict[str, Any]
    group: str
    job_type: str


@dataclass(frozen=True)
class _EvalWandbPayload:
    eval_name: str
    eval_result: EvalResult
    backend_name: str
    mode: str
    score_variant: str
    score_alpha: float
    seed: int | None
    duration_s: float
    training_history: list[dict[str, float]] | None
    dataset_revision: str
    feature_dim: int
    backend_vocab_size: int
    device_name: str
    use_cache: bool
    cache_status: dict[str, bool]
    train_utterance_count: int
    test_utterance_count: int
    workers: int
    limit: int
    checkpoint_dir: Path | None


@dataclass(frozen=True)
class _AlphaSweepWandbPayload:
    backend_name: str
    source_alpha: float
    alphas: list[float]
    rows: list[dict[str, str]]
    best_alpha: float
    best_result: EvalResult
    dataset_revision: str
    summary_path: Path
    metadata_path: Path


def _cuda_device_count() -> int:
    import torch  # noqa: PLC0415

    return torch.cuda.device_count()


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


def _cache_dir(
    features_dir: Path,
    backend_name: str,
) -> Path:
    """Return the backend-specific cache directory."""
    safe_name = re.sub(r"[^\w\-.]", "_", backend_name)
    return features_dir / safe_name


def _prepared_cache_path(
    features_dir: Path,
    backend_name: str,
    split: str,
    transport_dtype: str,
) -> Path:
    """Return the prepared-input cache bundle path for one backend split."""
    safe_name = re.sub(r"[^\w\-.]", "_", backend_name)
    dtype_slug = transport_dtype.lower()
    return features_dir / safe_name / "_prepared" / f"{split}_{dtype_slug}.bundle"


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


def _load_cache_any_alpha(
    *,
    features_dir: Path,
    backend_name: str,
    split: str,
    score_variant: ScoreVariant,
    dataset_revision: str,
    extract_features: bool,
    preferred_alpha: float | None = None,
) -> tuple[list[tuple[str, float, float]], list[UtteranceFeats]] | None:
    """Load a compatible cache, preferring an exact alpha match when present."""
    if preferred_alpha is not None:
        preferred = _load_cache(
            _cache_path(
                features_dir,
                backend_name,
                split,
                score_variant,
                preferred_alpha,
            ),
            backend_name,
            dataset_revision,
            extract_features,
        )
        if preferred is not None:
            return preferred

    pattern = f"{score_variant}_a*/{split}.pt"
    for path in sorted(_cache_dir(features_dir, backend_name).glob(pattern)):
        cached = _load_cache(path, backend_name, dataset_revision, extract_features)
        if cached is not None:
            return cached
    return None


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
    from p003_compact.gop import compute_gop_scalar_scores_only  # noqa: PLC0415

    posteriors, phone_indices, blank = args
    return compute_gop_scalar_scores_only(posteriors, phone_indices, blank)


def _scalar_gop_worker_memmap(
    args: tuple[str, int, int, list[int], int],
) -> ScalarGOPResult:
    """Worker that loads a posterior slice from a split-level `.npy` memmap."""
    import numpy as np  # noqa: PLC0415

    from p003_compact.gop import compute_gop_scalar  # noqa: PLC0415

    posterior_path, start_row, end_row, phone_indices, blank = args
    posteriors_all = np.load(
        posterior_path,
        allow_pickle=False,
        mmap_mode="c",
    )
    return compute_gop_scalar(posteriors_all[start_row:end_row], phone_indices, blank)


def _prepared_memmap_spec(posteriors: np.ndarray) -> tuple[str, int, int] | None:
    """Return the backing file and row slice for a memmapped posterior view."""
    import numpy as np  # noqa: PLC0415

    candidate: object | None = posteriors
    memmap_base: np.memmap | None = None
    while candidate is not None:
        if isinstance(candidate, np.memmap):
            memmap_base = candidate
            break
        candidate = getattr(candidate, "base", None)

    if memmap_base is None:
        return None
    filename = getattr(memmap_base, "filename", None)
    if filename is None:
        return None
    if posteriors.ndim != _POSTERIOR_NDIM or memmap_base.ndim != _POSTERIOR_NDIM:
        return None

    base_ptr = int(np.asarray(memmap_base).__array_interface__["data"][0])
    view_ptr = int(np.asarray(posteriors).__array_interface__["data"][0])
    row_width = int(posteriors.shape[1]) * int(posteriors.dtype.itemsize)
    if row_width == 0:
        return str(filename), 0, 0
    byte_offset = view_ptr - base_ptr
    start_row = byte_offset // row_width
    end_row = start_row + int(posteriors.shape[0])
    return str(filename), start_row, end_row


def _prepare_split_inputs(
    *,
    utterances: list[Utterance],
    split_name: str,
    backend: PhonemeBackend,
) -> tuple[list[PreparedItem], int]:
    """Collect posterior matrices and aligned phone targets for a split."""
    from tqdm import tqdm  # noqa: PLC0415

    prepared: list[PreparedItem] = []
    skipped = 0
    batch_size = max(1, settings.ctc_posterior_batch_size)
    batch_getter = getattr(backend, "get_posteriors_batch", None)
    use_batched_posteriors = callable(batch_getter) and batch_size > 1

    pending_audios: list[np.ndarray] = []
    pending_rates: list[int] = []
    pending_meta: list[tuple[list[int], list[str], list[float]]] = []

    progress = tqdm(utterances, desc=f"{split_name} [posteriors]", unit="utt")

    def flush_pending() -> None:
        if not pending_meta:
            return
        if use_batched_posteriors:
            batch_getter_fn = cast(
                "Callable[[list[np.ndarray], list[int]], list[np.ndarray]]",
                batch_getter,
            )
            batch_posteriors = batch_getter_fn(pending_audios, pending_rates)
        else:
            batch_posteriors = [
                backend.get_posteriors(audio, sample_rate)
                for audio, sample_rate in zip(
                    pending_audios,
                    pending_rates,
                    strict=True,
                )
            ]
        for posteriors, (phone_indices, valid_phones, valid_scores) in zip(
            batch_posteriors,
            pending_meta,
            strict=True,
        ):
            prepared.append((posteriors, phone_indices, valid_phones, valid_scores))
        pending_audios.clear()
        pending_rates.clear()
        pending_meta.clear()

    for utt in progress:
        phone_indices: list[int] = []
        valid_phones: list[str] = []
        valid_scores: list[float] = []

        for phone, score in zip(
            utt.phones,
            utt.phone_scores,
            strict=True,
        ):
            indices = backend.map_phone(phone)
            if indices is not None:
                phone_indices.extend(indices)
                valid_phones.append(phone)
                valid_scores.append(score)

        if len(phone_indices) < 2:  # noqa: PLR2004
            skipped += 1
            continue

        pending_audios.append(utt.audio)
        pending_rates.append(utt.sample_rate)
        pending_meta.append((phone_indices, valid_phones, valid_scores))
        if len(pending_meta) >= batch_size:
            flush_pending()

    flush_pending()

    return prepared, skipped


def _resolve_scalar_worker_count(
    requested_workers: int,
) -> int:
    if requested_workers > 0:
        return requested_workers

    import os  # noqa: PLC0415

    return max(1, (os.cpu_count() or 2) - 2)


def _compute_scalar_gop_phase(
    *,
    prepared: list[PreparedItem],
    split_name: str,
    blank: int,
    extract_features: bool,
    device: torch.device,
    n_workers: int,
) -> list[ScalarGOPResult | GOPResult]:
    """Run the scalar GOP phase for a prepared split."""
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415

    from tqdm import tqdm  # noqa: PLC0415

    from p003_compact.gop import (  # noqa: PLC0415
        compute_gop,
        compute_gop_scalar_scores_only,
    )
    from p003_compact.k2_scalar import (  # noqa: PLC0415
        compute_scalar_terms_k2_batch,
        k2_available,
    )
    from p003_compact.settings import settings  # noqa: PLC0415

    n_workers = _resolve_scalar_worker_count(n_workers)

    scalar_gop_results: list[ScalarGOPResult | GOPResult]
    use_parallel_scalar = n_workers > 1 and len(prepared) > 1
    if settings.ctc_scalar_backend == "k2":
        use_parallel_scalar = False
        logger.info(
            "[%s] Computing scalar GOP serially because ctc_scalar_backend='k2'.",
            split_name,
        )

    if (
        extract_features
        and settings.ctc_scalar_backend == "k2"
        and k2_available()
        and len(prepared) > 1
    ):
        batch_utts = max(1, settings.ctc_scalar_batch_utterances)
        logger.info(
            "[%s] Computing scalar GOP with batched k2 (%d utts/batch)...",
            split_name,
            batch_utts,
        )
        return cast(
            "list[ScalarGOPResult | GOPResult]",
            _compute_scalar_gop_phase_k2_batched(
                prepared=prepared,
                split_name=split_name,
                blank=blank,
                batch_utts=batch_utts,
                compute_scalar_terms_k2_batch=compute_scalar_terms_k2_batch,
            ),
        )

    if use_parallel_scalar:
        memmap_worker_args = [
            (path, start_row, end_row, pidx, blank)
            for post, pidx, _, _ in prepared
            if (spec := _prepared_memmap_spec(post)) is not None
            for path, start_row, end_row in [spec]
        ]
        use_memmap_handoff = len(memmap_worker_args) == len(prepared)
        logger.info(
            "[%s] Computing scalar GOP with %d workers%s...",
            split_name,
            n_workers,
            " via memmap handoff" if use_memmap_handoff else "",
        )
        pending_scalar_results: list[ScalarGOPResult | None] = [None for _ in prepared]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            if use_memmap_handoff:
                worker_args = memmap_worker_args
                futures = {
                    pool.submit(_scalar_gop_worker_memmap, args): index
                    for index, args in enumerate(worker_args)
                }
            else:
                worker_args = [(post, pidx, blank) for post, pidx, _, _ in prepared]
                futures = {
                    pool.submit(_scalar_gop_worker, args): index
                    for index, args in enumerate(worker_args)
                }
            progress = tqdm(
                total=len(worker_args),
                desc=f"{split_name} [GOP]",
                unit="utt",
            )
            try:
                for future in as_completed(futures):
                    index = futures[future]
                    pending_scalar_results[index] = future.result()
                    progress.update(1)
            finally:
                progress.close()
        scalar_gop_results = [
            result for result in pending_scalar_results if result is not None
        ]
    else:
        scalar_gop_results = []
        for post, pidx, _, _ in tqdm(
            prepared,
            desc=f"{split_name} [GOP]",
            unit="utt",
        ):
            if extract_features:
                scalar_gop_results.append(
                    compute_gop(
                        posteriors=post,
                        phone_indices=pidx,
                        blank=blank,
                        extract_features=True,
                        device=device,
                    )
                )
            else:
                scalar_gop_results.append(
                    compute_gop_scalar_scores_only(post, pidx, blank),
                )

    return scalar_gop_results


def _compute_scalar_gop_phase_k2_batched(
    *,
    prepared: list[PreparedItem],
    split_name: str,
    blank: int,
    batch_utts: int,
    compute_scalar_terms_k2_batch: _BatchScalarK2Fn,
) -> list[ScalarGOPResult]:
    """Compute scalar GOP with k2, recursively splitting oversized batches."""
    from p003_compact.gop import ScalarGOPResult  # noqa: PLC0415
    from p003_compact.settings import settings  # noqa: PLC0415

    prepared_chunks = _partition_prepared_for_k2(
        prepared=prepared,
        max_utts=batch_utts,
        max_phone_positions=max(1, settings.ctc_scalar_batch_phone_positions),
        max_case_frame_budget=max(1, settings.ctc_scalar_batch_case_frame_budget),
    )
    logger.info(
        "[%s] k2 batch planner produced %d chunks "
        "(max_utts=%d, max_phone_positions=%d, max_case_frame_budget=%d).",
        split_name,
        len(prepared_chunks),
        batch_utts,
        settings.ctc_scalar_batch_phone_positions,
        settings.ctc_scalar_batch_case_frame_budget,
    )

    def run_chunk(
        chunk: list[PreparedItem],
    ) -> list[ScalarGOPResult]:
        chunk_cases = [(post, pidx) for post, pidx, _, _ in chunk]
        try:
            chunk_results = compute_scalar_terms_k2_batch(
                chunk_cases,
                blank=blank,
                device=settings.scalar_torch_device,
            )
        except RuntimeError as exc:
            if len(chunk) == 1:
                raise
            split_at = max(1, len(chunk) // 2)
            logger.warning(
                "[%s] k2 scalar batch of %d utterances failed (%s); "
                "retrying as %d + %d.",
                split_name,
                len(chunk),
                exc.__class__.__name__,
                split_at,
                len(chunk) - split_at,
            )
            return [*run_chunk(chunk[:split_at]), *run_chunk(chunk[split_at:])]

        return [
            ScalarGOPResult(
                ll_self=ll_self,
                scores=scores,
                occupancies=occupancies,
            )
            for ll_self, scores, occupancies in chunk_results
        ]

    results: list[ScalarGOPResult] = []
    for chunk in prepared_chunks:
        results.extend(run_chunk(chunk))
    return results


def _partition_prepared_for_k2(
    *,
    prepared: list[PreparedItem],
    max_utts: int,
    max_phone_positions: int,
    max_case_frame_budget: int,
) -> list[list[PreparedItem]]:
    """Greedily partition prepared utterances by estimated k2 workload."""
    chunks: list[list[PreparedItem]] = []
    current_chunk: list[PreparedItem] = []
    current_phone_positions = 0
    current_max_frames = 0

    for item in prepared:
        posteriors, phone_indices, _, _ = item
        item_frames = int(posteriors.shape[0])
        item_phone_positions = len(phone_indices)

        if not current_chunk:
            current_chunk = [item]
            current_phone_positions = item_phone_positions
            current_max_frames = item_frames
            continue

        next_phone_positions = current_phone_positions + item_phone_positions
        next_max_frames = max(current_max_frames, item_frames)
        next_case_frame_budget = next_phone_positions * next_max_frames

        if (
            len(current_chunk) >= max_utts
            or next_phone_positions > max_phone_positions
            or next_case_frame_budget > max_case_frame_budget
        ):
            chunks.append(current_chunk)
            current_chunk = [item]
            current_phone_positions = item_phone_positions
            current_max_frames = item_frames
            continue

        current_chunk.append(item)
        current_phone_positions = next_phone_positions
        current_max_frames = next_max_frames

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _collect_split_outputs(
    *,
    prepared: list[PreparedItem],
    scalar_gop_results: list[ScalarGOPResult | GOPResult],
    extract_features: bool,
    n_workers: int,
    blank: int,
    device: torch.device,
    split_name: str,
    score_config: tuple[ScoreVariant, float],
) -> tuple[list[tuple[str, float, float]], list[UtteranceFeats]]:
    from tqdm import tqdm  # noqa: PLC0415

    from p003_compact.gop import (  # noqa: PLC0415
        GOPResult,
        ScalarGOPResult,
        compute_gop_features,
    )
    from p003_compact.gopt_model import UtteranceFeats  # noqa: PLC0415
    from p003_compact.score_variants import apply_score_variant  # noqa: PLC0415

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
            valid_phones,
            adjusted_scores,
            valid_scores,
            strict=True,
        ):
            scalar_results.append((phone, gop_score, human_score))

        if not extract_features:
            continue

        if isinstance(gop, GOPResult):
            if gop.features is None:
                msg = "Expected feature vectors in sequential mode."
                raise RuntimeError(msg)
            features = gop.features
        else:
            if not isinstance(gop, ScalarGOPResult):
                msg = "Expected ScalarGOPResult in parallel mode."
                raise TypeError(msg)
            features = compute_gop_features(
                post,
                pidx,
                gop.ll_self,
                blank,
                device=device,
            )

        feat_vecs = [
            [*features[i].tolist(), gop.occupancies[i]]
            for i in range(len(valid_phones))
        ]
        utt_feats.append(
            UtteranceFeats(
                phones=valid_phones,
                feat_vecs=feat_vecs,
                scores=valid_scores,
            )
        )

    return scalar_results, utt_feats


def _load_prepared_cache(
    *,
    features_dir: Path,
    backend_name: str,
    split_name: str,
    dataset_revision: str,
    transport_dtype: str,
) -> tuple[list[PreparedItem], int] | None:
    """Load a prepared-input cache bundle for one split."""
    bundle = load_prepared_bundle(
        _prepared_cache_path(
            features_dir,
            backend_name,
            split_name,
            transport_dtype,
        )
    )
    if bundle is None:
        return None

    meta, splits = bundle
    if meta.get("backend") != backend_name:
        return None
    if meta.get("dataset_revision") != dataset_revision:
        return None
    if str(meta.get("transport_dtype")) != transport_dtype:
        return None

    prepared = splits.get(split_name)
    if prepared is None:
        return None
    skipped = meta.get("skipped", 0)
    skipped_count = int(skipped) if isinstance(skipped, int | float | str) else 0
    return prepared, skipped_count


def _save_prepared_cache(
    *,
    features_dir: Path,
    backend_name: str,
    split_name: str,
    dataset_revision: str,
    transport_dtype: str,
    prepared: list[PreparedItem],
    skipped: int,
) -> None:
    """Persist prepared split inputs as a memmap-friendly bundle."""
    save_prepared_bundle(
        _prepared_cache_path(
            features_dir,
            backend_name,
            split_name,
            transport_dtype,
        ),
        {split_name: prepared},
        {
            "backend": backend_name,
            "dataset_revision": dataset_revision,
            "transport_dtype": transport_dtype,
            "skipped": skipped,
        },
    )


def _process_split(
    *,
    utterances: list[Utterance],
    split_name: str,
    backend: PhonemeBackend,
    extract_features: bool,
    device: torch.device,
    score_config: tuple[ScoreVariant, float],
    use_cache: bool,
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
    # -- Phase 1: Collect posteriors (GPU, sequential) --
    from p003_compact.dataset import DATASET_REVISION  # noqa: PLC0415

    transport_dtype = settings.ctc_posterior_transport_dtype.lower()
    score_variant, score_alpha = score_config
    effective_workers = _resolve_scalar_worker_count(n_workers)
    prepared_cache = None
    if use_cache:
        prepared_cache = _load_prepared_cache(
            features_dir=settings.features_dir,
            backend_name=backend.name,
            split_name=split_name,
            dataset_revision=DATASET_REVISION,
            transport_dtype=transport_dtype,
        )

    if prepared_cache is not None:
        prepared, skipped = prepared_cache
        logger.info("[%s] Loaded prepared-input cache", split_name)
    else:
        prepared, skipped = _prepare_split_inputs(
            utterances=utterances,
            split_name=split_name,
            backend=backend,
        )
        if use_cache:
            _save_prepared_cache(
                features_dir=settings.features_dir,
                backend_name=backend.name,
                split_name=split_name,
                dataset_revision=DATASET_REVISION,
                transport_dtype=transport_dtype,
                prepared=prepared,
                skipped=skipped,
            )
            if effective_workers > 1 and len(prepared) > 1:
                reloaded = _load_prepared_cache(
                    features_dir=settings.features_dir,
                    backend_name=backend.name,
                    split_name=split_name,
                    dataset_revision=DATASET_REVISION,
                    transport_dtype=transport_dtype,
                )
                if reloaded is not None:
                    prepared, skipped = reloaded

    if (
        settings.ctc_scalar_backend == "k2"
        and settings.scalar_torch_device.type == "cuda"
    ):
        backend.unload()

    # -- Phase 2: Scalar GOP (CPU, parallel) --
    blank = backend.blank_index
    scalar_gop_results = _compute_scalar_gop_phase(
        prepared=prepared,
        split_name=split_name,
        blank=blank,
        extract_features=extract_features,
        device=device,
        n_workers=effective_workers,
    )

    scalar_results, utt_feats = _collect_split_outputs(
        prepared=prepared,
        scalar_gop_results=scalar_gop_results,
        extract_features=extract_features,
        n_workers=effective_workers,
        blank=blank,
        device=device,
        split_name=split_name,
        score_config=(score_variant, score_alpha),
    )

    logger.info(
        "[%s] %d utts, %d phones, %d skipped, %d workers",
        split_name,
        len(utterances),
        len(scalar_results),
        skipped,
        effective_workers,
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
    stamp = dt.datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S")
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


def _repo_root() -> Path | None:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def _git_commit_sha() -> str | None:
    repo_root = _repo_root()
    if repo_root is None:
        return None
    git_bin = shutil.which("git")
    if git_bin is None:
        return None
    result = subprocess.run(  # noqa: S603
        [git_bin, "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _log_directory_artifact(
    *,
    run: Run,
    directory: Path,
    artifact_name: str,
    artifact_type: str,
    metadata: dict[str, Any],
) -> bool:
    import wandb  # noqa: PLC0415

    if not directory.exists():
        return False
    files = sorted(path for path in directory.iterdir() if path.is_file())
    if not files:
        return False
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata,
    )
    for path in files:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    return True


def _import_wandb() -> Any | None:
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        logger.warning("wandb not installed — skipping W&B logging")
        return None
    return wandb


def _wandb_tags(*base_tags: str | None) -> list[str]:
    from p003_compact.settings import settings  # noqa: PLC0415

    extra_tags_raw = settings.wandb_tags
    extra_tags = [tag.strip() for tag in extra_tags_raw.split(",") if tag.strip()]
    tags: list[str] = []
    for raw_tag in [*base_tags, *extra_tags]:
        if not raw_tag:
            continue
        sanitized = _safe_slug(raw_tag)
        if len(sanitized) > WANDB_TAG_MAX_LEN:
            sanitized = sanitized[:WANDB_TAG_MAX_LEN].rstrip("_-.")
        if sanitized:
            tags.append(sanitized)
    return list(dict.fromkeys(tags))


def _wandb_init_kwargs(
    *,
    run_name: str,
    group: str,
    job_type: str,
    tags: list[str],
    config: dict[str, Any],
) -> _WandbInitKwargs:
    from p003_compact.settings import settings  # noqa: PLC0415

    init_kwargs: _WandbInitKwargs = {
        "group": group,
        "job_type": job_type,
        "name": run_name,
        "tags": tags,
        "config": config,
    }
    if not settings.wandb_sweep_id:
        init_kwargs["project"] = settings.wandb_project
        init_kwargs["entity"] = settings.wandb_entity
    return init_kwargs


def _wandb_run_name(*parts: str | None, seed: int | None = None) -> str:
    run_name = " | ".join(part for part in parts if part)
    if seed is not None:
        return f"{run_name} | seed={seed}"
    return run_name


def _with_git_sha(config: dict[str, Any]) -> dict[str, Any]:
    git_sha = _git_commit_sha()
    if git_sha is not None:
        config["git_sha"] = git_sha
    return config


def _cache_hit_counts(cache_status: dict[str, bool]) -> tuple[int, int]:
    cache_hits = sum(1 for hit in cache_status.values() if hit)
    cache_misses = len(cache_status) - cache_hits
    return cache_hits, cache_misses


def _build_eval_wandb_config(
    payload: _EvalWandbPayload,
    *,
    group: str,
    job_type: str,
    cache_hits: int,
    cache_misses: int,
) -> dict[str, Any]:
    from p003_compact.settings import settings  # noqa: PLC0415

    return _with_git_sha(
        {
            "track": settings.wandb_track,
            "project_id": settings.wandb_project_id,
            "phase": settings.wandb_phase,
            "job_id": settings.wandb_job_id,
            "backend": payload.backend_name,
            "backend_vocab_size": payload.backend_vocab_size,
            "mode": payload.mode,
            "score_variant": payload.score_variant,
            "score_alpha": payload.score_alpha,
            "seed": payload.seed,
            "dataset_revision": payload.dataset_revision,
            "feature_dim": payload.feature_dim,
            "device": payload.device_name,
            "use_cache": payload.use_cache,
            "cache_train_hit": payload.cache_status.get("train"),
            "cache_test_hit": payload.cache_status.get("test"),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "train_utterances": payload.train_utterance_count,
            "test_utterances": payload.test_utterance_count,
            "workers": payload.workers,
            "limit": payload.limit,
            "wandb_group": group,
            "wandb_job_type": job_type,
            "wandb_sweep_id": settings.wandb_sweep_id,
            "checkpoint_dir": (
                None if payload.checkpoint_dir is None else str(payload.checkpoint_dir)
            ),
        },
    )


def _build_eval_wandb_tags(payload: _EvalWandbPayload) -> list[str]:
    from p003_compact.settings import settings  # noqa: PLC0415

    return _wandb_tags(
        settings.wandb_project_id.lower(),
        settings.wandb_track,
        payload.backend_name,
        payload.mode,
        payload.score_variant,
        settings.wandb_phase,
        settings.wandb_job_id,
    )


def _log_eval_summary(
    run: Run,
    *,
    eval_result: EvalResult,
    duration_s: float,
    cache_hits: int,
    cache_misses: int,
) -> None:
    run.summary["pcc"] = eval_result.pcc
    run.summary["pcc_low"] = eval_result.pcc_low
    run.summary["pcc_high"] = eval_result.pcc_high
    run.summary["mse"] = eval_result.mse
    run.summary["n_phones"] = eval_result.n_phones
    run.summary["duration_s"] = duration_s
    run.summary["cache_hits"] = cache_hits
    run.summary["cache_misses"] = cache_misses


def _log_per_phone_pcc_table(
    run: Run,
    wandb: Any,
    per_phone_pcc: dict[str, float],
    *,
    step: int | None = None,
) -> None:
    if not per_phone_pcc:
        return

    phone_table = wandb.Table(columns=["phone", "pcc"])
    for phone, pcc in sorted(per_phone_pcc.items(), key=lambda item: -item[1]):
        phone_table.add_data(phone, pcc)
    if step is None:
        run.log({"per_phone_pcc": phone_table})
    else:
        run.log({"per_phone_pcc": phone_table}, step=step)


def _log_training_history_to_wandb(
    run: Run,
    training_history: list[dict[str, float]] | None,
) -> int:
    if not training_history:
        return 0

    for step, entry in enumerate(training_history):
        run.log({f"gopt/{key}": value for key, value in entry.items()}, step=step)
    final_entry = training_history[-1]
    run.summary["gopt_epochs"] = len(training_history)
    for key, value in final_entry.items():
        if key == "epoch":
            continue
        run.summary[f"gopt_final_{key}"] = value
    return len(training_history)


def _log_eval_checkpoint_artifact(
    run: Run,
    payload: _EvalWandbPayload,
) -> None:
    from p003_compact.settings import settings  # noqa: PLC0415

    checkpoint_dir = payload.checkpoint_dir
    if checkpoint_dir is None:
        return

    artifact_job = settings.wandb_job_id or _safe_slug(payload.mode)
    artifact_phase = settings.wandb_phase or "phase"
    seed_slug = "none" if payload.seed is None else str(payload.seed)
    artifact_name = (
        f"{settings.wandb_project_id.lower()}-"
        f"{_safe_slug(artifact_phase)}-"
        f"{_safe_slug(artifact_job)}-"
        f"{_safe_slug(payload.backend_name)}-"
        f"seed{seed_slug}"
    )
    artifact_metadata = {
        "track": settings.wandb_track,
        "project_id": settings.wandb_project_id,
        "phase": settings.wandb_phase,
        "job_id": settings.wandb_job_id,
        "backend": payload.backend_name,
        "mode": payload.mode,
        "score_variant": payload.score_variant,
        "score_alpha": payload.score_alpha,
        "seed": payload.seed,
        "dataset_revision": payload.dataset_revision,
    }
    logged_artifact = _log_directory_artifact(
        run=run,
        directory=checkpoint_dir,
        artifact_name=artifact_name,
        artifact_type="model",
        metadata=artifact_metadata,
    )
    if logged_artifact:
        run.summary["checkpoint_artifact"] = f"{artifact_name}:latest"


def _build_alpha_wandb_config(
    payload: _AlphaSweepWandbPayload,
    *,
    group: str,
    job_type: str,
) -> dict[str, Any]:
    from p003_compact.settings import settings  # noqa: PLC0415

    return _with_git_sha(
        {
            "track": settings.wandb_track,
            "project_id": settings.wandb_project_id,
            "phase": settings.wandb_phase,
            "job_id": settings.wandb_job_id,
            "backend": payload.backend_name,
            "mode": "alpha_sweep",
            "source_alpha": payload.source_alpha,
            "alphas": payload.alphas,
            "dataset_revision": payload.dataset_revision,
            "summary_path": str(payload.summary_path),
            "metadata_path": str(payload.metadata_path),
            "wandb_group": group,
            "wandb_job_type": job_type,
        },
    )


def _build_alpha_wandb_tags(payload: _AlphaSweepWandbPayload) -> list[str]:
    from p003_compact.settings import settings  # noqa: PLC0415

    return _wandb_tags(
        settings.wandb_project_id.lower(),
        settings.wandb_track,
        payload.backend_name,
        "alpha-sweep",
        settings.wandb_phase,
        settings.wandb_job_id,
    )


def _log_alpha_sweep_table(
    run: Run,
    wandb: Any,
    rows: list[dict[str, str]],
) -> None:
    table = wandb.Table(
        columns=["alpha", "pcc", "pcc_low", "pcc_high", "mse", "n_phones"],
    )
    for row in rows:
        table.add_data(
            float(row["alpha"]),
            float(row["pcc"]),
            float(row["pcc_low"]),
            float(row["pcc_high"]),
            float(row["mse"]),
            int(row["n_phones"]),
        )
    run.log({"alpha_sweep": table})


def _log_alpha_summary(
    run: Run,
    payload: _AlphaSweepWandbPayload,
) -> None:
    run.summary["best_alpha"] = payload.best_alpha
    run.summary["best_pcc"] = payload.best_result.pcc
    run.summary["best_mse"] = payload.best_result.mse
    run.summary["best_pcc_low"] = payload.best_result.pcc_low
    run.summary["best_pcc_high"] = payload.best_result.pcc_high
    run.summary["n_alphas"] = len(payload.alphas)


def _log_alpha_artifact(
    run: Run,
    payload: _AlphaSweepWandbPayload,
) -> None:
    from p003_compact.settings import settings  # noqa: PLC0415

    artifact_phase = settings.wandb_phase or "phase"
    artifact_job = settings.wandb_job_id or "alpha-sweep"
    artifact_name = (
        f"{settings.wandb_project_id.lower()}-"
        f"{_safe_slug(artifact_phase)}-"
        f"{_safe_slug(artifact_job)}-"
        f"{_safe_slug(payload.backend_name)}"
    )
    artifact_metadata = {
        "track": settings.wandb_track,
        "project_id": settings.wandb_project_id,
        "phase": settings.wandb_phase,
        "job_id": settings.wandb_job_id,
        "backend": payload.backend_name,
        "source_alpha": payload.source_alpha,
        "best_alpha": payload.best_alpha,
        "best_pcc": payload.best_result.pcc,
        "best_mse": payload.best_result.mse,
        "dataset_revision": payload.dataset_revision,
    }
    logged_artifact = _log_directory_artifact(
        run=run,
        directory=payload.summary_path.parent,
        artifact_name=artifact_name,
        artifact_type="analysis",
        metadata=artifact_metadata,
    )
    if logged_artifact:
        run.summary["analysis_artifact"] = f"{artifact_name}:latest"


def _maybe_upload_checkpoint_to_hf(checkpoint_dir: Path) -> None:
    from p003_compact.hf_checkpoints import upload_checkpoint_folder  # noqa: PLC0415
    from p003_compact.settings import settings  # noqa: PLC0415

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


def _log_eval_to_wandb(payload: _EvalWandbPayload) -> None:
    """Log evaluation results to W&B."""
    from p003_compact.settings import settings  # noqa: PLC0415

    wandb = _import_wandb()
    if wandb is None or settings.wandb_mode == "disabled":
        return

    group = settings.wandb_group or f"eval-{payload.score_variant}"
    job_type = settings.wandb_job_type
    run_prefix = settings.wandb_run_prefix
    cache_hits, cache_misses = _cache_hit_counts(payload.cache_status)
    run_name = _wandb_run_name(
        run_prefix,
        settings.wandb_phase,
        settings.wandb_job_id,
        payload.eval_name,
        seed=payload.seed,
    )
    config = _build_eval_wandb_config(
        payload,
        group=group,
        job_type=job_type,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )
    tags = _build_eval_wandb_tags(payload)
    run = wandb.init(
        **_wandb_init_kwargs(
            run_name=run_name,
            group=group,
            job_type=job_type,
            tags=tags,
            config=config,
        ),
    )

    _log_eval_summary(
        run,
        eval_result=payload.eval_result,
        duration_s=payload.duration_s,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )
    final_step = _log_training_history_to_wandb(run, payload.training_history)
    _log_per_phone_pcc_table(
        run,
        wandb,
        payload.eval_result.per_phone_pcc,
        step=final_step,
    )
    _log_eval_checkpoint_artifact(run, payload)

    wandb.finish()


def _log_alpha_sweep_to_wandb(payload: _AlphaSweepWandbPayload) -> None:
    from p003_compact.settings import settings  # noqa: PLC0415

    wandb = _import_wandb()
    if wandb is None or settings.wandb_mode == "disabled":
        return

    group = settings.wandb_group or f"alpha-sweep-{payload.backend_name}"
    job_type = settings.wandb_job_type
    run_prefix = settings.wandb_run_prefix
    run_name = _wandb_run_name(
        run_prefix,
        settings.wandb_phase,
        settings.wandb_job_id,
        f"{payload.backend_name} alpha sweep",
    )
    config = _build_alpha_wandb_config(payload, group=group, job_type=job_type)
    tags = _build_alpha_wandb_tags(payload)
    run = wandb.init(
        **_wandb_init_kwargs(
            run_name=run_name,
            group=group,
            job_type=job_type,
            tags=tags,
            config=config,
        ),
    )

    _log_alpha_sweep_table(run, wandb, payload.rows)
    _log_alpha_summary(run, payload)
    _log_alpha_artifact(run, payload)
    wandb.finish()


def cmd_run(args: argparse.Namespace) -> tuple[str, EvalResult]:  # noqa: PLR0912,PLR0915
    """Run GOP-SF with a specified backend and evaluate."""
    from p003_compact.backend_loader import get_backend  # noqa: PLC0415
    from p003_compact.dataset import (  # noqa: PLC0415
        DATASET_REVISION,
        load_speechocean762,
    )
    from p003_compact.settings import settings  # noqa: PLC0415

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
    cache_status: dict[str, bool] = {}
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
        if (
            cached is None
            and use_cache
            and not extract_features
            and score_variant == "logit_combined"
        ):
            base_cached = _load_cache_any_alpha(
                features_dir=settings.features_dir,
                backend_name=backend.name,
                split=split_name,
                score_variant="gop_sf",
                dataset_revision=DATASET_REVISION,
                extract_features=False,
                preferred_alpha=0.5,
            )
            margin_cached = _load_cache_any_alpha(
                features_dir=settings.features_dir,
                backend_name=backend.name,
                split=split_name,
                score_variant="logit_margin",
                dataset_revision=DATASET_REVISION,
                extract_features=False,
                preferred_alpha=0.5,
            )
            if base_cached is not None and margin_cached is not None:
                logger.info(
                    "[%s] Synthesizing logit_combined cache from gop_sf + logit_margin",
                    split_name,
                )
                cached = (
                    _mix_scalar_results(
                        base=base_cached[0],
                        margin=margin_cached[0],
                        alpha=score_alpha,
                    ),
                    [],
                )
                _save_cache(
                    path=split_cache_path,
                    backend_name=backend.name,
                    dataset_revision=DATASET_REVISION,
                    extract_features=False,
                    scalar_results=cached[0],
                    utt_feats=[],
                )
        if cached is not None:
            logger.info("[%s] Loaded from cache", split_name)
            results[split_name] = cached
            cache_status[split_name] = True
        else:
            result = _process_split(
                utterances=utterances,
                split_name=split_name,
                backend=backend,
                extract_features=extract_features,
                device=device,
                score_config=(score_variant, score_alpha),
                use_cache=use_cache,
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
            cache_status[split_name] = False

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

    feature_dim = 1
    if extract_features:
        observed_feat_dims = {
            len(vec) for utt in [*train_utts, *test_utts] for vec in utt.feat_vecs
        }
        if observed_feat_dims:
            feature_dim = next(iter(observed_feat_dims))

    # --- Log to W&B ---
    _log_eval_to_wandb(
        _EvalWandbPayload(
            eval_name=eval_name,
            eval_result=eval_result,
            backend_name=backend.name,
            mode=mode_name,
            score_variant=score_variant,
            score_alpha=score_alpha,
            seed=run_seed,
            duration_s=perf_counter() - run_start,
            training_history=training_history,
            dataset_revision=DATASET_REVISION,
            feature_dim=feature_dim,
            backend_vocab_size=len(backend.vocab),
            device_name=str(device),
            use_cache=use_cache,
            cache_status=cache_status,
            train_utterance_count=len(data.train),
            test_utterance_count=len(data.test),
            workers=getattr(args, "workers", 0),
            limit=args.limit,
            checkpoint_dir=checkpoint_dir,
        ),
    )

    return eval_name, eval_result


def cmd_prewarm_k2(args: argparse.Namespace) -> None:
    """Populate prepared-input and k2 topology caches without running eval."""
    from p003_compact.backend_loader import get_backend  # noqa: PLC0415
    from p003_compact.dataset import (  # noqa: PLC0415
        DATASET_REVISION,
        load_speechocean762,
    )
    from p003_compact.k2_scalar import (  # noqa: PLC0415
        k2_available,
        prewarm_topology_cache,
        topology_cache_dir,
    )

    if getattr(args, "device", None) is not None:
        settings.device = args.device

    if not k2_available():
        msg = "k2 is not installed in this environment."
        raise RuntimeError(msg)

    backend = get_backend(args.backend)()
    logger.info("Backend: %s", backend.name)
    backend.load()
    logger.info(
        "Prewarming k2 topology cache (split=%s, limit=%d, posterior_batch_size=%d)",
        args.split,
        args.limit,
        settings.ctc_posterior_batch_size,
    )

    data = load_speechocean762(limit=args.limit)
    split_map = {
        "train": data.train,
        "test": data.test,
        "both": None,
    }
    split_names = ["train", "test"] if args.split == "both" else [args.split]
    total_cache_hits = 0
    total_cache_misses = 0
    total_unique_topologies = 0
    total_cases = 0

    for split_name in split_names:
        utterances = split_map[split_name]
        if utterances is None:
            msg = f"Unsupported split: {split_name}"
            raise ValueError(msg)

        transport_dtype = settings.ctc_posterior_transport_dtype.lower()
        prepared_cache = _load_prepared_cache(
            features_dir=settings.features_dir,
            backend_name=backend.name,
            split_name=split_name,
            dataset_revision=DATASET_REVISION,
            transport_dtype=transport_dtype,
        )
        if prepared_cache is not None:
            prepared, skipped = prepared_cache
            logger.info("[%s] Loaded prepared-input cache", split_name)
        else:
            prepared, skipped = _prepare_split_inputs(
                utterances=utterances,
                split_name=split_name,
                backend=backend,
            )
            _save_prepared_cache(
                features_dir=settings.features_dir,
                backend_name=backend.name,
                split_name=split_name,
                dataset_revision=DATASET_REVISION,
                transport_dtype=transport_dtype,
                prepared=prepared,
                skipped=skipped,
            )
            logger.info("[%s] Saved prepared-input cache", split_name)

        blank = backend.blank_index
        start = perf_counter()
        stats = prewarm_topology_cache(prepared, blank=blank)
        duration = perf_counter() - start
        total_cache_hits += stats["cache_hits"]
        total_cache_misses += stats["cache_misses"]
        total_unique_topologies += stats["unique_topologies"]
        total_cases += stats["total_cases"]
        logger.info(
            "[%s] k2 prewarm complete in %.2fs | prepared=%d skipped=%d "
            "cases=%d unique_topologies=%d hits=%d misses=%d cache_files=%d",
            split_name,
            duration,
            stats["prepared_utterances"],
            skipped,
            stats["total_cases"],
            stats["unique_topologies"],
            stats["cache_hits"],
            stats["cache_misses"],
            stats["cache_files"],
        )

    logger.info(
        "k2 prewarm complete | splits=%s cases=%d unique_topologies=%d "
        "hits=%d misses=%d cache_dir=%s",
        ",".join(split_names),
        total_cases,
        total_unique_topologies,
        total_cache_hits,
        total_cache_misses,
        topology_cache_dir(),
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
    seed: int | None = None,
    checkpoint_dir: Path | None = None,
    on_gopt_epoch: Callable[[int, int, float, float, float], None] | None = None,
) -> tuple[str, EvalResult, list[dict[str, float]] | None]:
    """Dispatch to the appropriate evaluation method."""
    from p003_compact.evaluate import (  # noqa: PLC0415
        EvalResult,
        evaluate_gop,
        evaluate_gop_feats,
    )

    training_history: list[dict[str, float]] | None = None
    if use_gopt:
        from p003_compact.gopt_model import (  # noqa: PLC0415
            train_and_evaluate_gopt,
        )

        observed_feat_dims = {
            len(vec) for utt in [*train_utts, *test_utts] for vec in utt.feat_vecs
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
                feat_dim,
                observed_feat_dim,
            )
        feat_dim = observed_feat_dim

        logger.info(
            "Training GOPT transformer (feat_dim=%d)...",
            feat_dim,
        )
        training_history = []
        eval_result = train_and_evaluate_gopt(
            train_utts,
            test_utts,
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
                u.phones,
                u.feat_vecs,
                u.scores,
                strict=True,
            )
        ]
        test_flat = [
            (p, f, s)
            for u in test_utts
            for p, f, s in zip(
                u.phones,
                u.feat_vecs,
                u.scores,
                strict=True,
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
    from p003_compact.evaluate import EvalResult  # noqa: PLC0415

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


def _parse_alpha_values(
    *,
    alphas_arg: str | None,
    start: float,
    stop: float,
    step: float,
) -> list[float]:
    if alphas_arg:
        alphas: list[float] = []
        for token in alphas_arg.split(","):
            value = float(token.strip())
            if not (0.0 <= value <= 1.0):
                msg = f"alpha value out of range [0, 1]: {value}"
                raise ValueError(msg)
            alphas.append(value)
        if len(alphas) == 0:
            msg = "--alphas did not contain any values"
            raise ValueError(msg)
        return sorted(set(alphas))

    if step <= 0.0:
        msg = "alpha step must be > 0"
        raise ValueError(msg)
    if start > stop:
        msg = "alpha start must be <= alpha stop"
        raise ValueError(msg)
    if not (0.0 <= start <= 1.0 and 0.0 <= stop <= 1.0):
        msg = "alpha range must be within [0, 1]"
        raise ValueError(msg)

    alphas = []
    current = start
    epsilon = 1e-12
    while current <= stop + epsilon:
        alphas.append(round(current, 10))
        current += step
    return alphas


def _mix_scalar_results(
    *,
    base: list[tuple[str, float, float]],
    margin: list[tuple[str, float, float]],
    alpha: float,
) -> list[tuple[str, float, float]]:
    if len(base) != len(margin):
        msg = (
            "Cache mismatch: baseline and margin scalar result lengths differ "
            f"({len(base)} vs {len(margin)})."
        )
        raise ValueError(msg)

    out: list[tuple[str, float, float]] = []
    label_tolerance = 1e-6
    for index, (b_row, m_row) in enumerate(zip(base, margin, strict=True)):
        b_phone, b_score, b_label = b_row
        m_phone, m_score, m_label = m_row
        if b_phone != m_phone:
            msg = (
                "Cache mismatch: phone order differs at index "
                f"{index}: {b_phone} != {m_phone}"
            )
            raise ValueError(msg)
        if abs(b_label - m_label) > label_tolerance:
            msg = (
                "Cache mismatch: human label differs at index "
                f"{index}: {b_label} != {m_label}"
            )
            raise ValueError(msg)
        mixed_score = (1.0 - alpha) * b_score + alpha * m_score
        out.append((b_phone, float(mixed_score), b_label))
    return out


def cmd_sweep_alpha(args: argparse.Namespace) -> None:  # noqa: PLR0915
    """Fast scalar alpha sweep from cached GOP-SF and logit-margin scores."""
    from p003_compact.backend_loader import resolve_backend_name  # noqa: PLC0415
    from p003_compact.dataset import DATASET_REVISION  # noqa: PLC0415
    from p003_compact.evaluate import evaluate_gop  # noqa: PLC0415
    from p003_compact.settings import settings  # noqa: PLC0415

    backend_input = args.backend
    try:
        backend_name = resolve_backend_name(backend_input)
    except Exception:  # noqa: BLE001
        backend_name = backend_input
    source_alpha = float(args.source_alpha)
    if not (0.0 <= source_alpha <= 1.0):
        msg = "source_alpha must be in [0, 1]"
        raise ValueError(msg)

    alphas = _parse_alpha_values(
        alphas_arg=args.alphas,
        start=float(args.alpha_start),
        stop=float(args.alpha_stop),
        step=float(args.alpha_step),
    )

    def _load_scalar_or_fail(
        split: str,
        variant: ScoreVariant,
    ) -> list[tuple[str, float, float]]:
        cached = _load_cache_any_alpha(
            features_dir=settings.features_dir,
            backend_name=backend_name,
            split=split,
            score_variant=variant,
            dataset_revision=DATASET_REVISION,
            extract_features=False,
            preferred_alpha=source_alpha,
        )
        if cached is None:
            msg = (
                "Missing compatible scalar cache for "
                f"backend={backend_name} split={split} variant={variant} "
                f"alpha={source_alpha:.4f}. Expected cache under: "
                f"{_cache_dir(settings.features_dir, backend_name)}"
            )
            raise FileNotFoundError(msg)
        return cached[0]

    train_gop = _load_scalar_or_fail("train", "gop_sf")
    test_gop = _load_scalar_or_fail("test", "gop_sf")
    train_margin = _load_scalar_or_fail("train", "logit_margin")
    test_margin = _load_scalar_or_fail("test", "logit_margin")

    stamp = dt.datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{stamp}_alpha_sweep_{_safe_slug(backend_name)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    best_alpha = None
    best_result = None

    for alpha in alphas:
        train_mix = _mix_scalar_results(
            base=train_gop,
            margin=train_margin,
            alpha=alpha,
        )
        test_mix = _mix_scalar_results(
            base=test_gop,
            margin=test_margin,
            alpha=alpha,
        )
        result = evaluate_gop(train_mix, test_mix)
        logger.info(
            "[sweep-alpha] alpha=%.4f pcc=%.4f mse=%.4f n_phones=%d",
            alpha,
            result.pcc,
            result.mse,
            result.n_phones,
        )
        rows.append(
            {
                "alpha": f"{alpha:.4f}",
                "pcc": f"{result.pcc:.4f}",
                "pcc_low": f"{result.pcc_low:.4f}",
                "pcc_high": f"{result.pcc_high:.4f}",
                "mse": f"{result.mse:.4f}",
                "n_phones": str(result.n_phones),
            },
        )
        if best_result is None or result.pcc > best_result.pcc:
            best_alpha = alpha
            best_result = result

    summary_path = run_dir / "alpha_sweep.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["alpha", "pcc", "pcc_low", "pcc_high", "mse", "n_phones"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    metadata = {
        "backend": backend_name,
        "source_alpha": source_alpha,
        "alphas": alphas,
        "cache_root": str(settings.features_dir),
        "summary_path": str(summary_path),
        "best_alpha": best_alpha,
        "best_pcc": None if best_result is None else best_result.pcc,
        "best_mse": None if best_result is None else best_result.mse,
    }
    metadata_path = run_dir / "alpha_sweep_meta.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if best_result is None or best_alpha is None:
        msg = "Alpha sweep produced no results."
        raise RuntimeError(msg)

    _log_alpha_sweep_to_wandb(
        _AlphaSweepWandbPayload(
            backend_name=backend_name,
            source_alpha=source_alpha,
            alphas=alphas,
            rows=rows,
            best_alpha=best_alpha,
            best_result=best_result,
            dataset_revision=DATASET_REVISION,
            summary_path=summary_path,
            metadata_path=metadata_path,
        ),
    )

    print(f"Run dir: {run_dir}")  # noqa: T201
    print(f"Summary: {summary_path}")  # noqa: T201
    print(  # noqa: T201
        "Best alpha: "
        f"{best_alpha:.4f}  PCC={best_result.pcc:.4f}  MSE={best_result.mse:.4f}",
    )
