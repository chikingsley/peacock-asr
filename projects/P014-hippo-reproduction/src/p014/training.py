from __future__ import annotations

import json
import math
import random
import sys
import warnings
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from p014.config import ScenarioName, TrainingRunSettings, load_named_config
from p014.cono import cono_loss
from p014.data import (
    UTTERANCE_SCORE_KEYS,
    WORD_SCORE_KEYS,
    ReadAloudBatch,
    ReadAloudFeatureDataset,
    collate_read_aloud_batch,
    load_freespeak_resources,
    load_read_aloud_resources,
)
from p014.metrics import EvaluationMetrics, masked_mse, safe_pcc
from p014.model import HiPPOReadAloud, ReadAloudPredictions


@dataclass(frozen=True)
class TrialSummary:
    seed: int
    best_epoch: int
    best_phone_mse: float
    best_phone_pcc: float
    word_pcc: dict[str, float]
    utterance_pcc: dict[str, float]
    artifact_dir: str


@dataclass(frozen=True)
class AggregateReport:
    trials: list[TrialSummary]
    mean: dict[str, float]
    std: dict[str, float]


@dataclass
class _TrackioHandle:
    """Lightweight wrapper so we can fall back to local logging gracefully."""

    active: bool
    module: Any = None
    local_log: list[dict[str, Any]] = field(default_factory=list)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        entry = dict(metrics)
        if step is not None:
            entry["step"] = step
        self.local_log.append(entry)
        if self.active and self.module is not None:
            try:
                self.module.log(metrics, step=step)
            except Exception as exc:  # pragma: no cover - network/runtime issues
                print(
                    f"[trackio] log failed, switching to local-only mode: {exc}",
                    file=sys.stderr,
                )
                self.active = False

    def finish(self) -> None:
        if self.active and self.module is not None:
            try:
                self.module.finish()
            except Exception as exc:  # pragma: no cover
                print(f"[trackio] finish failed: {exc}", file=sys.stderr)
        self.active = False


def _init_trackio(settings: TrainingRunSettings, seed: int) -> _TrackioHandle:
    try:
        trackio_module = cast(Any, import_module("trackio"))
    except Exception as exc:
        print(
            f"[trackio] import failed ({exc}); continuing with local-only logging.",
            file=sys.stderr,
        )
        return _TrackioHandle(active=False)

    config_payload = settings.model_dump(mode="json")
    config_payload["seed"] = seed
    init_kwargs: dict[str, Any] = {
        "project": settings.trackio_project,
        "name": f"{settings.run_name}-seed{seed}",
        "config": config_payload,
    }
    if settings.trackio_space is not None:
        init_kwargs["space_id"] = settings.trackio_space
    try:
        trackio_module.init(**init_kwargs)
    except Exception as exc:  # pragma: no cover - network/auth issues
        print(
            f"[trackio] init failed ({exc}); continuing with local-only logging.",
            file=sys.stderr,
        )
        return _TrackioHandle(active=False, module=trackio_module)

    return _TrackioHandle(active=True, module=trackio_module)


def train_hippo(settings: TrainingRunSettings) -> list[TrialSummary]:
    """Run the 5-trial HiPPO training protocol (read-aloud or free-speaking).

    One Adam trial per seed: build a fresh model, train for ``settings.epochs``
    epochs, select the best epoch by min phone MSE, and record the metrics at
    that epoch (Appendix B of Yan et al. 2025).

    When ``settings.scenario == "free_speaking"`` and ``settings.use_curriculum``
    is true, the trainer mixes read-aloud and free-speaking batches at each
    iteration with Bernoulli probability ``p = τ / T`` (paper eq. 18).
    """

    scenario_name = ScenarioName(settings.scenario)
    paper_config = load_named_config(scenario_name)
    device = resolve_device(settings.device)
    embedding_device = torch.device("cpu") if device.type == "cpu" else device

    run_dir = settings.output_dir / settings.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Datasets are deterministic across seeds — cache once.
    read_train, read_test, read_num_phones, read_gop_dim = load_read_aloud_resources(
        cache_dir=settings.cache_dir,
        modernbert_model=settings.modernbert_model,
        embedding_device=embedding_device,
        max_train_examples=settings.max_train_examples,
        max_test_examples=settings.max_test_examples,
    )

    free_train: ReadAloudFeatureDataset | None = None
    free_test: ReadAloudFeatureDataset | None = None
    num_phone_tokens = read_num_phones
    gop_dim = read_gop_dim
    if scenario_name is ScenarioName.FREE_SPEAKING:
        free_train, free_test, free_num_phones, free_gop_dim = load_freespeak_resources(
            cache_dir=settings.cache_dir,
            modernbert_model=settings.modernbert_model,
            embedding_device=embedding_device,
            max_train_examples=settings.max_train_examples,
            max_test_examples=settings.max_test_examples,
        )
        # Shared phone vocab is built over the union; pick the larger to size
        # the embedding table. The free-speaking builder already unions with
        # the read-aloud references, so ``free_num_phones`` is the superset.
        num_phone_tokens = max(read_num_phones, free_num_phones)
        if free_gop_dim != read_gop_dim:
            raise ValueError(
                f"read-aloud GOP dim {read_gop_dim} != free-speaking {free_gop_dim}"
            )
        gop_dim = free_gop_dim

    summaries: list[TrialSummary] = []
    for seed in settings.seeds:
        summary = _run_trial(
            seed=seed,
            settings=settings,
            paper_config_hidden=paper_config.model.hidden_units,
            paper_num_heads=paper_config.model.mhsa_heads,
            phone_depth=paper_config.model.phone_encoder_blocks,
            word_depth=paper_config.model.word_encoder_blocks,
            utterance_depth=paper_config.model.utterance_encoder_blocks,
            num_phone_tokens=num_phone_tokens,
            gop_dim=gop_dim,
            read_train_dataset=read_train,
            read_test_dataset=read_test,
            free_train_dataset=free_train,
            free_test_dataset=free_test,
            scenario=scenario_name,
            device=device,
            run_dir=run_dir,
        )
        summaries.append(summary)

    aggregate = _aggregate(summaries)
    (run_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "trials": [asdict(trial) for trial in aggregate.trials],
                "mean": aggregate.mean,
                "std": aggregate.std,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _print_aggregate_table(aggregate)
    return summaries


def _run_trial(
    *,
    seed: int,
    settings: TrainingRunSettings,
    paper_config_hidden: int,
    paper_num_heads: int,
    phone_depth: int,
    word_depth: int,
    utterance_depth: int,
    num_phone_tokens: int,
    gop_dim: int,
    read_train_dataset: ReadAloudFeatureDataset,
    read_test_dataset: ReadAloudFeatureDataset,
    free_train_dataset: ReadAloudFeatureDataset | None,
    free_test_dataset: ReadAloudFeatureDataset | None,
    scenario: ScenarioName,
    device: torch.device,
    run_dir: Path,
) -> TrialSummary:
    set_seed(seed)
    trial_dir = run_dir / f"trial_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    trackio = _init_trackio(settings, seed)

    curriculum_active = (
        scenario is ScenarioName.FREE_SPEAKING
        and settings.use_curriculum
        and free_train_dataset is not None
    )

    if scenario is ScenarioName.FREE_SPEAKING:
        if free_train_dataset is None or free_test_dataset is None:
            raise RuntimeError(
                "free-speaking scenario requires free-speaking datasets"
            )
        primary_train = free_train_dataset
        primary_test = free_test_dataset
    else:
        primary_train = read_train_dataset
        primary_test = read_test_dataset

    primary_loader = cast(
        DataLoader[ReadAloudBatch],
        DataLoader(
            primary_train,
            batch_size=settings.batch_size,
            shuffle=True,
            num_workers=settings.num_workers,
            collate_fn=collate_read_aloud_batch,
        ),
    )
    test_loader = cast(
        DataLoader[ReadAloudBatch],
        DataLoader(
            primary_test,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
            collate_fn=collate_read_aloud_batch,
        ),
    )

    read_loader_for_curriculum: DataLoader[ReadAloudBatch] | None = None
    if curriculum_active:
        read_loader_for_curriculum = cast(
            DataLoader[ReadAloudBatch],
            DataLoader(
                read_train_dataset,
                batch_size=settings.batch_size,
                shuffle=True,
                num_workers=settings.num_workers,
                collate_fn=collate_read_aloud_batch,
            ),
        )

    model = HiPPOReadAloud(
        num_phone_tokens=num_phone_tokens,
        hidden_dim=paper_config_hidden,
        num_heads=paper_num_heads,
        phone_depth=phone_depth,
        word_depth=word_depth,
        utterance_depth=utterance_depth,
        gop_dim=gop_dim,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=settings.learning_rate)

    best_phone_mse = float("inf")
    best_metrics: EvaluationMetrics | None = None
    best_epoch = -1
    history: list[dict[str, Any]] = []
    best_checkpoint = trial_dir / "best.pt"

    # Paper (Yan et al. 2025, eq. 18): ``P(τ) = τ / T`` where ``τ ∈ [0, T]``
    # is a *global* iteration counter across the full training schedule.
    # ``T`` defaults to ``epochs * steps_per_epoch`` on the free-speaking
    # loader (the longer of the two, since eval always uses the FS split), or
    # it can be set explicitly via ``curriculum_total_iters``.
    steps_per_epoch = len(primary_loader) if curriculum_active else 0
    curriculum_total = (
        int(settings.curriculum_total_iters)
        if settings.curriculum_total_iters is not None
        else max(1, settings.epochs * max(steps_per_epoch, 1))
    )
    curriculum_state: dict[str, int] = {"tau": 0, "total": curriculum_total}

    for epoch in range(1, settings.epochs + 1):
        train_totals = _train_one_epoch(
            model=model,
            loader=primary_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=settings.epochs,
            settings=settings,
            curriculum_active=curriculum_active,
            read_loader=read_loader_for_curriculum,
            curriculum_state=curriculum_state,
            trackio=trackio,
        )
        metrics = _evaluate(model, test_loader, device, settings)
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_totals["total"],
            "train_apa_loss": train_totals["apa"],
            "train_cono_loss": train_totals["cono"],
            "eval_loss": metrics.loss,
            "phone_mse": metrics.phone_mse,
            "phone_pcc": metrics.phone_pcc,
        }
        for aspect_name, value in metrics.word_pcc.items():
            epoch_record[f"word_pcc/{aspect_name}"] = value
        for aspect_name, value in metrics.utterance_pcc.items():
            epoch_record[f"utterance_pcc/{aspect_name}"] = value
        history.append(epoch_record)
        trackio.log(epoch_record, step=epoch)

        if metrics.phone_mse < best_phone_mse:
            best_phone_mse = metrics.phone_mse
            best_metrics = metrics
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint)

    if best_metrics is None:
        raise RuntimeError("training produced no evaluation metrics")

    summary = TrialSummary(
        seed=seed,
        best_epoch=best_epoch,
        best_phone_mse=best_metrics.phone_mse,
        best_phone_pcc=best_metrics.phone_pcc,
        word_pcc=dict(best_metrics.word_pcc),
        utterance_pcc=dict(best_metrics.utterance_pcc),
        artifact_dir=str(trial_dir),
    )
    (trial_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    (trial_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    trackio.log(
        {
            "best/phone_mse": summary.best_phone_mse,
            "best/phone_pcc": summary.best_phone_pcc,
            "best/epoch": summary.best_epoch,
        },
        step=settings.epochs,
    )
    trackio.finish()
    return summary


def _iter_cycle(
    loader: DataLoader[ReadAloudBatch],
) -> Iterator[ReadAloudBatch]:
    """Infinitely cycle a DataLoader, re-creating the iterator on each pass.

    The curriculum in Yan et al. 2025 (eq. 18) draws from read-aloud and
    free-speaking independently at every iteration, so whichever loader runs
    out first must wrap around. Re-creating the iterator (rather than
    pre-materialising) keeps per-epoch shuffling in place.
    """

    while True:
        yield from loader


def _train_one_epoch(
    *,
    model: HiPPOReadAloud,
    loader: DataLoader[ReadAloudBatch],
    optimizer: Adam,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    settings: TrainingRunSettings,
    curriculum_active: bool,
    read_loader: DataLoader[ReadAloudBatch] | None,
    curriculum_state: dict[str, int],
    trackio: _TrackioHandle,
) -> dict[str, float]:
    model.train()
    running_total = 0.0
    running_apa = 0.0
    running_cono = 0.0
    num_batches = 0

    # The "primary" loader drives the epoch length. For the read-aloud scenario
    # and for plain (non-curriculum) free-speaking training, that's just the
    # scenario's own loader. When curriculum is on, the primary is the
    # free-speaking loader; the read-aloud loader is wrapped as an infinite
    # cycle and draws a batch only when the Bernoulli toss calls for it.
    read_cycle: Iterator[ReadAloudBatch] | None = None
    free_cycle: Iterator[ReadAloudBatch] | None = None
    if curriculum_active:
        if read_loader is None:
            raise RuntimeError("curriculum_active requires a read_loader")
        read_cycle = _iter_cycle(read_loader)
        free_cycle = _iter_cycle(loader)
        total_steps = max(len(loader), len(read_loader))
        primary_iter = range(total_steps)
        progress = tqdm(primary_iter, desc=f"train {epoch}/{total_epochs}", leave=False)
    else:
        progress = tqdm(loader, desc=f"train {epoch}/{total_epochs}", leave=False)

    for step_input in progress:
        scenario_drawn = 0  # 0 = read_aloud, 1 = free_speak
        if curriculum_active:
            _ = step_input
            if free_cycle is None or read_cycle is None:
                raise RuntimeError("curriculum_active implies both cycles exist")
            tau = curriculum_state["tau"]
            total = curriculum_state["total"]
            p = min(1.0, max(0.0, tau / max(total, 1)))
            if torch.rand(()).item() < p:
                batch = next(free_cycle)
                scenario_drawn = 1
            else:
                batch = next(read_cycle)
                scenario_drawn = 0
        else:
            batch = cast(ReadAloudBatch, step_input)
            # Non-curriculum training always reports the scenario that matches
            # ``settings.scenario``.
            scenario_drawn = 1 if settings.scenario == "free_speaking" else 0

        device_batch = move_batch_to_device(batch, device)
        predictions = model(
            gop_features=device_batch.gop_features,
            ssl_utterance=device_batch.ssl_utterance,
            phone_ids=device_batch.phone_ids,
            phone_to_word=device_batch.phone_to_word,
            word_embeddings=device_batch.word_embeddings,
            phone_mask=device_batch.phone_mask,
            word_mask=device_batch.word_mask,
        )
        apa_loss = compute_loss(
            predictions,
            device_batch,
            lambda_phone=float(settings.lambda_phone),
            lambda_word=float(settings.lambda_word),
            lambda_utterance=float(settings.lambda_utterance),
        )
        if settings.use_cono:
            cono_term = cono_loss(
                features=predictions.utterance_feature,
                utterance_accuracy=device_batch.utterance_targets[:, 0],
                lambda_diversity=float(settings.lambda_diversity),
                lambda_tightness=float(settings.lambda_tightness),
            )
            total_loss = apa_loss + float(settings.lambda_cono) * cono_term
            running_cono += float(cono_term.detach().item())
        else:
            total_loss = apa_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if curriculum_active:
            tau = curriculum_state["tau"]
            total = curriculum_state["total"]
            p = min(1.0, max(0.0, tau / max(total, 1)))
            step_record: dict[str, Any] = {
                "curriculum/p": p,
                "curriculum/scenario_drawn": scenario_drawn,
            }
            if scenario_drawn == 0:
                step_record["loss/read_aloud"] = float(total_loss.detach().item())
            else:
                step_record["loss/free_speak"] = float(total_loss.detach().item())
            trackio.log(step_record)
            curriculum_state["tau"] = tau + 1

        running_total += float(total_loss.detach().item())
        running_apa += float(apa_loss.detach().item())
        num_batches += 1
        progress.set_postfix(loss=f"{total_loss.item():.4f}")

    denom = max(num_batches, 1)
    return {
        "total": running_total / denom,
        "apa": running_apa / denom,
        "cono": running_cono / denom,
    }


@torch.no_grad()
def _evaluate(
    model: HiPPOReadAloud,
    loader: DataLoader[ReadAloudBatch],
    device: torch.device,
    settings: TrainingRunSettings,
) -> EvaluationMetrics:
    model.eval()
    phone_preds: list[np.ndarray] = []
    phone_golds: list[np.ndarray] = []
    word_preds: dict[str, list[np.ndarray]] = {name: [] for name in WORD_SCORE_KEYS}
    word_golds: dict[str, list[np.ndarray]] = {name: [] for name in WORD_SCORE_KEYS}
    utt_preds: list[Tensor] = []
    utt_golds: list[Tensor] = []
    losses: list[float] = []

    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        predictions = model(
            gop_features=device_batch.gop_features,
            ssl_utterance=device_batch.ssl_utterance,
            phone_ids=device_batch.phone_ids,
            phone_to_word=device_batch.phone_to_word,
            word_embeddings=device_batch.word_embeddings,
            phone_mask=device_batch.phone_mask,
            word_mask=device_batch.word_mask,
        )
        apa_loss = compute_loss(
            predictions,
            device_batch,
            lambda_phone=float(settings.lambda_phone),
            lambda_word=float(settings.lambda_word),
            lambda_utterance=float(settings.lambda_utterance),
        )
        if settings.use_cono:
            cono_term = cono_loss(
                features=predictions.utterance_feature,
                utterance_accuracy=device_batch.utterance_targets[:, 0],
                lambda_diversity=float(settings.lambda_diversity),
                lambda_tightness=float(settings.lambda_tightness),
            )
            eval_loss = apa_loss + float(settings.lambda_cono) * cono_term
        else:
            eval_loss = apa_loss
        losses.append(float(eval_loss.item()))

        mask = device_batch.phone_mask
        phone_preds.append(predictions.phone_scores[mask].cpu().numpy())
        phone_golds.append(device_batch.phone_targets[mask].cpu().numpy())
        for aspect_index, aspect_name in enumerate(WORD_SCORE_KEYS):
            word_mask = device_batch.word_mask
            word_preds[aspect_name].append(
                predictions.word_scores[:, :, aspect_index][word_mask].cpu().numpy()
            )
            word_golds[aspect_name].append(
                device_batch.word_targets[:, :, aspect_index][word_mask].cpu().numpy()
            )
        utt_preds.append(predictions.utterance_scores.cpu())
        utt_golds.append(device_batch.utterance_targets.cpu())

    flat_phone_preds = np.concatenate(phone_preds) if phone_preds else np.array([], dtype=np.float32)
    flat_phone_golds = np.concatenate(phone_golds) if phone_golds else np.array([], dtype=np.float32)
    phone_mse = (
        float(np.mean((flat_phone_preds - flat_phone_golds) ** 2))
        if flat_phone_preds.size
        else 0.0
    )
    phone_pcc = safe_pcc(flat_phone_preds, flat_phone_golds)

    word_metrics = {
        aspect_name: safe_pcc(
            _concat(word_preds[aspect_name]),
            _concat(word_golds[aspect_name]),
        )
        for aspect_name in WORD_SCORE_KEYS
    }
    utt_tensor_preds = torch.cat(utt_preds, dim=0) if utt_preds else torch.empty(0, 5)
    utt_tensor_golds = torch.cat(utt_golds, dim=0) if utt_golds else torch.empty(0, 5)
    utterance_metrics = {
        aspect_name: safe_pcc(
            _utterance_column(utt_tensor_preds, aspect_index),
            _utterance_column(utt_tensor_golds, aspect_index),
        )
        for aspect_index, aspect_name in enumerate(UTTERANCE_SCORE_KEYS)
    }
    return EvaluationMetrics(
        phone_mse=phone_mse,
        phone_pcc=phone_pcc,
        word_pcc=word_metrics,
        utterance_pcc=utterance_metrics,
        loss=float(np.mean(losses) if losses else 0.0),
    )


def compute_loss(
    predictions: ReadAloudPredictions,
    batch: ReadAloudBatch,
    *,
    lambda_phone: float = 1.0,
    lambda_word: float = 1.0,
    lambda_utterance: float = 1.0,
) -> Tensor:
    phone_loss = masked_mse(predictions.phone_scores, batch.phone_targets, batch.phone_mask)
    word_mask = batch.word_mask.unsqueeze(-1).expand_as(predictions.word_scores)
    word_loss = masked_mse(predictions.word_scores, batch.word_targets, word_mask)
    utterance_loss = torch.mean((predictions.utterance_scores - batch.utterance_targets) ** 2)
    return (
        float(lambda_phone) * phone_loss
        + float(lambda_word) * word_loss
        + float(lambda_utterance) * utterance_loss
    )


def move_batch_to_device(batch: ReadAloudBatch, device: torch.device) -> ReadAloudBatch:
    return ReadAloudBatch(
        gop_features=batch.gop_features.to(device),
        ssl_utterance=batch.ssl_utterance.to(device),
        phone_ids=batch.phone_ids.to(device),
        phone_to_word=batch.phone_to_word.to(device),
        word_embeddings=batch.word_embeddings.to(device),
        phone_mask=batch.phone_mask.to(device),
        word_mask=batch.word_mask.to(device),
        phone_targets=batch.phone_targets.to(device),
        word_targets=batch.word_targets.to(device),
        utterance_targets=batch.utterance_targets.to(device),
    )


def _concat(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float32)
    return np.concatenate(values)


def _utterance_column(values: Tensor, column_index: int) -> np.ndarray:
    if values.numel() == 0:
        return np.array([], dtype=np.float32)
    return values[:, column_index].numpy()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_metric_values(summaries: list[TrialSummary]) -> dict[str, list[float]]:
    """Flatten per-trial metrics into a {metric_key: [values]} dict."""

    collected: dict[str, list[float]] = {
        "best_epoch": [],
        "phone_mse": [],
        "phone_pcc": [],
    }
    for aspect in WORD_SCORE_KEYS:
        collected[f"word_pcc/{aspect}"] = []
    for aspect in UTTERANCE_SCORE_KEYS:
        collected[f"utterance_pcc/{aspect}"] = []

    for summary in summaries:
        collected["best_epoch"].append(float(summary.best_epoch))
        collected["phone_mse"].append(summary.best_phone_mse)
        collected["phone_pcc"].append(summary.best_phone_pcc)
        for aspect in WORD_SCORE_KEYS:
            collected[f"word_pcc/{aspect}"].append(summary.word_pcc[aspect])
        for aspect in UTTERANCE_SCORE_KEYS:
            collected[f"utterance_pcc/{aspect}"].append(summary.utterance_pcc[aspect])
    return collected


def _sample_std(values: list[float]) -> float:
    """Bessel-corrected (N-1) sample standard deviation."""

    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def _aggregate(summaries: list[TrialSummary]) -> AggregateReport:
    collected = _collect_metric_values(summaries)
    mean = {key: (sum(values) / len(values) if values else 0.0) for key, values in collected.items()}
    std = {key: _sample_std(values) for key, values in collected.items()}
    return AggregateReport(trials=summaries, mean=mean, std=std)


def _print_aggregate_table(aggregate: AggregateReport) -> None:
    console = Console()
    table = Table(title=f"HiPPO read-aloud — {len(aggregate.trials)} trials (mean ± std, N-1)")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column("std", justify="right")

    ordered_keys = ["phone_mse", "phone_pcc"]
    ordered_keys.extend(f"word_pcc/{aspect}" for aspect in WORD_SCORE_KEYS)
    ordered_keys.extend(f"utterance_pcc/{aspect}" for aspect in UTTERANCE_SCORE_KEYS)
    ordered_keys.append("best_epoch")

    for key in ordered_keys:
        table.add_row(
            key,
            f"{aggregate.mean[key]:.4f}",
            f"{aggregate.std[key]:.4f}",
        )
    console.print(table)


# Suppress a noisy warning from numpy when all values are identical inside
# safe_pcc — safe_pcc already handles the degenerate case explicitly.
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
