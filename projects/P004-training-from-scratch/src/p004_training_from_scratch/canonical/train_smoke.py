"""Local canonical-lane train smoke using real phone targets.

References
- PyTorch `torch.compile`: https://pytorch.org/docs/stable/generated/torch.compile.html
- PyTorch `CTCLoss`: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
- torchaudio `MelSpectrogram`: https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html
- Hugging Face Accelerate quicktour: https://huggingface.co/docs/accelerate/main/en/quicktour
- W&B run init/logging: https://docs.wandb.ai/models/ref/python/functions/init
"""

from __future__ import annotations

import contextlib
import gzip
import importlib.metadata as metadata
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from p004_training_from_scratch.canonical.common import (
    DEFAULT_NUM_MELS,
    CanonicalAttentionBackend,
    CanonicalModelConfig,
    CanonicalModelType,
    build_canonical_ctc_model,
    load_log_mel_features,
)
from p004_training_from_scratch.machine_manifest import capture_machine_manifest
from p004_training_from_scratch.settings import PROJECT_ROOT, ProjectSettings
from p004_training_from_scratch.tracking import resolve_local_wandb_mode

DEFAULT_TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "manifests_phone_smoke"
    / "librispeech_cuts_train-clean-100.jsonl.gz"
)
DEFAULT_DEV_MANIFEST = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "manifests_phone_smoke"
    / "librispeech_cuts_dev-clean.jsonl.gz"
)
DEFAULT_TOKENS_PATH = (
    PROJECT_ROOT / "experiments" / "data" / "lang_phone" / "tokens.txt"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "experiments" / "checkpoints" / "canonical_phone_ctc"
)
LossComputeDType = Literal["model", "float32"]


@dataclass(frozen=True, slots=True)
class SmokeCut:
    cut_id: str
    audio_path: Path
    phones: tuple[str, ...]
    duration_seconds: float | None


@dataclass(frozen=True, slots=True)
class PreparedExample:
    cut_id: str
    features: Any
    target_ids: tuple[int, ...]
    duration_seconds: float | None


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    epoch: int
    model_state: Any
    optimizer_state: Any


def load_phone_token_table(tokens_path: Path = DEFAULT_TOKENS_PATH) -> dict[str, int]:
    if not tokens_path.is_file():
        msg = f"token table not found: {tokens_path}"
        raise FileNotFoundError(msg)

    token_table: dict[str, int] = {}
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        token, token_id = stripped.rsplit(maxsplit=1)
        token_table[token] = int(token_id)
    if "<eps>" not in token_table:
        msg = f"blank token <eps> missing from token table: {tokens_path}"
        raise ValueError(msg)
    return token_table


def read_smoke_cuts(
    manifest_path: Path,
    *,
    limit: int,
    token_table: dict[str, int] | None = None,
) -> list[SmokeCut]:
    if limit <= 0:
        msg = "limit must be positive"
        raise ValueError(msg)
    if not manifest_path.is_file():
        msg = f"manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)

    if manifest_path.suffix == ".gz":
        with gzip.open(manifest_path, "rt", encoding="utf-8") as handle:
            cuts = _read_smoke_cut_lines(
                handle,
                limit=limit,
                token_table=token_table,
            )
    else:
        with manifest_path.open("rt", encoding="utf-8") as handle:
            cuts = _read_smoke_cut_lines(
                handle,
                limit=limit,
                token_table=token_table,
            )

    if not cuts:
        msg = f"no usable cuts found in manifest: {manifest_path}"
        raise ValueError(msg)
    return cuts


def prepare_examples(
    *,
    cuts: list[SmokeCut],
    token_table: dict[str, int],
    torch: Any,
    torchaudio: Any,
) -> list[PreparedExample]:
    examples: list[PreparedExample] = []
    for cut in cuts:
        features = load_log_mel_features(
            path=cut.audio_path,
            torch=torch,
            torchaudio=torchaudio,
            device=None,
            n_mels=DEFAULT_NUM_MELS,
        )
        examples.append(
            PreparedExample(
                cut_id=cut.cut_id,
                features=features,
                target_ids=tuple(token_table[phone] for phone in cut.phones),
                duration_seconds=cut.duration_seconds,
            )
        )
    return examples


def _read_smoke_cut_lines(
    lines: Iterable[str],
    *,
    limit: int,
    token_table: dict[str, int] | None,
) -> list[SmokeCut]:
    cuts: list[SmokeCut] = []
    for raw_line in lines:
        payload = json.loads(raw_line)
        phones = tuple(
            phone for phone in str(payload["supervisions"][0]["text"]).split() if phone
        )
        if not phones:
            continue

        source = Path(str(payload["recording"]["sources"][0]["source"]))
        if not source.is_file():
            msg = f"manifest listed missing audio file: {source}"
            raise FileNotFoundError(msg)

        if token_table is not None:
            unknown = sorted(
                {phone for phone in phones if phone not in token_table}
            )
            if unknown:
                msg = f"manifest contains unknown phones: {', '.join(unknown)}"
                raise ValueError(msg)

        cuts.append(
            SmokeCut(
                cut_id=str(payload["id"]),
                audio_path=source,
                phones=phones,
                duration_seconds=_coerce_optional_float(payload.get("duration")),
            )
        )
        if len(cuts) == limit:
            break
    return cuts


def collate_prepared_examples(
    examples: list[PreparedExample],
    *,
    torch: Any,
) -> dict[str, Any]:
    feature_rows = [example.features for example in examples]
    input_lengths = torch.tensor(
        [int(feature.shape[0]) for feature in feature_rows],
        dtype=torch.long,
    )
    target_lengths = torch.tensor(
        [len(example.target_ids) for example in examples],
        dtype=torch.long,
    )
    pairs = zip(
        input_lengths.tolist(),
        target_lengths.tolist(),
        [example.cut_id for example in examples],
        strict=True,
    )
    for input_length, target_length, cut_id in pairs:
        if int(target_length) >= int(input_length):
            msg = (
                f"CTC target sequence is longer than its input sequence for {cut_id}: "
                f"target_length={target_length}, input_length={input_length}"
            )
            raise ValueError(msg)

    flat_targets = torch.tensor(
        [token for example in examples for token in example.target_ids],
        dtype=torch.long,
    )
    if flat_targets.numel() == 0:
        msg = "CTC targets cannot be empty"
        raise ValueError(msg)

    padded_features = torch.nn.utils.rnn.pad_sequence(feature_rows, batch_first=True)
    return {
        "cut_ids": [example.cut_id for example in examples],
        "features": padded_features,
        "input_lengths": input_lengths,
        "targets": flat_targets,
        "target_lengths": target_lengths,
    }


def run_canonical_train_smoke(
    *,
    output_dir: Path,
    train_manifest: Path = DEFAULT_TRAIN_MANIFEST,
    dev_manifest: Path = DEFAULT_DEV_MANIFEST,
    tokens_path: Path = DEFAULT_TOKENS_PATH,
    train_limit: int = 12,
    dev_limit: int = 4,
    epochs: int = 1,
    batch_size: int = 3,
    model_type: CanonicalModelType = "tiny",
    attention_backend: CanonicalAttentionBackend = "mha",
    hidden_dim: int = 256,
    encoder_layers: int = 4,
    attention_heads: int = 4,
    conv_kernel_size: int = 15,
    dropout: float = 0.1,
    learning_rate: float = 3e-4,
    loss_compute_dtype: LossComputeDType = "model",
    seed: int = 42,
    resume_from: Path | None = None,
    enable_compile: bool = True,
    allow_online_trackers: bool = False,
    with_wandb: bool = True,
) -> dict[str, Any]:
    if epochs <= 0:
        msg = "epochs must be positive"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if _uses_flex_attention(attention_backend) and not enable_compile:
        msg = (
            "flex attention backends require enable_compile=True in the canonical "
            "trainer"
        )
        raise ValueError(msg)

    settings = ProjectSettings.from_env()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    machine_manifest_path = output_dir / "machine_manifest.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    model_state_path = output_dir / "model_state.pt"
    metrics_path = output_dir / "metrics.jsonl"

    report: dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(output_dir),
        "machine_manifest_path": str(machine_manifest_path),
        "train_manifest": str(train_manifest),
        "dev_manifest": str(dev_manifest),
        "tokens_path": str(tokens_path),
        "config": {
            "train_limit": train_limit,
            "dev_limit": dev_limit,
            "epochs": epochs,
            "batch_size": batch_size,
            "model_type": model_type,
            "attention_backend": attention_backend,
            "hidden_dim": hidden_dim,
            "encoder_layers": encoder_layers,
            "attention_heads": attention_heads,
            "conv_kernel_size": conv_kernel_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "loss_compute_dtype": loss_compute_dtype,
            "seed": seed,
            "resume_from": str(resume_from) if resume_from is not None else None,
            "enable_compile": enable_compile,
        },
        "success": False,
    }

    report["machine_manifest"] = capture_machine_manifest(output=machine_manifest_path)

    try:
        import accelerate
        import torch
        import torchaudio
        import wandb
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:  # pragma: no cover - exercised by real runtime
        report["error"] = _format_exception(exc)
        return _write_report(report, report_path)

    report["versions"] = {
        "accelerate": accelerate.__version__,
        "torch": torch.__version__,
        "torchcodec": _package_version("torchcodec"),
        "torchaudio": getattr(torchaudio, "__version__", "unknown"),
        "wandb": wandb.__version__,
    }

    if not torch.cuda.is_available():
        report["error"] = {
            "type": "RuntimeError",
            "message": "torch.cuda.is_available() returned false",
        }
        return _write_report(report, report_path)

    torch.manual_seed(seed)
    accelerator = accelerate.Accelerator(mixed_precision="bf16")
    report["device"] = {
        "device": str(accelerator.device),
        "mixed_precision": accelerator.mixed_precision,
    }
    report["attention"] = _capture_sdpa_state(torch)

    token_table = load_phone_token_table(tokens_path)
    blank_id = token_table["<eps>"]
    train_cuts = read_smoke_cuts(
        train_manifest,
        limit=train_limit,
        token_table=token_table,
    )
    dev_cuts = read_smoke_cuts(dev_manifest, limit=dev_limit, token_table=token_table)
    train_examples = prepare_examples(
        cuts=train_cuts,
        token_table=token_table,
        torch=torch,
        torchaudio=torchaudio,
    )
    dev_examples = prepare_examples(
        cuts=dev_cuts,
        token_table=token_table,
        torch=torch,
        torchaudio=torchaudio,
    )

    report["dataset"] = {
        "blank_id": blank_id,
        "vocab_size": len(token_table),
        "train_cut_count": len(train_examples),
        "dev_cut_count": len(dev_examples),
        "train_cut_ids": [example.cut_id for example in train_examples],
        "dev_cut_ids": [example.cut_id for example in dev_examples],
    }

    class PreparedExampleDataset(Dataset[PreparedExample]):
        def __init__(self, examples: list[PreparedExample]) -> None:
            self._examples = examples

        def __len__(self) -> int:
            return len(self._examples)

        def __getitem__(self, index: int) -> PreparedExample:
            return self._examples[index]

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        PreparedExampleDataset(train_examples),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda rows: collate_prepared_examples(rows, torch=torch),
    )
    dev_loader = DataLoader(
        PreparedExampleDataset(dev_examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_prepared_examples(rows, torch=torch),
    )

    model_config = CanonicalModelConfig(
        model_type=model_type,
        attention_backend=attention_backend,
        hidden_dim=hidden_dim,
        encoder_layers=encoder_layers,
        attention_heads=attention_heads,
        conv_kernel_size=conv_kernel_size,
        dropout=dropout,
    )
    model = build_canonical_ctc_model(
        torch=torch,
        input_dim=DEFAULT_NUM_MELS,
        vocab_size=len(token_table),
        config=model_config,
    )
    report["model"] = {
        "model_type": model_type,
        "attention_backend": attention_backend,
        "hidden_dim": hidden_dim,
        "encoder_layers": encoder_layers,
        "attention_heads": attention_heads,
        "conv_kernel_size": conv_kernel_size,
        "dropout": dropout,
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    initial_epoch = 0
    if resume_from is not None:
        if not resume_from.is_file():
            report["error"] = {
                "type": "FileNotFoundError",
                "message": f"resume checkpoint not found: {resume_from}",
            }
            return _write_report(report, report_path)
        resume_payload = _load_resume_checkpoint(
            torch.load(resume_from, map_location="cpu")
        )
        model.load_state_dict(resume_payload.model_state)
        optimizer.load_state_dict(resume_payload.optimizer_state)
        initial_epoch = resume_payload.epoch + 1
        if initial_epoch >= epochs:
            report["error"] = {
                "type": "ValueError",
                "message": (
                    "resume checkpoint already meets or exceeds the requested "
                    f"epoch target: resume_epoch={resume_payload.epoch}, "
                    f"epochs={epochs}"
                ),
            }
            return _write_report(report, report_path)
        report["resume"] = {
            "resume_from": str(resume_from),
            "loaded_epoch": resume_payload.epoch,
            "start_epoch": initial_epoch,
        }
    if enable_compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    loss_fn = torch.nn.CTCLoss(blank=blank_id, zero_infinity=True)
    model, optimizer, train_loader, dev_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        dev_loader,
    )

    wandb_run = None
    if with_wandb:
        mode, reason = resolve_local_wandb_mode(
            settings,
            allow_online=allow_online_trackers,
        )
        wandb_run = wandb.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            name=output_dir.name,
            group="p004-canonical-local-smoke",
            job_type="train-smoke",
            mode=mode,
            tags=["p004", "canonical", "train-smoke"],
            notes=(
                "Local canonical-lane CTC smoke with real phone targets from the "
                "prepared smoke manifests."
            ),
            config={
                "config": report["config"],
                "dataset": {
                    "train_manifest": str(train_manifest),
                    "dev_manifest": str(dev_manifest),
                    "tokens_path": str(tokens_path),
                    "train_cut_count": len(train_examples),
                    "dev_cut_count": len(dev_examples),
                },
                "versions": report["versions"],
                "wandb_mode_reason": reason,
            },
        )
        report["tracking"] = {
            "wandb_mode": mode,
            "wandb_reason": reason,
            "wandb_run_id": getattr(wandb_run, "id", None),
            "wandb_run_path": getattr(wandb_run, "path", None),
        }

    start = time.perf_counter()
    metrics_log: list[dict[str, Any]] = []
    train_epoch_summaries: list[dict[str, Any]] = []
    epoch_checkpoint_paths: list[str] = []
    step_durations_seconds: list[float] = []

    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    for epoch_index in range(initial_epoch, epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch_index, batch in enumerate(train_loader):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                _mark_cudagraph_step_begin(
                    torch=torch,
                    compile_enabled=enable_compile,
                    attention_backend=attention_backend,
                )
                logits = model(batch["features"])
                log_probs = _build_ctc_log_probs(
                    logits=logits,
                    loss_compute_dtype=loss_compute_dtype,
                ).transpose(0, 1)
                loss = loss_fn(
                    log_probs,
                    batch["targets"],
                    batch["input_lengths"],
                    batch["target_lengths"],
                )
            loss_value = float(loss.detach().float().item())
            if not math.isfinite(loss_value):
                batch_metric = {
                    "epoch": epoch_index,
                    "batch_index": batch_index,
                    "train_loss": loss_value,
                    "batch_size": len(batch["cut_ids"]),
                    "max_input_frames": int(batch["input_lengths"].max().item()),
                    "max_target_length": int(batch["target_lengths"].max().item()),
                    "step_time_seconds": round(time.perf_counter() - step_start, 6),
                }
                metrics_log.append(batch_metric)
                report["error"] = {
                    "type": "RuntimeError",
                    "message": (
                        "non-finite train loss detected "
                        f"at epoch={epoch_index}, batch_index={batch_index}"
                    ),
                }
                report["failed_batch"] = batch_metric
                if accelerator.is_main_process:
                    _write_metrics(metrics_log, metrics_path)
                if wandb_run is not None:
                    with contextlib.suppress(Exception):
                        wandb.finish()
                return _write_report(report, report_path)
            accelerator.backward(loss)
            optimizer.step()
            _synchronize_if_cuda(torch, accelerator.device)
            step_duration_seconds = time.perf_counter() - step_start

            batch_metric = {
                "epoch": epoch_index,
                "batch_index": batch_index,
                "train_loss": loss_value,
                "batch_size": len(batch["cut_ids"]),
                "max_input_frames": int(batch["input_lengths"].max().item()),
                "max_target_length": int(batch["target_lengths"].max().item()),
                "step_time_seconds": round(step_duration_seconds, 6),
            }
            metrics_log.append(batch_metric)
            epoch_losses.append(loss_value)
            step_durations_seconds.append(step_duration_seconds)
            if wandb_run is not None and accelerator.is_main_process:
                wandb.log(batch_metric)

        train_epoch_summaries.append(
            {
                "epoch": epoch_index,
                "mean_train_loss": _mean(epoch_losses),
                "last_train_loss": epoch_losses[-1],
                "batch_count": len(epoch_losses),
            }
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            import torch

            epoch_checkpoint_path = output_dir / f"epoch-{epoch_index}.pt"
            checkpoint_payload = _build_training_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                epoch=epoch_index,
                config=report["config"],
            )
            torch.save(checkpoint_payload, epoch_checkpoint_path)
            torch.save(checkpoint_payload, checkpoint_path)
            epoch_checkpoint_paths.append(str(epoch_checkpoint_path))

    dev_summary = _evaluate_dev(
        model=model,
        dev_loader=dev_loader,
        accelerator=accelerator,
        loss_fn=loss_fn,
        blank_id=blank_id,
        attention_backend=attention_backend,
        compile_enabled=enable_compile,
        loss_compute_dtype=loss_compute_dtype,
    )
    if "error" in dev_summary:
        report["dev"] = dev_summary
        report["error"] = dev_summary["error"]
        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        if accelerator.is_main_process:
            _write_metrics(metrics_log, metrics_path)
        if wandb_run is not None:
            with contextlib.suppress(Exception):
                wandb.finish()
        return _write_report(report, report_path)
    dev_metric = {
        "epoch": epochs - 1,
        "dev_loss": dev_summary["dev_loss"],
        "dev_per": dev_summary["dev_per"],
    }
    metrics_log.append(dev_metric)
    if wandb_run is not None and accelerator.is_main_process:
        wandb.log(dev_metric)

    if accelerator.is_main_process:
        model_state = _normalize_model_state_dict(accelerator.get_state_dict(model))
        torch.save(model_state, model_state_path)
        _write_metrics(metrics_log, metrics_path)

    elapsed_seconds = round(time.perf_counter() - start, 3)
    report["train"] = {
        "epochs": train_epoch_summaries,
        "step_count": sum(epoch["batch_count"] for epoch in train_epoch_summaries),
    }
    report["dev"] = dev_summary
    report["artifacts"] = {
        "checkpoint_path": str(checkpoint_path),
        "epoch_checkpoint_paths": epoch_checkpoint_paths,
        "metrics_path": str(metrics_path),
        "model_state_path": str(model_state_path),
    }
    report["benchmark"] = _summarize_benchmark_metrics(
        step_durations_seconds=step_durations_seconds,
        elapsed_seconds=elapsed_seconds,
        torch=torch,
        device=accelerator.device,
        compile_enabled=enable_compile,
    )
    report["elapsed_seconds"] = elapsed_seconds
    report["success"] = checkpoint_path.is_file() and bool(train_epoch_summaries)

    if wandb_run is not None:
        with contextlib.suppress(Exception):
            wandb.finish()
    return _write_report(report, report_path)


def _evaluate_dev(
    *,
    model: Any,
    dev_loader: Any,
    accelerator: Any,
    loss_fn: Any,
    blank_id: int,
    attention_backend: CanonicalAttentionBackend,
    compile_enabled: bool,
    loss_compute_dtype: LossComputeDType,
) -> dict[str, Any]:
    import torch

    loss_values: list[float] = []
    error_count = 0
    token_count = 0

    model.eval()
    with _inference_mode():
        for batch in dev_loader:
            with accelerator.autocast():
                _mark_cudagraph_step_begin(
                    torch=torch,
                    compile_enabled=compile_enabled,
                    attention_backend=attention_backend,
                )
                logits = model(batch["features"])
                log_probs = _build_ctc_log_probs(
                    logits=logits,
                    loss_compute_dtype=loss_compute_dtype,
                ).transpose(0, 1)
                loss = loss_fn(
                    log_probs,
                    batch["targets"],
                    batch["input_lengths"],
                    batch["target_lengths"],
                )
            loss_value = float(loss.detach().float().item())
            if not math.isfinite(loss_value):
                return {
                    "dev_loss": float("nan"),
                    "dev_per": 1.0,
                    "token_error_count": error_count,
                    "token_count": token_count,
                    "error": {
                        "type": "RuntimeError",
                        "message": "non-finite dev loss detected during evaluation",
                    },
                }
            loss_values.append(loss_value)

            predictions = _ctc_greedy_decode(
                logits=logits,
                input_lengths=batch["input_lengths"],
                blank_id=blank_id,
            )
            targets = _split_targets(
                flat_targets=batch["targets"],
                target_lengths=batch["target_lengths"],
            )
            for prediction, target in zip(predictions, targets, strict=True):
                error_count += _edit_distance(prediction, target)
                token_count += len(target)

    return {
        "dev_loss": _mean(loss_values),
        "dev_per": float(error_count / token_count) if token_count else 0.0,
        "token_error_count": error_count,
        "token_count": token_count,
    }


def _uses_flex_attention(attention_backend: CanonicalAttentionBackend) -> bool:
    return attention_backend != "mha"


def _build_ctc_log_probs(
    *,
    logits: Any,
    loss_compute_dtype: LossComputeDType,
) -> Any:
    if loss_compute_dtype == "float32":
        logits = logits.float()
    return logits.log_softmax(dim=-1)


def _mark_cudagraph_step_begin(
    *,
    torch: Any,
    compile_enabled: bool,
    attention_backend: CanonicalAttentionBackend,
) -> None:
    if not compile_enabled or not _uses_flex_attention(attention_backend):
        return
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return
    mark_step_begin = getattr(compiler, "cudagraph_mark_step_begin", None)
    if mark_step_begin is None:
        return
    mark_step_begin()


def _ctc_greedy_decode(
    *,
    logits: Any,
    input_lengths: Any,
    blank_id: int,
) -> list[list[int]]:
    greedy = logits.argmax(dim=-1).detach().cpu()
    decoded: list[list[int]] = []
    for row, input_length in zip(
        greedy,
        input_lengths.detach().cpu().tolist(),
        strict=True,
    ):
        frame_ids = row[:input_length].tolist()
        collapsed: list[int] = []
        previous: int | None = None
        for token_id in frame_ids:
            if token_id == blank_id:
                previous = None
                continue
            if token_id != previous:
                collapsed.append(int(token_id))
            previous = int(token_id)
        decoded.append(collapsed)
    return decoded


def _split_targets(*, flat_targets: Any, target_lengths: Any) -> list[list[int]]:
    flat = flat_targets.detach().cpu().tolist()
    lengths = target_lengths.detach().cpu().tolist()
    pieces: list[list[int]] = []
    cursor = 0
    for length in lengths:
        next_cursor = cursor + int(length)
        pieces.append([int(token_id) for token_id in flat[cursor:next_cursor]])
        cursor = next_cursor
    return pieces


def _edit_distance(prediction: list[int], target: list[int]) -> int:
    if not prediction:
        return len(target)
    if not target:
        return len(prediction)

    previous_row = list(range(len(target) + 1))
    for pred_index, pred_token in enumerate(prediction, start=1):
        current_row = [pred_index]
        for target_index, target_token in enumerate(target, start=1):
            substitution_cost = 0 if pred_token == target_token else 1
            current_row.append(
                min(
                    previous_row[target_index] + 1,
                    current_row[target_index - 1] + 1,
                    previous_row[target_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]


def _load_resume_checkpoint(payload: Any) -> ResumeCheckpoint:
    if not isinstance(payload, dict):
        msg = "resume checkpoint must be a dictionary payload"
        raise ValueError(msg)

    if "model_state" not in payload:
        msg = "resume checkpoint is missing model_state"
        raise ValueError(msg)
    if "optimizer_state" not in payload:
        msg = "resume checkpoint is missing optimizer_state"
        raise ValueError(msg)

    try:
        epoch = int(payload["epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        msg = "resume checkpoint is missing a valid integer epoch"
        raise ValueError(msg) from exc

    return ResumeCheckpoint(
        epoch=epoch,
        model_state=_normalize_model_state_dict(payload["model_state"]),
        optimizer_state=payload["optimizer_state"],
    )


def _build_training_checkpoint(
    *,
    accelerator: Any,
    model: Any,
    optimizer: Any,
    epoch: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": _normalize_model_state_dict(accelerator.get_state_dict(model)),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }


def _normalize_model_state_dict(state_dict: Any) -> dict[str, Any]:
    if not isinstance(state_dict, dict):
        msg = "model_state must be a dictionary payload"
        raise ValueError(msg)

    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return dict(state_dict)

    return {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
    }


def _capture_sdpa_state(torch: Any) -> dict[str, Any]:
    cuda_backends = torch.backends.cuda
    return {
        "flash_sdp_enabled": bool(cuda_backends.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(
            cuda_backends.mem_efficient_sdp_enabled()
        ),
        "math_sdp_enabled": bool(cuda_backends.math_sdp_enabled()),
    }


def _summarize_benchmark_metrics(
    *,
    step_durations_seconds: list[float],
    elapsed_seconds: float,
    torch: Any,
    device: Any,
    compile_enabled: bool,
) -> dict[str, Any]:
    peak_memory_allocated_mb: float | None = None
    peak_memory_reserved_mb: float | None = None
    if device.type == "cuda":
        peak_memory_allocated_mb = round(
            float(torch.cuda.max_memory_allocated(device)) / (1024**2),
            3,
        )
        peak_memory_reserved_mb = round(
            float(torch.cuda.max_memory_reserved(device)) / (1024**2),
            3,
        )

    steady_state = step_durations_seconds[1:] if len(step_durations_seconds) > 1 else []
    steps_per_second = (
        len(step_durations_seconds) / elapsed_seconds if elapsed_seconds > 0 else None
    )
    return {
        "compile_enabled": compile_enabled,
        "step_count": len(step_durations_seconds),
        "first_step_seconds": _rounded_or_none(
            step_durations_seconds[0] if step_durations_seconds else None
        ),
        "mean_step_seconds": _rounded_or_none(_mean(step_durations_seconds)),
        "median_step_seconds": _rounded_or_none(_median(step_durations_seconds)),
        "steady_state_mean_step_seconds": _rounded_or_none(_mean(steady_state)),
        "steady_state_median_step_seconds": _rounded_or_none(_median(steady_state)),
        "steps_per_second": _rounded_or_none(steps_per_second),
        "peak_memory_allocated_mb": peak_memory_allocated_mb,
        "peak_memory_reserved_mb": peak_memory_reserved_mb,
    }


def _synchronize_if_cuda(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_metrics(metrics: list[dict[str, Any]], path: Path) -> None:
    path.write_text(
        "".join(f"{json.dumps(metric, sort_keys=True)}\n" for metric in metrics),
        encoding="utf-8",
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[middle])
    return float((sorted_values[middle - 1] + sorted_values[middle]) / 2)


def _rounded_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _inference_mode():
    import torch

    return torch.inference_mode()


def _format_exception(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _write_report(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    report_path.write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "DEFAULT_DEV_MANIFEST",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TOKENS_PATH",
    "DEFAULT_TRAIN_MANIFEST",
    "PreparedExample",
    "ResumeCheckpoint",
    "SmokeCut",
    "_build_training_checkpoint",
    "_load_resume_checkpoint",
    "_mark_cudagraph_step_begin",
    "_uses_flex_attention",
    "collate_prepared_examples",
    "load_phone_token_table",
    "prepare_examples",
    "read_smoke_cuts",
    "run_canonical_train_smoke",
]
