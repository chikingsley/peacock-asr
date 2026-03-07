from __future__ import annotations

import contextlib
import importlib.metadata as metadata
import json
import logging
import time
from pathlib import Path
from typing import Any

from p004_training_from_scratch.canonical.common import (
    build_tiny_canonical_ctc,
    load_log_mel_features,
)
from p004_training_from_scratch.machine_manifest import capture_machine_manifest
from p004_training_from_scratch.settings import PROJECT_ROOT, ProjectSettings
from p004_training_from_scratch.tracking import resolve_local_wandb_mode

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "validation" / "canonical_local_preflight.json"
)
DEFAULT_MACHINE_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "validation"
    / "canonical_local_machine_manifest.json"
)
DEFAULT_SMOKE_AUDIO_LIST = (
    PROJECT_ROOT / "experiments" / "data" / "manifests_phone_smoke" / "audio_files.txt"
)


def resolve_preflight_wandb_mode(
    settings: ProjectSettings,
    *,
    allow_online: bool,
    netrc_path: Path | None = None,
) -> tuple[str, str]:
    return resolve_local_wandb_mode(
        settings,
        allow_online=allow_online,
        netrc_path=netrc_path,
        offline_reason="forcing offline mode for local canonical preflight",
    )


def read_smoke_audio_paths(
    audio_list_path: Path = DEFAULT_SMOKE_AUDIO_LIST,
    *,
    limit: int = 2,
) -> list[Path]:
    if limit <= 0:
        msg = "limit must be positive"
        raise ValueError(msg)
    if not audio_list_path.is_file():
        msg = f"smoke audio list not found: {audio_list_path}"
        raise FileNotFoundError(msg)

    paths: list[Path] = []
    for raw_line in audio_list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        path = PROJECT_ROOT / line
        if not path.is_file():
            msg = f"smoke audio file listed but missing: {path}"
            raise FileNotFoundError(msg)
        paths.append(path)
        if len(paths) == limit:
            break

    if not paths:
        msg = f"no smoke audio paths found in {audio_list_path}"
        raise ValueError(msg)
    return paths


def run_canonical_preflight(
    *,
    output: Path = DEFAULT_OUTPUT,
    machine_output: Path = DEFAULT_MACHINE_OUTPUT,
    audio_list_path: Path = DEFAULT_SMOKE_AUDIO_LIST,
    allow_online_trackers: bool = False,
    with_wandb: bool = True,
) -> dict[str, Any]:
    settings = ProjectSettings.from_env()
    report: dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_root": str(PROJECT_ROOT),
        "machine_manifest_path": str(machine_output),
        "checks": [],
        "overall_passed": False,
    }

    machine_payload = capture_machine_manifest(output=machine_output)
    report["machine_manifest"] = machine_payload

    try:
        import accelerate
        import torch
        import torch.nn.functional as functional
        import torchaudio
        import wandb
    except Exception as exc:  # pragma: no cover - exercised by real runtime
        report["checks"].append(
            _failed_check(
                "imports",
                exc,
                details={"hint": "Run `uv sync --group dev --group canonical` first."},
            )
        )
        return _write_report(report, output)

    report["versions"] = {
        "accelerate": accelerate.__version__,
        "torch": torch.__version__,
        "torchcodec": _package_version("torchcodec"),
        "torchaudio": getattr(torchaudio, "__version__", "unknown"),
        "wandb": wandb.__version__,
    }

    if not torch.cuda.is_available():
        report["checks"].append(
            _failed_check(
                "cuda_available",
                RuntimeError("torch.cuda.is_available() returned false"),
            )
        )
        return _write_report(report, output)

    device = torch.device("cuda")
    device_props = torch.cuda.get_device_properties(device)
    report["device"] = {
        "name": device_props.name,
        "total_memory_bytes": device_props.total_memory,
        "multi_processor_count": device_props.multi_processor_count,
        "compute_capability": f"{device_props.major}.{device_props.minor}",
    }

    audio_paths = read_smoke_audio_paths(audio_list_path)
    report["audio_paths"] = [str(path) for path in audio_paths]

    report["checks"].append(_passed_check("cuda_available", report["device"]))

    try:
        bf16_metrics = _run_bf16_matmul_smoke(torch, device)
        report["checks"].append(_passed_check("bf16_matmul_backward", bf16_metrics))
    except Exception as exc:  # pragma: no cover - exercised by real runtime
        report["checks"].append(_failed_check("bf16_matmul_backward", exc))

    try:
        sdpa_metrics = _run_sdpa_smoke(torch, functional, device)
        report["checks"].append(
            _passed_check("scaled_dot_product_attention", sdpa_metrics)
        )
    except Exception as exc:  # pragma: no cover - exercised by real runtime
        report["checks"].append(_failed_check("scaled_dot_product_attention", exc))

    try:
        import torchcodec

        _ = torchcodec.decoders.AudioDecoder
        report["checks"].append(
            _passed_check(
                "torchcodec_runtime",
                {"version": getattr(torchcodec, "__version__", "unknown")},
            )
        )
    except Exception as exc:  # pragma: no cover - exercised by real runtime
        report["checks"].append(_failed_check("torchcodec_runtime", exc))

    try:
        feature_batch = _load_feature_batch(
            torch=torch,
            torchaudio=torchaudio,
            audio_paths=audio_paths,
            device=device,
        )
        report["checks"].append(
            _passed_check(
                "torchcodec_melspec_batch",
                {
                    "batch_shape": list(feature_batch["features"].shape),
                    "input_lengths": feature_batch["input_lengths"],
                },
            )
        )
    except Exception as exc:  # pragma: no cover - exercised by real runtime
        report["checks"].append(_failed_check("torchcodec_melspec_batch", exc))
        return _write_report(report, output)

    try:
        compile_metrics = _run_compiled_ctc_smoke(
            torch=torch,
            feature_batch=feature_batch,
            device=device,
        )
        report["checks"].append(
            _passed_check("compiled_ctc_train_smoke", compile_metrics)
        )
    except Exception as exc:  # pragma: no cover - exercised by real runtime
        report["checks"].append(_failed_check("compiled_ctc_train_smoke", exc))

    try:
        accelerate_metrics = _run_accelerate_smoke(
            accelerate=accelerate,
            torch=torch,
        )
        report["checks"].append(
            _passed_check("accelerate_train_step", accelerate_metrics)
        )
    except Exception as exc:  # pragma: no cover - exercised by real runtime
        report["checks"].append(_failed_check("accelerate_train_step", exc))

    if with_wandb:
        try:
            mode, reason = resolve_preflight_wandb_mode(
                settings,
                allow_online=allow_online_trackers,
            )
            wandb_metrics = _run_wandb_smoke(
                wandb=wandb,
                settings=settings,
                mode=mode,
                reason=reason,
                report=report,
            )
            report["checks"].append(_passed_check("wandb_smoke", wandb_metrics))
        except Exception as exc:  # pragma: no cover - exercised by real runtime
            report["checks"].append(_failed_check("wandb_smoke", exc))

    report["overall_passed"] = all(check["passed"] for check in report["checks"])
    return _write_report(report, output)


def _run_bf16_matmul_smoke(torch, device):
    a = torch.randn(
        (1024, 1024),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    b = torch.randn(
        (1024, 1024),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    start = time.perf_counter()
    value = (a @ b).float().mean()
    value.backward()
    torch.cuda.synchronize(device)
    return {
        "loss": float(value.item()),
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
    }


def _run_sdpa_smoke(torch, functional, device):
    q = torch.randn(
        (2, 4, 128, 64),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        (2, 4, 128, 64),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn(
        (2, 4, 128, 64),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    start = time.perf_counter()
    out = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    loss = out.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    return {
        "loss": float(loss.item()),
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
        "output_shape": list(out.shape),
    }


def _load_feature_batch(*, torch, torchaudio, audio_paths, device):
    features = []
    input_lengths = []
    for path in audio_paths:
        feats = load_log_mel_features(
            path=path,
            torch=torch,
            torchaudio=torchaudio,
            device=device,
        )[: 16_000 * 4 // 160]
        features.append(feats)
        input_lengths.append(int(feats.shape[0]))

    padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    return {
        "features": padded,
        "input_lengths": input_lengths,
    }


def _run_compiled_ctc_smoke(*, torch, feature_batch, device):
    vocab_size = 48
    model = build_tiny_canonical_ctc(
        torch=torch,
        input_dim=80,
        hidden_dim=192,
        vocab_size=vocab_size,
    ).to(device)
    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    features = feature_batch["features"]
    input_lengths = torch.tensor(feature_batch["input_lengths"], dtype=torch.long)

    target_rows = []
    target_lengths = []
    max_target = 0
    for input_length in feature_batch["input_lengths"]:
        target_length = max(5, min(32, input_length // 4))
        target_rows.append(
            torch.randint(
                low=1,
                high=vocab_size,
                size=(target_length,),
                device=device,
                dtype=torch.long,
            )
        )
        target_lengths.append(target_length)
        max_target = max(max_target, target_length)

    padded_targets = torch.zeros(
        (len(target_rows), max_target),
        dtype=torch.long,
        device=device,
    )
    for row_index, row in enumerate(target_rows):
        padded_targets[row_index, : row.numel()] = row

    losses = []
    start = time.perf_counter()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = compiled_model(features)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = loss_fn(
            log_probs,
            padded_targets,
            input_lengths,
            torch.tensor(target_lengths, dtype=torch.long, device=device),
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    torch.cuda.synchronize(device)

    return {
        "losses": losses,
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
        "compiled_model_type": type(compiled_model).__name__,
    }


def _run_accelerate_smoke(*, accelerate, torch):
    accelerator = accelerate.Accelerator(mixed_precision="bf16")
    model = torch.nn.Linear(64, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn((16, 64))
    targets = torch.randn((16, 16))
    model, optimizer = accelerator.prepare(model, optimizer)
    inputs = inputs.to(accelerator.device)
    targets = targets.to(accelerator.device)

    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        losses.append(float(loss.detach().float().item()))

    return {
        "device": str(accelerator.device),
        "mixed_precision": accelerator.mixed_precision,
        "losses": losses,
    }


def _run_wandb_smoke(*, wandb, settings, mode, reason, report):
    run = wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        name="p004-canonical-preflight-local",
        group="p004-canonical-preflight",
        job_type="debug",
        mode=mode,
        tags=["p004", "canonical", "preflight"],
        notes=(
            "Local canonical-lane preflight smoke on the Blackwell-class desktop GPU."
        ),
        config={
            "device": report.get("device"),
            "versions": report.get("versions"),
        },
    )
    if run is None:
        return {"mode": mode, "reason": reason, "run_path": None}
    wandb.log(
        {
            "preflight/check_count": len(report["checks"]),
            "preflight/device_total_memory_bytes": report["device"][
                "total_memory_bytes"
            ],
        }
    )
    run_path = getattr(run, "path", None)
    run_id = getattr(run, "id", None)
    with contextlib.suppress(Exception):
        wandb.finish()
    return {
        "mode": mode,
        "reason": reason,
        "run_id": run_id,
        "run_path": run_path,
    }


def _passed_check(name: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": True, "details": details}


def _failed_check(
    name: str,
    exc: Exception,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "passed": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if details:
        payload["details"] = details
    return payload


def _write_report(report: dict[str, Any], output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return report


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


__all__ = [
    "DEFAULT_MACHINE_OUTPUT",
    "DEFAULT_OUTPUT",
    "DEFAULT_SMOKE_AUDIO_LIST",
    "read_smoke_audio_paths",
    "resolve_preflight_wandb_mode",
    "run_canonical_preflight",
]
