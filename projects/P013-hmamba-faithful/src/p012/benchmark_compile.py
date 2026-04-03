from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import torch
import yaml

from .models import HMamba
from .runtime import require_cuda_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("conf/so762/HMamba.yaml"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--compile-backend", type=str, default=None)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_synthetic_batch(
    conf: dict,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gop = torch.randn(batch_size, seq_len, conf["gop_dim"], device=device)
    ssl = [torch.randn(batch_size, seq_len, dim, device=device) for dim in conf["ssl_dim"]]
    raw = torch.randn(batch_size, seq_len, conf["raw_dim"], device=device)
    canophn = torch.randint(0, conf["vocab_size"], (batch_size, seq_len), device=device)
    bies = torch.randint(0, 6, (batch_size, seq_len), device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    return gop, ssl, raw, canophn, bies, mask


def synthetic_loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    loss = outputs[0].float().mean()
    for output in outputs[1:]:
        loss = loss + output.float().mean()
    return loss


def timed_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(*batch)
    loss = synthetic_loss(outputs)
    loss.backward()
    optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed, float(loss.detach())


def benchmark_case(conf: dict, args: argparse.Namespace, device: torch.device, compiled: bool) -> dict:
    model = HMamba(**conf).to(device)
    resolved_backend = model.resolved_mamba_backend
    if compiled:
        compile_kwargs = {}
        if args.compile_backend:
            compile_kwargs["backend"] = args.compile_backend
        if args.compile_mode:
            compile_kwargs["mode"] = args.compile_mode
        model = cast(torch.nn.Module, torch.compile(model, **compile_kwargs))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batch = make_synthetic_batch(conf, args.batch_size, args.seq_len, device)

    cold_step_s, cold_loss = timed_train_step(model, optimizer, batch, device)
    for _ in range(max(args.warmup - 1, 0)):
        timed_train_step(model, optimizer, batch, device)

    steady_times = []
    last_loss = cold_loss
    for _ in range(args.iters):
        step_time_s, last_loss = timed_train_step(model, optimizer, batch, device)
        steady_times.append(step_time_s)

    return {
        "mode": "compiled" if compiled else "eager",
        "resolved_mamba_backend": resolved_backend,
        "cold_step_ms": round(cold_step_s * 1000, 3),
        "mean_step_ms": round(sum(steady_times) * 1000 / len(steady_times), 3),
        "min_step_ms": round(min(steady_times) * 1000, 3),
        "max_step_ms": round(max(steady_times) * 1000, 3),
        "last_loss": round(last_loss, 6),
    }


def main() -> None:
    args = parse_args()
    conf = yaml.safe_load(args.config.read_text())
    device = require_cuda_device("HMamba benchmarking")

    results = {
        "torch_version": torch.__version__,
        "device": str(device),
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "warmup": args.warmup,
            "iters": args.iters,
            "compile_mode": args.compile_mode,
            "compile_backend": args.compile_backend,
        },
        "cases": [],
    }

    eager_case = benchmark_case(conf, args, device, compiled=False)
    results["cases"].append(eager_case)

    try:
        compiled_case = benchmark_case(conf, args, device, compiled=True)
    except Exception as exc:  # pragma: no cover - depends on local compiler/runtime state
        compiled_case = {"mode": "compiled", "error": f"{type(exc).__name__}: {exc}"}
    results["cases"].append(compiled_case)

    if "mean_step_ms" in eager_case and "mean_step_ms" in compiled_case:
        results["compiled_speedup"] = round(eager_case["mean_step_ms"] / compiled_case["mean_step_ms"], 3)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
