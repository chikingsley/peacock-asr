"""Nightly-lane attention benchmark for C2.3.

References
- PyTorch FlexAttention docs:
  https://pytorch.org/docs/main/nn.attention.flex_attention.html
- PyTorch `scaled_dot_product_attention`:
  https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch FlexAttention + FlashAttention-4 blog:
  https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, cast

from p004_training_from_scratch.machine_manifest import capture_machine_manifest
from p004_training_from_scratch.settings import PROJECT_ROOT

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "experiments" / "benchmarks" / "canonical_phone_ctc"
)


def run_nightly_attention_benchmark(
    *,
    output_dir: Path,
    batch_size: int = 2,
    attention_heads: int = 4,
    seq_len: int = 512,
    head_dim: int = 64,
    warmup_iters: int = 1,
    timed_iters: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    machine_manifest_path = output_dir / "machine_manifest.json"
    payload: dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(output_dir),
        "machine_manifest_path": str(machine_manifest_path),
        "config": {
            "batch_size": batch_size,
            "attention_heads": attention_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "warmup_iters": warmup_iters,
            "timed_iters": timed_iters,
        },
        "success": False,
    }
    payload["machine_manifest"] = capture_machine_manifest(output=machine_manifest_path)

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - runtime-only path
        payload["environment"] = {
            "torch_import_error": _error_payload(exc),
        }
        return _write_report(payload, report_path)

    payload["environment"] = _build_environment_payload(torch)
    if not torch.cuda.is_available():
        payload["failure_reason"] = "torch.cuda.is_available() returned false"
        return _write_report(payload, report_path)

    payload["fa4_expected_supported_on_device"] = _supports_fa4_capability(
        payload["environment"]["device_capability"]
    )

    try:
        payload["sdpa"] = _run_sdpa_case(
            torch=torch,
            batch_size=batch_size,
            attention_heads=attention_heads,
            seq_len=seq_len,
            head_dim=head_dim,
        )
    except Exception as exc:  # pragma: no cover - runtime-only path
        payload["sdpa"] = {
            "ok": False,
            "error": _error_payload(exc),
        }
        return _write_report(payload, report_path)

    payload["flash_attn_direct"] = _run_optional_case(
        lambda: _run_flash_attn_direct_case(
            torch=torch,
            batch_size=batch_size,
            attention_heads=attention_heads,
            seq_len=seq_len,
            head_dim=head_dim,
        )
    )
    payload["flex_attention_compiled"] = {
        name: _run_optional_case(
            lambda kernel_options=kernel_options: _run_compiled_flex_case(
                torch=torch,
                batch_size=batch_size,
                attention_heads=attention_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                kernel_options=kernel_options,
            )
        )
        for name, kernel_options in (
            ("auto", None),
            ("triton", {"BACKEND": "TRITON"}),
            ("flash", {"BACKEND": "FLASH"}),
        )
    }
    payload["comparison"] = _build_comparison(payload)
    payload["success"] = bool(
        payload["sdpa"]["ok"]
        and payload["flex_attention_compiled"]["auto"]["ok"]
        and payload["flex_attention_compiled"]["triton"]["ok"]
    )
    return _write_report(payload, report_path)


def _build_environment_payload(torch: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "flash_attn_4_version": None,
    }
    try:
        payload["flash_attn_4_version"] = importlib.metadata.version("flash-attn-4")
    except importlib.metadata.PackageNotFoundError:
        payload["flash_attn_4_version"] = None

    if torch.cuda.is_available():
        payload["device_name"] = torch.cuda.get_device_name(0)
        payload["device_capability"] = list(torch.cuda.get_device_capability(0))
        payload["device_count"] = int(torch.cuda.device_count())
    return payload


def _run_sdpa_case(
    *,
    torch: Any,
    batch_size: int,
    attention_heads: int,
    seq_len: int,
    head_dim: int,
) -> dict[str, Any]:
    functional = torch.nn.functional
    q, k, v = _fresh_bhld_tensors(
        torch=torch,
        batch_size=batch_size,
        attention_heads=attention_heads,
        seq_len=seq_len,
        head_dim=head_dim,
    )

    out, forward_seconds = _measure_cuda(
        torch,
        lambda: functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        ),
    )
    loss = out.float().square().mean()
    _, backward_seconds = _measure_cuda(torch, loss.backward)
    return {
        "ok": True,
        "shape": list(out.shape),
        "forward_seconds": round(forward_seconds, 6),
        "backward_seconds": round(backward_seconds, 6),
        "mean_abs": round(float(out.float().abs().mean().item()), 6),
    }


def _run_flash_attn_direct_case(
    *,
    torch: Any,
    batch_size: int,
    attention_heads: int,
    seq_len: int,
    head_dim: int,
) -> dict[str, Any]:
    interface = importlib.import_module("flash_attn.cute.interface")
    flash_attn_func = interface.flash_attn_func
    q, k, v = _fresh_bshd_tensors(
        torch=torch,
        batch_size=batch_size,
        attention_heads=attention_heads,
        seq_len=seq_len,
        head_dim=head_dim,
    )
    out, forward_seconds = _measure_cuda(
        torch,
        lambda: flash_attn_func(q, k, v, causal=False),
    )
    loss = out.float().square().mean()
    _, backward_seconds = _measure_cuda(torch, loss.backward)
    return {
        "ok": True,
        "shape": list(out.shape),
        "forward_seconds": round(forward_seconds, 6),
        "backward_seconds": round(backward_seconds, 6),
        "mean_abs": round(float(out.float().abs().mean().item()), 6),
    }


def _run_compiled_flex_case(
    *,
    torch: Any,
    batch_size: int,
    attention_heads: int,
    seq_len: int,
    head_dim: int,
    warmup_iters: int,
    timed_iters: int,
    kernel_options: dict[str, str] | None,
) -> dict[str, Any]:
    flex_module = importlib.import_module("torch.nn.attention.flex_attention")
    flex_attention = flex_module.flex_attention
    q_base, k_base, v_base = _fresh_bhld_tensors(
        torch=torch,
        batch_size=batch_size,
        attention_heads=attention_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        requires_grad=False,
    )

    def runner(q: Any, k: Any, v: Any) -> Any:
        return flex_attention(
            q,
            k,
            v,
            kernel_options=cast(Any, kernel_options),
        )

    compiled_runner = torch.compile(runner, fullgraph=False, mode="reduce-overhead")
    warmup_seconds: list[float] = []
    for _ in range(warmup_iters):
        q, k, v = _clone_for_gradients(q_base, k_base, v_base)
        torch.compiler.cudagraph_mark_step_begin()
        out, duration = _measure_cuda(
            torch,
            lambda q=q, k=k, v=v: compiled_runner(q, k, v).clone(),
        )
        loss = out.float().square().mean()
        _, backward_seconds = _measure_cuda(torch, loss.backward)
        warmup_seconds.append(duration + backward_seconds)

    forward_seconds: list[float] = []
    backward_seconds: list[float] = []
    last_output = None
    for _ in range(timed_iters):
        q, k, v = _clone_for_gradients(q_base, k_base, v_base)
        torch.compiler.cudagraph_mark_step_begin()
        out, duration = _measure_cuda(
            torch,
            lambda q=q, k=k, v=v: compiled_runner(q, k, v).clone(),
        )
        loss = out.float().square().mean()
        _, backward_duration = _measure_cuda(torch, loss.backward)
        forward_seconds.append(duration)
        backward_seconds.append(backward_duration)
        last_output = out

    if last_output is None:
        msg = "timed_iters must be positive"
        raise ValueError(msg)

    return {
        "ok": True,
        "kernel_options": kernel_options,
        "shape": list(last_output.shape),
        "warmup_total_seconds": round(sum(warmup_seconds), 6),
        "mean_forward_seconds": round(_mean(forward_seconds), 6),
        "mean_backward_seconds": round(_mean(backward_seconds), 6),
        "mean_abs": round(float(last_output.float().abs().mean().item()), 6),
    }


def _fresh_bhld_tensors(
    *,
    torch: Any,
    batch_size: int,
    attention_heads: int,
    seq_len: int,
    head_dim: int,
    requires_grad: bool = True,
) -> tuple[Any, Any, Any]:
    kwargs = {
        "device": "cuda",
        "dtype": torch.bfloat16,
    }
    q = torch.randn(batch_size, attention_heads, seq_len, head_dim, **kwargs)
    k = torch.randn(batch_size, attention_heads, seq_len, head_dim, **kwargs)
    v = torch.randn(batch_size, attention_heads, seq_len, head_dim, **kwargs)
    return _clone_for_gradients(q, k, v, requires_grad=requires_grad)


def _fresh_bshd_tensors(
    *,
    torch: Any,
    batch_size: int,
    attention_heads: int,
    seq_len: int,
    head_dim: int,
) -> tuple[Any, Any, Any]:
    kwargs = {
        "device": "cuda",
        "dtype": torch.bfloat16,
    }
    q = torch.randn(batch_size, seq_len, attention_heads, head_dim, **kwargs)
    k = torch.randn(batch_size, seq_len, attention_heads, head_dim, **kwargs)
    v = torch.randn(batch_size, seq_len, attention_heads, head_dim, **kwargs)
    return _clone_for_gradients(q, k, v)


def _clone_for_gradients(
    q: Any,
    k: Any,
    v: Any,
    *,
    requires_grad: bool = True,
) -> tuple[Any, Any, Any]:
    return (
        q.detach().clone().requires_grad_(requires_grad),
        k.detach().clone().requires_grad_(requires_grad),
        v.detach().clone().requires_grad_(requires_grad),
    )


def _measure_cuda(torch: Any, fn: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def _supports_fa4_capability(capability: list[int] | tuple[int, int] | None) -> bool:
    if not capability or len(capability) != 2:
        return False
    major = int(capability[0])
    return major in {9, 10, 11}


def _run_optional_case(fn: Any) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - runtime-only path
        return {
            "ok": False,
            "error": _error_payload(exc),
        }


def _build_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    sdpa = payload["sdpa"]
    auto = payload["flex_attention_compiled"]["auto"]
    triton = payload["flex_attention_compiled"]["triton"]
    flash = payload["flex_attention_compiled"]["flash"]
    return {
        "auto_vs_sdpa_forward_ratio": _safe_ratio(
            auto.get("mean_forward_seconds"),
            sdpa.get("forward_seconds"),
        )
        if auto.get("ok")
        else None,
        "triton_vs_sdpa_forward_ratio": _safe_ratio(
            triton.get("mean_forward_seconds"),
            sdpa.get("forward_seconds"),
        )
        if triton.get("ok")
        else None,
        "flash_vs_sdpa_forward_ratio": _safe_ratio(
            flash.get("mean_forward_seconds"),
            sdpa.get("forward_seconds"),
        )
        if flash.get("ok")
        else None,
        "flash_backend_supported": bool(flash.get("ok")),
        "flash_backend_expected_supported": payload["fa4_expected_supported_on_device"],
        "flash_backend_error": flash.get("error"),
        "direct_flash_attn_ok": bool(payload["flash_attn_direct"].get("ok")),
    }


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 6)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _error_payload(exc: Exception) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _write_report(payload: dict[str, Any], report_path: Path) -> dict[str, Any]:
    report_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "_build_comparison",
    "_supports_fa4_capability",
    "run_nightly_attention_benchmark",
]
