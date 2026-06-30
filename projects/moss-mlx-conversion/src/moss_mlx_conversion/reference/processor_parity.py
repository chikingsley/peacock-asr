from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from moss_mlx_conversion import DEFAULT_MODEL_ID
from moss_mlx_conversion.dump import ensure_dir, tensor_stats, write_json
from moss_mlx_conversion.paths import REFERENCE_DIR
from moss_mlx_conversion.processor import MelConfig, MossProcessor
from moss_mlx_conversion.reference.hf import (
    download_template,
    load_remote_processor_classes,
    load_tokenizer,
)
from moss_mlx_conversion.runtime.audio import load_waveform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local MOSS processor against upstream.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--dump-dir", type=Path, default=REFERENCE_DIR / "processor-parity")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def exact_equal(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    same = torch.equal(left.cpu(), right.cpu())
    return {
        "equal": bool(same),
        "left": tensor_stats(left),
        "right": tensor_stats(right),
    }


def allclose(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_f = left.detach().float().cpu()
    right_f = right.detach().float().cpu()
    diff = (left_f - right_f).abs()
    return {
        "allclose_atol_1e-5": bool(torch.allclose(left_f, right_f, atol=1e-5, rtol=1e-5)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "left": tensor_stats(left),
        "right": tensor_stats(right),
    }


def batch_tensor(batch: Any, key: str) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor for {key}, got {type(value).__name__}")
    return value


def main() -> None:
    args = parse_args()
    dump_dir = ensure_dir(args.dump_dir)

    waveform, audio_path = load_waveform(args.audio)
    tokenizer = load_tokenizer(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    template_path = download_template(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    upstream_processor_cls, upstream_mel_config_cls = load_remote_processor_classes(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    upstream_cfg = upstream_mel_config_cls(
        mel_sr=16_000,
        mel_dim=128,
        mel_n_fft=400,
        mel_hop_length=160,
    )
    upstream = upstream_processor_cls(tokenizer, config=upstream_cfg, enable_time_marker=False)
    upstream.load_template(str(template_path))

    local = MossProcessor(
        tokenizer,
        config=MelConfig(),
        template_path=template_path,
        enable_time_marker=False,
    )

    upstream_inputs = upstream(audio=waveform, return_tensors="pt")
    local_inputs = local(audio=waveform, return_tensors="pt")

    report = {
        "model_id": args.model_id,
        "revision": args.revision,
        "audio_path": str(audio_path),
        "template_path": str(template_path),
        "comparisons": {
            "input_ids": exact_equal(
                batch_tensor(local_inputs, "input_ids"),
                batch_tensor(upstream_inputs, "input_ids"),
            ),
            "attention_mask": exact_equal(
                batch_tensor(local_inputs, "attention_mask"),
                batch_tensor(upstream_inputs, "attention_mask"),
            ),
            "audio_input_mask": exact_equal(
                batch_tensor(local_inputs, "audio_input_mask"),
                batch_tensor(upstream_inputs, "audio_input_mask"),
            ),
            "audio_data_seqlens": exact_equal(
                batch_tensor(local_inputs, "audio_data_seqlens"),
                batch_tensor(upstream_inputs, "audio_data_seqlens"),
            ),
            "audio_data": allclose(
                batch_tensor(local_inputs, "audio_data"),
                batch_tensor(upstream_inputs, "audio_data"),
            ),
        },
    }

    write_json(dump_dir / "processor_parity.json", report)
    failed = [
        name
        for name, comparison in report["comparisons"].items()
        if not (comparison.get("equal") is True or comparison.get("allclose_atol_1e-5") is True)
    ]
    print(f"processor parity report: {dump_dir / 'processor_parity.json'}")
    if failed:
        raise SystemExit(f"processor parity failed: {', '.join(failed)}")
    print("processor parity passed")


if __name__ == "__main__":
    main()
