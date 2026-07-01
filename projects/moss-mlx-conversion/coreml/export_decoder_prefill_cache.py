from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from export_decoder_step import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WEIGHTS,
    StaticQwen3CacheBuilder,
    build_model,
    coreml_compute_precision,
    topk,
    torch_dtype,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MOSS Qwen3 prefill logits plus explicit KV cache to CoreML."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--package-name", default="moss_decoder_prefill_cache.mlpackage")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--trace-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--compute-precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--validate-predict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def diff_stats(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def export_prefill_cache(
    *,
    weights: Path,
    config: Path,
    output_dir: Path,
    package_name: str,
    seq_len: int,
    num_layers: int,
    trace_dtype_name: str,
    compute_precision_name: str,
    validate_predict: bool,
    overwrite: bool,
) -> dict[str, Any]:
    import coremltools as ct

    dtype = torch_dtype(trace_dtype_name)
    model = build_model(config_path=config, weights_path=weights, dtype=dtype)
    module = StaticQwen3CacheBuilder(
        model=model,
        seq_len=seq_len,
        num_layers=num_layers,
    )
    module.to(dtype=dtype)
    module.eval()

    inputs_embeds = torch.zeros(1, seq_len, int(model.config.hidden_size), dtype=dtype)
    with torch.no_grad():
        torch_logits, torch_keys, torch_values = module(inputs_embeds)
        traced = torch.jit.trace(module, (inputs_embeds,), strict=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / package_name
    if package_path.exists():
        if not overwrite:
            raise FileExistsError(f"{package_path} exists; pass --overwrite to replace it")
        shutil.rmtree(package_path)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="inputs_embeds",
                shape=inputs_embeds.shape,
                dtype=np.float32,
            ),
        ],
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="past_keys"),
            ct.TensorType(name="past_values"),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=coreml_compute_precision(compute_precision_name),
    )
    mlmodel.save(str(package_path))

    coreml_validation: dict[str, Any] | None = None
    if validate_predict:
        prediction = mlmodel.predict(
            {"inputs_embeds": inputs_embeds.detach().cpu().float().numpy()}
        )
        coreml_validation = {
            "vs_torch_logits": diff_stats(
                np.asarray(prediction["logits"]),
                torch_logits.detach().cpu().float().numpy(),
            ),
            "vs_torch_past_keys": diff_stats(
                np.asarray(prediction["past_keys"]),
                torch_keys.detach().cpu().float().numpy(),
            ),
            "vs_torch_past_values": diff_stats(
                np.asarray(prediction["past_values"]),
                torch_values.detach().cpu().float().numpy(),
            ),
            "coreml_topk": topk(np.asarray(prediction["logits"])),
        }

    torch_logits_np = torch_logits.detach().cpu().float().numpy()
    return {
        "weights": str(weights),
        "config": str(config),
        "output_package": str(package_path),
        "seq_len": seq_len,
        "num_layers": num_layers,
        "trace_dtype": trace_dtype_name,
        "compute_precision": compute_precision_name,
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "logits_shape": list(torch_logits.shape),
        "past_keys_shape": list(torch_keys.shape),
        "past_values_shape": list(torch_values.shape),
        "torch_topk_zero_input": topk(torch_logits_np),
        "coreml_validation": coreml_validation,
    }


def main() -> None:
    args = parse_args()
    manifest = export_prefill_cache(
        weights=args.weights.resolve(),
        config=args.config.resolve(),
        output_dir=args.output_dir.resolve(),
        package_name=args.package_name,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        trace_dtype_name=args.trace_dtype,
        compute_precision_name=args.compute_precision,
        validate_predict=args.validate_predict,
        overwrite=args.overwrite,
    )
    manifest_path = Path(manifest["output_package"]).with_suffix(".json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
