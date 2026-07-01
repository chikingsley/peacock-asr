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


class StaticQwen3PaddedCacheBuilder(StaticQwen3CacheBuilder):
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        last_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = inputs_embeds
        keys = []
        values = []
        for layer in self.model.layers[: self.num_layers]:
            hidden_states, key_states, value_states = self._static_layer(layer, hidden_states)
            keys.append(key_states)
            values.append(value_states)
        hidden_states = self.model.norm(hidden_states)
        selected_hidden = torch.sum(
            hidden_states * last_token_mask.to(hidden_states.dtype),
            dim=1,
        )
        logits = torch.matmul(
            selected_hidden,
            self.model.embed_tokens.weight.transpose(0, 1),
        )
        return logits, torch.stack(keys), torch.stack(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MOSS Qwen3 prefill logits plus explicit KV cache to CoreML."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--package-name", default="moss_decoder_prefill_cache.mlpackage")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--mode", choices=["fixed", "padded"], default="fixed")
    parser.add_argument(
        "--validation-prompt-len",
        type=int,
        default=None,
        help="Optional exact-prefix Torch comparison length for padded mode.",
    )
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--trace-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--compute-precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--torch-check-only", action="store_true")
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


def build_last_token_mask(*, seq_len: int, prompt_len: int, dtype: torch.dtype) -> torch.Tensor:
    if prompt_len < 1 or prompt_len > seq_len:
        raise ValueError(f"prompt_len={prompt_len} must be in [1, {seq_len}]")
    mask = torch.zeros(1, seq_len, 1, dtype=dtype)
    mask[:, prompt_len - 1, :] = 1
    return mask


def padded_torch_validation(
    *,
    model: Any,
    padded_module: StaticQwen3PaddedCacheBuilder,
    seq_len: int,
    prompt_len: int,
    num_layers: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    exact_module = StaticQwen3CacheBuilder(
        model=model,
        seq_len=prompt_len,
        num_layers=num_layers,
    )
    exact_module.to(dtype=dtype)
    exact_module.eval()
    total_values = prompt_len * int(model.config.hidden_size)
    prefix = torch.linspace(-0.25, 0.25, total_values, dtype=dtype).reshape(
        1,
        prompt_len,
        int(model.config.hidden_size),
    )
    padded = torch.zeros(1, seq_len, int(model.config.hidden_size), dtype=dtype)
    padded[:, :prompt_len, :] = prefix
    last_token_mask = build_last_token_mask(
        seq_len=seq_len,
        prompt_len=prompt_len,
        dtype=dtype,
    )
    with torch.no_grad():
        exact_logits, exact_keys, exact_values = exact_module(prefix)
        padded_logits, padded_keys, padded_values = padded_module(padded, last_token_mask)
    return {
        "prompt_len": prompt_len,
        "padded_logits_vs_exact": diff_stats(
            padded_logits.detach().cpu().float().numpy(),
            exact_logits.detach().cpu().float().numpy(),
        ),
        "padded_valid_keys_vs_exact": diff_stats(
            padded_keys[:, :, :, :prompt_len, :].detach().cpu().float().numpy(),
            exact_keys.detach().cpu().float().numpy(),
        ),
        "padded_valid_values_vs_exact": diff_stats(
            padded_values[:, :, :, :prompt_len, :].detach().cpu().float().numpy(),
            exact_values.detach().cpu().float().numpy(),
        ),
    }


def export_prefill_cache(
    *,
    weights: Path,
    config: Path,
    output_dir: Path,
    package_name: str,
    seq_len: int,
    mode: str,
    validation_prompt_len: int | None,
    num_layers: int,
    trace_dtype_name: str,
    compute_precision_name: str,
    validate_predict: bool,
    overwrite: bool,
) -> dict[str, Any]:
    import coremltools as ct

    dtype = torch_dtype(trace_dtype_name)
    model = build_model(config_path=config, weights_path=weights, dtype=dtype)
    if mode == "padded":
        module = StaticQwen3PaddedCacheBuilder(
            model=model,
            seq_len=seq_len,
            num_layers=num_layers,
        )
    else:
        module = StaticQwen3CacheBuilder(
            model=model,
            seq_len=seq_len,
            num_layers=num_layers,
        )
    module.to(dtype=dtype)
    module.eval()

    inputs_embeds = torch.zeros(1, seq_len, int(model.config.hidden_size), dtype=dtype)
    last_token_mask = build_last_token_mask(seq_len=seq_len, prompt_len=seq_len, dtype=dtype)
    trace_inputs = (
        (inputs_embeds, last_token_mask) if mode == "padded" else (inputs_embeds,)
    )
    with torch.no_grad():
        torch_logits, torch_keys, torch_values = module(*trace_inputs)
        traced = torch.jit.trace(module, trace_inputs, strict=False)

    torch_validation: dict[str, Any] | None = None
    if mode == "padded" and validation_prompt_len is not None:
        torch_validation = padded_torch_validation(
            model=model,
            padded_module=module,
            seq_len=seq_len,
            prompt_len=validation_prompt_len,
            num_layers=num_layers,
            dtype=dtype,
        )

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
        ]
        + (
            [
                ct.TensorType(
                    name="last_token_mask",
                    shape=last_token_mask.shape,
                    dtype=np.float32,
                )
            ]
            if mode == "padded"
            else []
        ),
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
        predict_inputs = {"inputs_embeds": inputs_embeds.detach().cpu().float().numpy()}
        if mode == "padded":
            predict_inputs["last_token_mask"] = last_token_mask.detach().cpu().float().numpy()
        prediction = mlmodel.predict(
            predict_inputs
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
        "mode": mode,
        "validation_prompt_len": validation_prompt_len,
        "num_layers": num_layers,
        "trace_dtype": trace_dtype_name,
        "compute_precision": compute_precision_name,
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "last_token_mask_shape": list(last_token_mask.shape) if mode == "padded" else None,
        "logits_shape": list(torch_logits.shape),
        "past_keys_shape": list(torch_keys.shape),
        "past_values_shape": list(torch_values.shape),
        "torch_topk_zero_input": topk(torch_logits_np),
        "torch_validation": torch_validation,
        "coreml_validation": coreml_validation,
    }


def torch_check_prefill_cache(
    *,
    weights: Path,
    config: Path,
    seq_len: int,
    mode: str,
    validation_prompt_len: int | None,
    num_layers: int,
    trace_dtype_name: str,
) -> dict[str, Any]:
    dtype = torch_dtype(trace_dtype_name)
    model = build_model(config_path=config, weights_path=weights, dtype=dtype)
    if mode == "padded":
        module = StaticQwen3PaddedCacheBuilder(
            model=model,
            seq_len=seq_len,
            num_layers=num_layers,
        )
    else:
        module = StaticQwen3CacheBuilder(
            model=model,
            seq_len=seq_len,
            num_layers=num_layers,
        )
    module.to(dtype=dtype)
    module.eval()

    inputs_embeds = torch.zeros(1, seq_len, int(model.config.hidden_size), dtype=dtype)
    trace_inputs = (
        (
            inputs_embeds,
            build_last_token_mask(seq_len=seq_len, prompt_len=seq_len, dtype=dtype),
        )
        if mode == "padded"
        else (inputs_embeds,)
    )
    with torch.no_grad():
        torch_logits, torch_keys, torch_values = module(*trace_inputs)

    torch_validation: dict[str, Any] | None = None
    if mode == "padded" and validation_prompt_len is not None:
        torch_validation = padded_torch_validation(
            model=model,
            padded_module=module,
            seq_len=seq_len,
            prompt_len=validation_prompt_len,
            num_layers=num_layers,
            dtype=dtype,
        )

    return {
        "weights": str(weights),
        "config": str(config),
        "seq_len": seq_len,
        "mode": mode,
        "validation_prompt_len": validation_prompt_len,
        "num_layers": num_layers,
        "trace_dtype": trace_dtype_name,
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "logits_shape": list(torch_logits.shape),
        "past_keys_shape": list(torch_keys.shape),
        "past_values_shape": list(torch_values.shape),
        "torch_topk_zero_input": topk(torch_logits.detach().cpu().float().numpy()),
        "torch_validation": torch_validation,
    }


def main() -> None:
    args = parse_args()
    if args.torch_check_only:
        manifest = torch_check_prefill_cache(
            weights=args.weights.resolve(),
            config=args.config.resolve(),
            seq_len=args.seq_len,
            mode=args.mode,
            validation_prompt_len=args.validation_prompt_len,
            num_layers=args.num_layers,
            trace_dtype_name=args.trace_dtype,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    manifest = export_prefill_cache(
        weights=args.weights.resolve(),
        config=args.config.resolve(),
        output_dir=args.output_dir.resolve(),
        package_name=args.package_name,
        seq_len=args.seq_len,
        mode=args.mode,
        validation_prompt_len=args.validation_prompt_len,
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
