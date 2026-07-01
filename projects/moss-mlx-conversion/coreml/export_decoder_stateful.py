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
    DEFAULT_REFERENCE_TENSORS,
    DEFAULT_WEIGHTS,
    build_fixture,
    build_model,
    coreml_compute_precision,
    diff_stats,
    topk,
    torch_dtype,
    write_json,
)
from torch import nn
from torch.nn import functional
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model


class StatefulFusedQwen3Decoder(nn.Module):
    """MOSS Qwen3 decoder with CoreML State-style KV cache buffers.

    The wrapper follows the Mobius Qwen3-ASR stateful decoder pattern: KV cache
    buffers are registered as module buffers, updated in-place, and later mapped
    to CoreML StateType tensors during conversion. The final RMSNorm and tied LM
    head projection are fused so each call returns logits for the last query
    position.
    """

    def __init__(
        self,
        *,
        model: Qwen3Model,
        cache_len: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.cache_len = cache_len
        self.num_layers = num_layers
        self.num_heads = int(model.config.num_attention_heads)
        self.num_key_value_heads = int(model.config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(model.config.head_dim)
        self.hidden_size = int(model.config.hidden_size)

        for layer_idx in range(num_layers):
            self.register_buffer(
                f"k_cache_{layer_idx}",
                torch.zeros(
                    1,
                    self.num_key_value_heads,
                    cache_len,
                    self.head_dim,
                    dtype=torch.float16,
                ),
            )
            self.register_buffer(
                f"v_cache_{layer_idx}",
                torch.zeros(
                    1,
                    self.num_key_value_heads,
                    cache_len,
                    self.head_dim,
                    dtype=torch.float16,
                ),
            )

    def reset_state(self) -> None:
        for layer_idx in range(self.num_layers):
            getattr(self, f"k_cache_{layer_idx}").zero_()
            getattr(self, f"v_cache_{layer_idx}").zero_()

    def _rotate_half(self, value: torch.Tensor) -> torch.Tensor:
        half_dim = self.head_dim // 2
        return torch.cat((-value[..., half_dim:], value[..., :half_dim]), dim=-1)

    def _apply_rotary_pos_emb(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query_embed = (query_states * cos) + (self._rotate_half(query_states) * sin)
        key_embed = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return query_embed, key_embed

    def _static_attention(
        self,
        layer: nn.Module,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        attention = layer.self_attn
        query_len = hidden_states.shape[1]
        end_step = attention_mask.shape[-1]
        past_len = end_step - query_len

        query_states = attention.q_norm(
            attention.q_proj(hidden_states).view(
                1,
                query_len,
                self.num_heads,
                self.head_dim,
            )
        ).transpose(1, 2)
        key_states = attention.k_norm(
            attention.k_proj(hidden_states).view(
                1,
                query_len,
                self.num_key_value_heads,
                self.head_dim,
            )
        ).transpose(1, 2)
        value_states = attention.v_proj(hidden_states).view(
            1,
            query_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        query_states, key_states = self._apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        key_cache = getattr(self, f"k_cache_{layer_idx}")
        value_cache = getattr(self, f"v_cache_{layer_idx}")
        key_cache[:, :, past_len:end_step, :] = key_states.to(dtype=key_cache.dtype)
        value_cache[:, :, past_len:end_step, :] = value_states.to(dtype=value_cache.dtype)

        attention_keys = key_cache[:, :, :end_step, :].to(dtype=query_states.dtype)
        attention_values = value_cache[:, :, :end_step, :].to(dtype=query_states.dtype)
        if self.num_key_value_groups > 1:
            attention_keys = attention_keys.repeat_interleave(self.num_key_value_groups, dim=1)
            attention_values = attention_values.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, attention_keys.transpose(2, 3))
        attn_weights = attn_weights * float(attention.scaling)
        attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)
        attn_weights = functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, attention_values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(1, query_len, self.hidden_size).contiguous()
        return attention.o_proj(attn_output)

    def _static_layer(
        self,
        layer: nn.Module,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states = self._static_attention(
            layer,
            layer_idx,
            hidden_states,
            attention_mask,
            cos,
            sin,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        return residual + hidden_states

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.model.layers[: self.num_layers]):
            hidden_states = self._static_layer(
                layer,
                layer_idx,
                hidden_states,
                attention_mask,
                cos,
                sin,
            )
        hidden_states = self.model.norm(hidden_states)
        last_hidden = hidden_states[:, -1, :]
        return torch.matmul(last_hidden, self.model.embed_tokens.weight.transpose(0, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MOSS Qwen3 fused stateful decoder to CoreML."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-tensors", type=Path, default=DEFAULT_REFERENCE_TENSORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--package-name", default="moss_decoder_stateful_fused.mlpackage")
    parser.add_argument("--torch-check-only", action="store_true")
    parser.add_argument("--validate-predict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trace-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--compute-precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--cache-len", type=int, default=768)
    parser.add_argument("--trace-query-len", type=int, default=1)
    parser.add_argument("--trace-end-step", type=int, default=5)
    parser.add_argument(
        "--default-query-len",
        type=int,
        default=None,
        help="CoreML RangeDim default for query length. Defaults to --trace-query-len.",
    )
    parser.add_argument(
        "--default-end-step",
        type=int,
        default=None,
        help="CoreML RangeDim default for attention-mask end step. Defaults to --trace-end-step.",
    )
    return parser.parse_args()


def rope_for_positions(
    *,
    model: Qwen3Model,
    start: int,
    length: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        hidden = torch.zeros(1, length, int(model.config.hidden_size), dtype=dtype)
        position_ids = torch.arange(start, start + length, dtype=torch.long).unsqueeze(0)
        cos, sin = model.rotary_emb(hidden, position_ids)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def causal_mask(length: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.triu(torch.ones(length, length, dtype=dtype), diagonal=1)
    return (mask * -1e9).reshape(1, 1, length, length)


def build_stateful_fixture(
    *,
    model: Qwen3Model,
    reference_tensors: Path,
    dtype: torch.dtype,
    num_layers: int,
    cache_len: int,
) -> dict[str, Any]:
    fixture = build_fixture(
        model=model,
        reference_tensors=reference_tensors,
        dtype=dtype,
        num_layers=num_layers,
    )
    prompt_len = int(fixture["merged_embeds"].shape[1])
    if cache_len <= prompt_len:
        raise ValueError(f"cache_len={cache_len} must be greater than prompt_len={prompt_len}")

    prefill_cos, prefill_sin = rope_for_positions(
        model=model,
        start=0,
        length=prompt_len,
        dtype=dtype,
    )
    step_cos, step_sin = rope_for_positions(
        model=model,
        start=prompt_len,
        length=1,
        dtype=dtype,
    )
    prefill_mask = causal_mask(prompt_len, dtype=dtype)
    step_mask = torch.zeros(1, 1, 1, prompt_len + 1, dtype=dtype)

    stateful = StatefulFusedQwen3Decoder(
        model=model,
        cache_len=cache_len,
        num_layers=num_layers,
    )
    stateful.eval()
    with torch.no_grad():
        stateful_prefill_logits = stateful(
            fixture["merged_embeds"],
            prefill_cos,
            prefill_sin,
            prefill_mask,
        )
        stateful_step_logits = stateful(
            fixture["first_token_embeds"],
            step_cos,
            step_sin,
            step_mask,
        )

    return {
        **fixture,
        "cache_len": cache_len,
        "prompt_len": prompt_len,
        "prefill_cos": prefill_cos,
        "prefill_sin": prefill_sin,
        "step_cos": step_cos,
        "step_sin": step_sin,
        "prefill_mask": prefill_mask,
        "step_mask": step_mask,
        "stateful_prefill_logits": stateful_prefill_logits.detach().cpu().float().numpy(),
        "stateful_step_logits": stateful_step_logits.detach().cpu().float().numpy(),
    }


def torch_check_stateful(
    *,
    weights: Path,
    config: Path,
    reference_tensors: Path,
    trace_dtype_name: str,
    num_layers: int,
    cache_len: int,
) -> dict[str, Any]:
    dtype = torch_dtype(trace_dtype_name)
    model = build_model(config_path=config, weights_path=weights, dtype=dtype)
    fixture = build_stateful_fixture(
        model=model,
        reference_tensors=reference_tensors,
        dtype=dtype,
        num_layers=num_layers,
        cache_len=cache_len,
    )
    stateful_prefill_topk = topk(fixture["stateful_prefill_logits"])
    stateful_step_topk = topk(fixture["stateful_step_logits"])
    return {
        "weights": str(weights),
        "config": str(config),
        "reference_tensors": str(reference_tensors),
        "trace_dtype": trace_dtype_name,
        "num_layers": num_layers,
        "cache_len": cache_len,
        "prompt_len": fixture["prompt_len"],
        "first_token_id": fixture["first_token_id"],
        "second_token_id": fixture["second_token_id"],
        "stateful_prefill_topk": stateful_prefill_topk,
        "stateful_step_topk": stateful_step_topk,
        "stateful_prefill_top1_matches_first_token": stateful_prefill_topk["indices"][0]
        == fixture["first_token_id"],
        "stateful_step_top1_matches_second_token": stateful_step_topk["indices"][0]
        == fixture["second_token_id"],
        "stateful_vs_static_prefill_logits": diff_stats(
            fixture["stateful_prefill_logits"],
            fixture["prefill_logits"],
        ),
        "stateful_vs_static_step_logits": diff_stats(
            fixture["stateful_step_logits"],
            fixture["step_logits"],
        ),
    }


def reset_traced_state(module: torch.jit.ScriptModule, *, num_layers: int) -> None:
    for layer_idx in range(num_layers):
        getattr(module, f"k_cache_{layer_idx}").zero_()
        getattr(module, f"v_cache_{layer_idx}").zero_()


def export_stateful_decoder(
    *,
    weights: Path,
    config: Path,
    reference_tensors: Path,
    output_dir: Path,
    package_name: str,
    trace_dtype_name: str,
    compute_precision_name: str,
    num_layers: int,
    cache_len: int,
    trace_query_len: int,
    trace_end_step: int,
    default_query_len: int | None,
    default_end_step: int | None,
    validate_predict: bool,
    overwrite: bool,
) -> dict[str, Any]:
    import coremltools as ct

    dtype = torch_dtype(trace_dtype_name)
    model = build_model(config_path=config, weights_path=weights, dtype=dtype)
    fixture = build_stateful_fixture(
        model=model,
        reference_tensors=reference_tensors,
        dtype=dtype,
        num_layers=num_layers,
        cache_len=cache_len,
    )
    stateful = StatefulFusedQwen3Decoder(
        model=model,
        cache_len=cache_len,
        num_layers=num_layers,
    )
    stateful.eval()

    if trace_end_step < trace_query_len:
        raise ValueError(
            f"trace_end_step={trace_end_step} must be >= trace_query_len={trace_query_len}"
        )
    default_query_len = trace_query_len if default_query_len is None else default_query_len
    default_end_step = trace_end_step if default_end_step is None else default_end_step
    if not 1 <= default_query_len <= cache_len:
        raise ValueError(
            f"default_query_len={default_query_len} must be in [1, {cache_len}]"
        )
    if not 1 <= default_end_step <= cache_len:
        raise ValueError(f"default_end_step={default_end_step} must be in [1, {cache_len}]")
    if default_end_step < default_query_len:
        raise ValueError(
            f"default_end_step={default_end_step} must be >= default_query_len={default_query_len}"
        )
    trace_inputs = (
        torch.zeros(1, trace_query_len, int(model.config.hidden_size), dtype=dtype),
        torch.zeros(1, trace_query_len, int(model.config.head_dim), dtype=dtype),
        torch.zeros(1, trace_query_len, int(model.config.head_dim), dtype=dtype),
        torch.zeros(1, 1, trace_query_len, trace_end_step, dtype=dtype),
    )
    with torch.no_grad():
        traced = torch.jit.trace(stateful, trace_inputs, strict=False)
    stateful.reset_state()
    reset_traced_state(traced, num_layers=num_layers)

    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / package_name
    if package_path.exists():
        if not overwrite:
            raise FileExistsError(f"{package_path} exists; pass --overwrite to replace it")
        shutil.rmtree(package_path)

    hidden_size = int(model.config.hidden_size)
    head_dim = int(model.config.head_dim)
    num_key_value_heads = int(model.config.num_key_value_heads)
    query_length = ct.RangeDim(lower_bound=1, upper_bound=cache_len, default=default_query_len)
    end_step = ct.RangeDim(lower_bound=1, upper_bound=cache_len, default=default_end_step)
    states = []
    for layer_idx in range(num_layers):
        states.extend(
            [
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(1, num_key_value_heads, cache_len, head_dim),
                        dtype=np.float16,
                    ),
                    name=f"k_cache_{layer_idx}",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(1, num_key_value_heads, cache_len, head_dim),
                        dtype=np.float16,
                    ),
                    name=f"v_cache_{layer_idx}",
                ),
            ]
        )

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="inputs_embeds",
                shape=(1, query_length, hidden_size),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="cos",
                shape=(1, query_length, head_dim),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="sin",
                shape=(1, query_length, head_dim),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(1, 1, query_length, end_step),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="logits")],
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=coreml_compute_precision(compute_precision_name),
    )
    mlmodel.save(str(package_path))

    coreml_validation: dict[str, Any] | None = None
    if validate_predict:
        state = mlmodel.make_state()
        prefill_prediction = mlmodel.predict(
            {
                "inputs_embeds": fixture["merged_embeds"].detach().cpu().float().numpy(),
                "cos": fixture["prefill_cos"].detach().cpu().float().numpy(),
                "sin": fixture["prefill_sin"].detach().cpu().float().numpy(),
                "attention_mask": fixture["prefill_mask"].detach().cpu().float().numpy(),
            },
            state=state,
        )
        step_prediction = mlmodel.predict(
            {
                "inputs_embeds": fixture["first_token_embeds"].detach().cpu().float().numpy(),
                "cos": fixture["step_cos"].detach().cpu().float().numpy(),
                "sin": fixture["step_sin"].detach().cpu().float().numpy(),
                "attention_mask": fixture["step_mask"].detach().cpu().float().numpy(),
            },
            state=state,
        )
        coreml_prefill_logits = np.asarray(prefill_prediction["logits"])
        coreml_step_logits = np.asarray(step_prediction["logits"])
        coreml_prefill_topk = topk(coreml_prefill_logits)
        coreml_step_topk = topk(coreml_step_logits)
        coreml_validation = {
            "prefill_topk": coreml_prefill_topk,
            "step_topk": coreml_step_topk,
            "prefill_top1_matches_first_token": coreml_prefill_topk["indices"][0]
            == fixture["first_token_id"],
            "step_top1_matches_second_token": coreml_step_topk["indices"][0]
            == fixture["second_token_id"],
            "vs_torch_stateful_prefill_logits": diff_stats(
                coreml_prefill_logits,
                fixture["stateful_prefill_logits"],
            ),
            "vs_torch_stateful_step_logits": diff_stats(
                coreml_step_logits,
                fixture["stateful_step_logits"],
            ),
            "vs_static_prefill_logits": diff_stats(
                coreml_prefill_logits,
                fixture["prefill_logits"],
            ),
            "vs_static_step_logits": diff_stats(
                coreml_step_logits,
                fixture["step_logits"],
            ),
        }

    stateful_prefill_topk = topk(fixture["stateful_prefill_logits"])
    stateful_step_topk = topk(fixture["stateful_step_logits"])
    return {
        "weights": str(weights),
        "config": str(config),
        "reference_tensors": str(reference_tensors),
        "output_package": str(package_path),
        "trace_dtype": trace_dtype_name,
        "compute_precision": compute_precision_name,
        "num_layers": num_layers,
        "cache_len": cache_len,
        "trace_query_len": trace_query_len,
        "trace_end_step": trace_end_step,
        "default_query_len": default_query_len,
        "default_end_step": default_end_step,
        "prompt_len": fixture["prompt_len"],
        "first_token_id": fixture["first_token_id"],
        "second_token_id": fixture["second_token_id"],
        "state_count": num_layers * 2,
        "state_shape_per_layer": [1, int(model.config.num_key_value_heads), cache_len, head_dim],
        "state_dtype": "float16",
        "minimum_deployment_target": "macOS15",
        "stateful_prefill_topk": stateful_prefill_topk,
        "stateful_step_topk": stateful_step_topk,
        "stateful_prefill_top1_matches_first_token": stateful_prefill_topk["indices"][0]
        == fixture["first_token_id"],
        "stateful_step_top1_matches_second_token": stateful_step_topk["indices"][0]
        == fixture["second_token_id"],
        "stateful_vs_static_prefill_logits": diff_stats(
            fixture["stateful_prefill_logits"],
            fixture["prefill_logits"],
        ),
        "stateful_vs_static_step_logits": diff_stats(
            fixture["stateful_step_logits"],
            fixture["step_logits"],
        ),
        "coreml_validation": coreml_validation,
    }


def main() -> None:
    args = parse_args()
    if args.torch_check_only:
        manifest = torch_check_stateful(
            weights=args.weights.resolve(),
            config=args.config.resolve(),
            reference_tensors=args.reference_tensors.resolve(),
            trace_dtype_name=args.trace_dtype,
            num_layers=args.num_layers,
            cache_len=args.cache_len,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    manifest = export_stateful_decoder(
        weights=args.weights.resolve(),
        config=args.config.resolve(),
        reference_tensors=args.reference_tensors.resolve(),
        output_dir=args.output_dir.resolve(),
        package_name=args.package_name,
        trace_dtype_name=args.trace_dtype,
        compute_precision_name=args.compute_precision,
        num_layers=args.num_layers,
        cache_len=args.cache_len,
        trace_query_len=args.trace_query_len,
        trace_end_step=args.trace_end_step,
        default_query_len=args.default_query_len,
        default_end_step=args.default_end_step,
        validate_predict=args.validate_predict,
        overwrite=args.overwrite,
    )
    manifest_path = Path(manifest["output_package"]).with_suffix(".json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
