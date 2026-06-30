from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = (
    PROJECT_ROOT
    / "artifacts/cache/huggingface/models--OpenMOSS-Team--MOSS-Transcribe-preview-2B"
    / "snapshots/c98175cb20e48bd9be4e95f6c85f2af18899f780"
)
DEFAULT_CONFIG = SNAPSHOT_DIR / "config.json"
DEFAULT_REFERENCE_TENSORS = (
    PROJECT_ROOT / "artifacts/reference/libri1-pytorch-bf16/reference_tensors.npz"
)
DEFAULT_WEIGHTS = PROJECT_ROOT / "artifacts/coreml/moss-qwen3-decoder-bf16.safetensors"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "coreml/build"
LANGUAGE_PREFIX = "model.language_model."


class StaticQwen3CacheBuilder(nn.Module):
    def __init__(self, *, model: Qwen3Model, seq_len: int, num_layers: int) -> None:
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_heads = int(model.config.num_attention_heads)
        self.num_key_value_heads = int(model.config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(model.config.head_dim)
        self.hidden_size = int(model.config.hidden_size)
        cos, sin = self._build_rope()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("causal_mask", self._build_causal_mask(), persistent=False)

    def _build_rope(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            hidden = torch.zeros(1, self.seq_len, self.hidden_size)
            position_ids = torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0)
            return self.model.rotary_emb(hidden, position_ids)

    def _build_causal_mask(self) -> torch.Tensor:
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1)
        mask = mask * -1e9
        return mask.reshape(1, 1, self.seq_len, self.seq_len)

    def _rotate_half(self, value: torch.Tensor) -> torch.Tensor:
        half_dim = self.head_dim // 2
        return torch.cat((-value[..., half_dim:], value[..., :half_dim]), dim=-1)

    def _apply_rotary_pos_emb(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos.unsqueeze(1)
        sin = self.sin.unsqueeze(1)
        query_embed = (query_states * cos) + (self._rotate_half(query_states) * sin)
        key_embed = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return query_embed, key_embed

    def _static_attention(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention = layer.self_attn
        query_states = attention.q_norm(
            attention.q_proj(hidden_states).view(
                1,
                self.seq_len,
                self.num_heads,
                self.head_dim,
            )
        ).transpose(1, 2)
        key_states = attention.k_norm(
            attention.k_proj(hidden_states).view(
                1,
                self.seq_len,
                self.num_key_value_heads,
                self.head_dim,
            )
        ).transpose(1, 2)
        value_states = attention.v_proj(hidden_states).view(
            1,
            self.seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states)
        attention_keys = key_states
        attention_values = value_states
        if self.num_key_value_groups > 1:
            attention_keys = attention_keys.repeat_interleave(self.num_key_value_groups, dim=1)
            attention_values = attention_values.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, attention_keys.transpose(2, 3))
        attn_weights = attn_weights * float(attention.scaling)
        attn_weights = attn_weights + self.causal_mask.to(attn_weights.dtype)
        attn_weights = functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, attention_values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(1, self.seq_len, self.hidden_size).contiguous()
        return attention.o_proj(attn_output), key_states, value_states

    def _static_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, key_states, value_states = self._static_attention(layer, hidden_states)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        return residual + hidden_states, key_states, value_states

    def forward(
        self,
        inputs_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = inputs_embeds
        keys = []
        values = []
        for layer in self.model.layers[: self.num_layers]:
            hidden_states, key_states, value_states = self._static_layer(layer, hidden_states)
            keys.append(key_states)
            values.append(value_states)
        hidden_states = self.model.norm(hidden_states)
        logits = torch.matmul(
            hidden_states[:, -1, :],
            self.model.embed_tokens.weight.transpose(0, 1),
        )
        return logits, torch.stack(keys), torch.stack(values)


class StaticQwen3DecoderStep(nn.Module):
    def __init__(self, *, model: Qwen3Model, past_len: int, num_layers: int) -> None:
        super().__init__()
        self.model = model
        self.past_len = past_len
        self.num_layers = num_layers
        self.num_heads = int(model.config.num_attention_heads)
        self.num_key_value_heads = int(model.config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(model.config.head_dim)
        self.hidden_size = int(model.config.hidden_size)
        cos, sin = self._build_step_rope()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _build_step_rope(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            hidden = torch.zeros(1, 1, self.hidden_size)
            position_ids = torch.tensor([[self.past_len]], dtype=torch.long)
            return self.model.rotary_emb(hidden, position_ids)

    def _rotate_half(self, value: torch.Tensor) -> torch.Tensor:
        half_dim = self.head_dim // 2
        return torch.cat((-value[..., half_dim:], value[..., :half_dim]), dim=-1)

    def _apply_rotary_pos_emb(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos.unsqueeze(1)
        sin = self.sin.unsqueeze(1)
        query_embed = (query_states * cos) + (self._rotate_half(query_states) * sin)
        key_embed = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return query_embed, key_embed

    def _static_attention(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention = layer.self_attn
        query_states = attention.q_norm(
            attention.q_proj(hidden_states).view(1, 1, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        key_states = attention.k_norm(
            attention.k_proj(hidden_states).view(1, 1, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = attention.v_proj(hidden_states).view(
            1,
            1,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states)
        updated_key = torch.cat((past_key, key_states), dim=2)
        updated_value = torch.cat((past_value, value_states), dim=2)
        attention_keys = updated_key
        attention_values = updated_value
        if self.num_key_value_groups > 1:
            attention_keys = attention_keys.repeat_interleave(self.num_key_value_groups, dim=1)
            attention_values = attention_values.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, attention_keys.transpose(2, 3))
        attn_weights = attn_weights * float(attention.scaling)
        attn_weights = functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, attention_values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(1, 1, self.hidden_size).contiguous()
        return attention.o_proj(attn_output), updated_key, updated_value

    def _static_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, updated_key, updated_value = self._static_attention(
            layer,
            hidden_states,
            past_key,
            past_value,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        return residual + hidden_states, updated_key, updated_value

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = inputs_embeds
        updated_keys = []
        updated_values = []
        for layer_idx, layer in enumerate(self.model.layers[: self.num_layers]):
            hidden_states, key_states, value_states = self._static_layer(
                layer,
                hidden_states,
                past_keys[layer_idx],
                past_values[layer_idx],
            )
            updated_keys.append(key_states)
            updated_values.append(value_states)
        hidden_states = self.model.norm(hidden_states)
        logits = torch.matmul(
            hidden_states[:, -1, :],
            self.model.embed_tokens.weight.transpose(0, 1),
        )
        return logits, torch.stack(updated_keys), torch.stack(updated_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MOSS Qwen3 one-token decoder step to CoreML."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-tensors", type=Path, default=DEFAULT_REFERENCE_TENSORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--package-name", default="moss_decoder_step_fixture.mlpackage")
    parser.add_argument("--torch-check-only", action="store_true")
    parser.add_argument("--validate-predict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trace-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--compute-precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--num-layers", type=int, default=28)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    return torch.float32


def coreml_compute_precision(name: str) -> Any:
    import coremltools as ct

    if name == "float32":
        return ct.precision.FLOAT32
    return ct.precision.FLOAT16


def load_reference_inputs(
    path: Path,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    tensors = np.load(path)
    merged_embeds = torch.from_numpy(tensors["merged_embeds"]).to(dtype=dtype)
    generated_ids = torch.from_numpy(tensors["generated_ids"]).to(dtype=torch.long)
    expected_generated_ids = tensors["generated_ids"]
    return merged_embeds, generated_ids, expected_generated_ids


def build_model(*, config_path: Path, weights_path: Path, dtype: torch.dtype) -> Qwen3Model:
    config_data = load_json(config_path)
    text_config = Qwen3Config(**config_data["language_config"])
    text_config._attn_implementation = "eager"
    model = Qwen3Model(text_config)
    tensors = load_file(str(weights_path), device="cpu")
    state = {
        key.removeprefix(LANGUAGE_PREFIX): value.to(dtype=dtype)
        for key, value in tensors.items()
        if key.startswith(LANGUAGE_PREFIX)
    }
    model.load_state_dict(state, strict=True)
    model.to(dtype=dtype)
    model.eval()
    return model


def diff_stats(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def topk(values: np.ndarray, *, k: int = 10) -> dict[str, list[float] | list[int]]:
    flat = values.reshape(-1)
    indices = np.argpartition(-flat, kth=range(k))[:k]
    indices = indices[np.argsort(-flat[indices])]
    return {
        "indices": indices.tolist(),
        "values": flat[indices].astype(float).tolist(),
    }


def build_fixture(
    *,
    model: Qwen3Model,
    reference_tensors: Path,
    dtype: torch.dtype,
    num_layers: int,
) -> dict[str, Any]:
    merged_embeds, generated_ids, expected_generated_ids = load_reference_inputs(
        reference_tensors,
        dtype=dtype,
    )
    cache_builder = StaticQwen3CacheBuilder(
        model=model,
        seq_len=int(merged_embeds.shape[1]),
        num_layers=num_layers,
    )
    step_module = StaticQwen3DecoderStep(
        model=model,
        past_len=int(merged_embeds.shape[1]),
        num_layers=num_layers,
    )
    first_token_id = generated_ids[:, 0:1]
    first_token_embeds = model.embed_tokens(first_token_id).to(dtype=dtype)
    with torch.no_grad():
        prefill_logits, past_keys, past_values = cache_builder(merged_embeds)
        step_logits, updated_keys, updated_values = step_module(
            first_token_embeds,
            past_keys,
            past_values,
        )
    return {
        "merged_embeds": merged_embeds,
        "first_token_embeds": first_token_embeds,
        "past_keys": past_keys,
        "past_values": past_values,
        "prefill_logits": prefill_logits.detach().cpu().float().numpy(),
        "step_logits": step_logits.detach().cpu().float().numpy(),
        "updated_keys": updated_keys.detach().cpu().float().numpy(),
        "updated_values": updated_values.detach().cpu().float().numpy(),
        "expected_generated_ids": expected_generated_ids,
        "first_token_id": int(first_token_id.item()),
        "second_token_id": int(generated_ids[:, 1:2].item()),
    }


def torch_check(
    *,
    weights: Path,
    config: Path,
    reference_tensors: Path,
    trace_dtype_name: str,
    num_layers: int,
) -> dict[str, Any]:
    dtype = torch_dtype(trace_dtype_name)
    model = build_model(config_path=config, weights_path=weights, dtype=dtype)
    fixture = build_fixture(
        model=model,
        reference_tensors=reference_tensors,
        dtype=dtype,
        num_layers=num_layers,
    )
    step_topk = topk(fixture["step_logits"])
    return {
        "weights": str(weights),
        "config": str(config),
        "reference_tensors": str(reference_tensors),
        "trace_dtype": trace_dtype_name,
        "num_layers": num_layers,
        "past_keys_shape": list(fixture["past_keys"].shape),
        "past_values_shape": list(fixture["past_values"].shape),
        "updated_keys_shape": list(fixture["updated_keys"].shape),
        "updated_values_shape": list(fixture["updated_values"].shape),
        "first_token_id": fixture["first_token_id"],
        "expected_second_token_id": fixture["second_token_id"],
        "step_logits_shape": list(fixture["step_logits"].shape),
        "step_logits_topk": step_topk,
        "step_top1_matches_expected_second_token": step_topk["indices"][0]
        == fixture["second_token_id"],
    }


def export_decoder_step(
    *,
    weights: Path,
    config: Path,
    reference_tensors: Path,
    output_dir: Path,
    package_name: str,
    trace_dtype_name: str,
    compute_precision_name: str,
    num_layers: int,
    validate_predict: bool,
    overwrite: bool,
) -> dict[str, Any]:
    import coremltools as ct

    dtype = torch_dtype(trace_dtype_name)
    model = build_model(config_path=config, weights_path=weights, dtype=dtype)
    fixture = build_fixture(
        model=model,
        reference_tensors=reference_tensors,
        dtype=dtype,
        num_layers=num_layers,
    )
    step_module = StaticQwen3DecoderStep(
        model=model,
        past_len=int(fixture["merged_embeds"].shape[1]),
        num_layers=num_layers,
    )
    step_module.eval()

    inputs = (
        fixture["first_token_embeds"],
        fixture["past_keys"],
        fixture["past_values"],
    )
    with torch.no_grad():
        torch_logits, torch_updated_keys, torch_updated_values = step_module(*inputs)
        traced = torch.jit.trace(step_module, inputs, strict=False)

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
                shape=fixture["first_token_embeds"].shape,
                dtype=np.float32,
            ),
            ct.TensorType(name="past_keys", shape=fixture["past_keys"].shape, dtype=np.float32),
            ct.TensorType(name="past_values", shape=fixture["past_values"].shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="updated_keys"),
            ct.TensorType(name="updated_values"),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=coreml_compute_precision(compute_precision_name),
    )
    mlmodel.save(str(package_path))

    coreml_validation: dict[str, Any] | None = None
    if validate_predict:
        prediction = mlmodel.predict(
            {
                "inputs_embeds": fixture["first_token_embeds"].detach().cpu().float().numpy(),
                "past_keys": fixture["past_keys"].detach().cpu().float().numpy(),
                "past_values": fixture["past_values"].detach().cpu().float().numpy(),
            }
        )
        coreml_logits = np.asarray(prediction["logits"])
        coreml_validation = {
            "vs_torch_logits": diff_stats(
                coreml_logits,
                torch_logits.detach().cpu().float().numpy(),
            ),
            "coreml_topk": topk(coreml_logits),
            "coreml_top1_matches_expected_second_token": topk(coreml_logits)["indices"][0]
            == fixture["second_token_id"],
        }
        if "updated_keys" in prediction:
            coreml_validation["vs_torch_updated_keys"] = diff_stats(
                np.asarray(prediction["updated_keys"]),
                torch_updated_keys.detach().cpu().float().numpy(),
            )
        if "updated_values" in prediction:
            coreml_validation["vs_torch_updated_values"] = diff_stats(
                np.asarray(prediction["updated_values"]),
                torch_updated_values.detach().cpu().float().numpy(),
            )

    torch_topk = topk(torch_logits.detach().cpu().float().numpy())
    return {
        "weights": str(weights),
        "config": str(config),
        "reference_tensors": str(reference_tensors),
        "output_package": str(package_path),
        "trace_dtype": trace_dtype_name,
        "compute_precision": compute_precision_name,
        "num_layers": num_layers,
        "past_len": int(fixture["merged_embeds"].shape[1]),
        "first_token_id": fixture["first_token_id"],
        "expected_second_token_id": fixture["second_token_id"],
        "past_keys_shape": list(fixture["past_keys"].shape),
        "past_values_shape": list(fixture["past_values"].shape),
        "updated_keys_shape": list(torch_updated_keys.shape),
        "updated_values_shape": list(torch_updated_values.shape),
        "torch_logits_shape": list(torch_logits.shape),
        "torch_logits_topk": torch_topk,
        "torch_top1_matches_expected_second_token": torch_topk["indices"][0]
        == fixture["second_token_id"],
        "coreml_validation": coreml_validation,
    }


def main() -> None:
    args = parse_args()
    if args.torch_check_only:
        manifest = torch_check(
            weights=args.weights.resolve(),
            config=args.config.resolve(),
            reference_tensors=args.reference_tensors.resolve(),
            trace_dtype_name=args.trace_dtype,
            num_layers=args.num_layers,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    manifest = export_decoder_step(
        weights=args.weights.resolve(),
        config=args.config.resolve(),
        reference_tensors=args.reference_tensors.resolve(),
        output_dir=args.output_dir.resolve(),
        package_name=args.package_name,
        trace_dtype_name=args.trace_dtype,
        compute_precision_name=args.compute_precision,
        num_layers=args.num_layers,
        validate_predict=args.validate_predict,
        overwrite=args.overwrite,
    )
    manifest_path = Path(manifest["output_package"]).with_suffix(".json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
