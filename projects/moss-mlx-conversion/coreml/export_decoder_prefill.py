from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
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
DEFAULT_SOURCE_WEIGHTS = SNAPSHOT_DIR / "model-00000-of-00001.safetensors"
DEFAULT_CONFIG = SNAPSHOT_DIR / "config.json"
DEFAULT_REFERENCE_TENSORS = (
    PROJECT_ROOT / "artifacts/reference/libri1-pytorch-bf16/reference_tensors.npz"
)
DEFAULT_EXTRACTED_WEIGHTS = PROJECT_ROOT / "artifacts/coreml/moss-qwen3-decoder-bf16.safetensors"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "coreml/build"
LANGUAGE_PREFIX = "model.language_model."


class StaticQwen3PrefillLogits(nn.Module):
    def __init__(
        self,
        *,
        model: Qwen3Model,
        seq_len: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_heads = int(model.config.num_attention_heads)
        self.num_key_value_heads = int(model.config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(model.config.head_dim)
        self.hidden_size = int(model.config.hidden_size)
        self.vocab_size = int(model.config.vocab_size)
        self.register_buffer("cos", self._build_rope()[0], persistent=False)
        self.register_buffer("sin", self._build_rope()[1], persistent=False)
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

    def _static_attention(self, layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
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
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights * float(attention.scaling)
        attn_weights = attn_weights + self.causal_mask.to(attn_weights.dtype)
        attn_weights = functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(1, self.seq_len, self.hidden_size).contiguous()
        return attention.o_proj(attn_output)

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

    def _static_layer(self, layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states = self._static_attention(layer, hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        return residual + hidden_states

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.model.layers[: self.num_layers]:
            hidden_states = self._static_layer(layer, hidden_states)
        hidden_states = self.model.norm(hidden_states)
        last_hidden = hidden_states[:, -1, :]
        return torch.matmul(last_hidden, self.model.embed_tokens.weight.transpose(0, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MOSS Qwen3 decoder prefill to CoreML.")
    parser.add_argument("--source-weights", type=Path, default=DEFAULT_SOURCE_WEIGHTS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_EXTRACTED_WEIGHTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-tensors", type=Path, default=DEFAULT_REFERENCE_TENSORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--package-name", default="moss_decoder_prefill_fixture.mlpackage")
    parser.add_argument("--extract-only", action="store_true")
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


def extract_decoder_weights(*, source: Path, output: Path) -> dict[str, Any]:
    tensors = load_file(str(source), device="cpu")
    selected = {
        key: value
        for key, value in tensors.items()
        if key.startswith(LANGUAGE_PREFIX)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(selected, str(output))
    return {
        "source_weights": str(source),
        "output": str(output),
        "tensor_count": len(selected),
        "bytes": output.stat().st_size,
    }


def load_inputs(path: Path, dtype: torch.dtype) -> tuple[torch.Tensor, np.ndarray]:
    tensors = np.load(path)
    inputs_embeds = torch.from_numpy(tensors["merged_embeds"]).to(dtype=dtype)
    expected_logits = tensors["prefill_last_logits"].astype(np.float32)
    return inputs_embeds, expected_logits


def build_module(
    *,
    config_path: Path,
    weights_path: Path,
    dtype: torch.dtype,
    seq_len: int,
    num_layers: int,
) -> StaticQwen3PrefillLogits:
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
    module = StaticQwen3PrefillLogits(
        model=model,
        seq_len=seq_len,
        num_layers=num_layers,
    )
    module.to(dtype=dtype)
    module.eval()
    return module


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


def torch_check(
    *,
    weights: Path,
    config: Path,
    reference_tensors: Path,
    trace_dtype_name: str,
    num_layers: int,
) -> dict[str, Any]:
    dtype = torch_dtype(trace_dtype_name)
    inputs_embeds, expected_logits = load_inputs(reference_tensors, dtype=dtype)
    module = build_module(
        config_path=config,
        weights_path=weights,
        dtype=dtype,
        seq_len=int(inputs_embeds.shape[1]),
        num_layers=num_layers,
    )
    with torch.no_grad():
        logits = module(inputs_embeds).detach().cpu().float().numpy()
    result: dict[str, Any] = {
        "weights": str(weights),
        "config": str(config),
        "reference_tensors": str(reference_tensors),
        "trace_dtype": trace_dtype_name,
        "num_layers": num_layers,
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "logits_shape": list(logits.shape),
        "logits_topk": topk(logits),
    }
    if num_layers == 28:
        result["torch_vs_reference_bf16"] = diff_stats(logits, expected_logits)
        result["reference_topk"] = topk(expected_logits)
    return result


def export_decoder_prefill(
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
    inputs_embeds, expected_logits = load_inputs(reference_tensors, dtype=dtype)
    module = build_module(
        config_path=config,
        weights_path=weights,
        dtype=dtype,
        seq_len=int(inputs_embeds.shape[1]),
        num_layers=num_layers,
    )
    with torch.no_grad():
        torch_logits = module(inputs_embeds).detach().cpu().float().numpy()
        traced = torch.jit.trace(module, inputs_embeds, strict=False)

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
            ct.TensorType(name="inputs_embeds", shape=inputs_embeds.shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=coreml_compute_precision(compute_precision_name),
    )
    mlmodel.save(str(package_path))

    coreml_validation: dict[str, Any] | None = None
    if validate_predict:
        prediction = mlmodel.predict(
            {"inputs_embeds": inputs_embeds.detach().cpu().float().numpy()}
        )
        output_key = "logits" if "logits" in prediction else next(iter(prediction))
        coreml_logits = np.asarray(prediction[output_key])
        coreml_validation = {
            "output_key": output_key,
            "vs_torch": diff_stats(coreml_logits, torch_logits),
            "coreml_topk": topk(coreml_logits),
        }
        if num_layers == 28:
            coreml_validation["vs_reference_bf16"] = diff_stats(
                coreml_logits,
                expected_logits,
            )

    manifest: dict[str, Any] = {
        "weights": str(weights),
        "config": str(config),
        "reference_tensors": str(reference_tensors),
        "output_package": str(package_path),
        "trace_dtype": trace_dtype_name,
        "compute_precision": compute_precision_name,
        "num_layers": num_layers,
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "torch_logits_shape": list(torch_logits.shape),
        "torch_logits_topk": topk(torch_logits),
        "coreml_validation": coreml_validation,
    }
    if num_layers == 28:
        manifest["torch_vs_reference_bf16"] = diff_stats(torch_logits, expected_logits)
        manifest["reference_topk"] = topk(expected_logits)
    return manifest


def main() -> None:
    args = parse_args()
    if args.extract_only:
        manifest = extract_decoder_weights(
            source=args.source_weights.resolve(),
            output=args.weights.resolve(),
        )
        write_json(args.weights.resolve().with_suffix(".json"), manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

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

    manifest = export_decoder_prefill(
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
