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
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)

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
DEFAULT_EXTRACTED_WEIGHTS = (
    PROJECT_ROOT / "artifacts/coreml/moss-audio-encoder-adapter-bf16.safetensors"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "coreml/build"
COMPONENT_PREFIXES = ("model.audio_model.", "model.audio_adapter.")


class MossGatedMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class AudioEncoderAdapter(nn.Module):
    def __init__(
        self,
        *,
        audio_model: Qwen3OmniMoeAudioEncoder,
        audio_adapter: MossGatedMLP,
    ) -> None:
        super().__init__()
        self.audio_model = audio_model
        self.audio_adapter = audio_adapter

    def forward(self, audio_data: torch.Tensor, audio_data_seqlens: torch.Tensor) -> torch.Tensor:
        audio_data_seqlens = audio_data_seqlens.to(torch.long)
        audio_outputs = self.audio_model(
            input_features=audio_data,
            feature_lens=audio_data_seqlens,
        )
        return self.audio_adapter(audio_outputs.last_hidden_state)


def feat_extract_output_length(length: int) -> int:
    input_lengths_leave = length % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (length // 100) * 13


def feat_extract_output_length_tensor(length: torch.Tensor) -> torch.Tensor:
    input_lengths_leave = torch.remainder(length, 100)
    feat_lengths = torch.div(input_lengths_leave - 1, 2, rounding_mode="floor") + 1
    return (
        torch.div(
            torch.div(feat_lengths - 1, 2, rounding_mode="floor"),
            2,
            rounding_mode="floor",
        )
        + 1
        + torch.div(length, 100, rounding_mode="floor") * 13
    )


def static_chunk_lengths(*, frames: int, chunk_size: int) -> list[int]:
    full_chunks, remainder = divmod(frames, chunk_size)
    lengths = [chunk_size] * full_chunks
    if remainder:
        lengths.append(remainder)
    if not lengths:
        lengths.append(chunk_size)
    return lengths


class StaticFixtureAudioEncoderAdapter(nn.Module):
    def __init__(
        self,
        *,
        audio_model: Qwen3OmniMoeAudioEncoder,
        audio_adapter: MossGatedMLP,
        frames: int,
    ) -> None:
        super().__init__()
        self.audio_model = audio_model
        self.audio_adapter = audio_adapter
        self.frames = frames
        self.chunk_size = int(audio_model.n_window * 2)
        self.chunk_lengths = static_chunk_lengths(frames=frames, chunk_size=self.chunk_size)
        self.num_chunks = len(self.chunk_lengths)
        self.max_chunk_len = max(self.chunk_lengths)
        self.valid_lens = [feat_extract_output_length(length) for length in self.chunk_lengths]
        self.max_valid_len = max(self.valid_lens)
        self.audio_seq_len = sum(self.valid_lens)
        self.freq_after_conv = ((((audio_model.num_mel_bins + 1) // 2) + 1) // 2 + 1) // 2
        self.conv_out_features = int(audio_model.conv2d3.out_channels * self.freq_after_conv)
        aftercnn_len = feat_extract_output_length(frames)
        window_aftercnn = self.max_valid_len * int(audio_model.n_window_infer // self.chunk_size)
        cu_chunk_lens = [0]
        cu_chunk_lens.extend([window_aftercnn] * (aftercnn_len // window_aftercnn))
        remainder = aftercnn_len % window_aftercnn
        if remainder:
            cu_chunk_lens.append(remainder)
        self.register_buffer(
            "cu_seqlens",
            torch.tensor(cu_chunk_lens, dtype=torch.int32).cumsum(-1),
            persistent=False,
        )

    def _fixed_chunks(self, audio_data: torch.Tensor) -> torch.Tensor:
        chunks = []
        start = 0
        for length in self.chunk_lengths:
            chunk = audio_data[:, start : start + length]
            if length < self.max_chunk_len:
                chunk = functional.pad(chunk, (0, self.max_chunk_len - length))
            chunks.append(chunk)
            start += length
        return torch.stack(chunks, dim=0)

    def _static_attention(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
        valid_key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_length = self.audio_seq_len
        query_states = attention.q_proj(hidden_states).reshape(
            seq_length,
            attention.num_heads,
            attention.head_dim,
        )
        key_states = attention.k_proj(hidden_states).reshape(
            seq_length,
            attention.num_heads,
            attention.head_dim,
        )
        value_states = attention.v_proj(hidden_states).reshape(
            seq_length,
            attention.num_heads,
            attention.head_dim,
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        if int(attention.num_key_value_groups) > 1:
            key_states = key_states.repeat_interleave(
                int(attention.num_key_value_groups),
                dim=1,
            )
            value_states = value_states.repeat_interleave(
                int(attention.num_key_value_groups),
                dim=1,
            )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights * float(attention.scaling)
        if valid_key_mask is not None:
            invalid_key_mask = torch.logical_not(valid_key_mask).reshape(1, 1, 1, seq_length)
            attn_weights = attn_weights.masked_fill(invalid_key_mask, -1e9)
        attn_weights = functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return attention.out_proj(attn_output)

    def _static_encoder_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        valid_key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = layer.self_attn_layer_norm(hidden_states)
        hidden_states = self._static_attention(layer.self_attn, hidden_states, valid_key_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.final_layer_norm(hidden_states)
        hidden_states = layer.fc1(hidden_states)
        hidden_states = layer.activation_fn(hidden_states)
        hidden_states = layer.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states

    def forward(self, audio_data: torch.Tensor, _audio_data_seqlens: torch.Tensor) -> torch.Tensor:
        padded_feature = self._fixed_chunks(audio_data).unsqueeze(1)
        padded_embed = functional.gelu(self.audio_model.conv2d1(padded_feature))
        padded_embed = functional.gelu(self.audio_model.conv2d2(padded_embed))
        padded_embed = functional.gelu(self.audio_model.conv2d3(padded_embed))
        padded_embed = self.audio_model.conv_out(
            padded_embed.permute(0, 3, 1, 2)
            .contiguous()
            .view(self.num_chunks, self.max_valid_len, self.conv_out_features)
        )
        positional_embedding = (
            self.audio_model.positional_embedding.positional_embedding[: self.max_valid_len, :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = torch.cat(
            [
                padded_embed[idx, :valid_len, :]
                for idx, valid_len in enumerate(self.valid_lens)
            ],
            dim=0,
        )

        for encoder_layer in self.audio_model.layers:
            hidden_states = self._static_encoder_layer(encoder_layer, hidden_states)

        hidden_states = self.audio_model.ln_post(hidden_states)
        hidden_states = self.audio_model.proj1(hidden_states)
        hidden_states = self.audio_model.act(hidden_states)
        hidden_states = self.audio_model.proj2(hidden_states)
        return self.audio_adapter(hidden_states)


class StaticPaddedAudioEncoderAdapter(StaticFixtureAudioEncoderAdapter):
    def forward(
        self,
        audio_data: torch.Tensor,
        audio_data_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        padded_feature = self._fixed_chunks(audio_data).unsqueeze(1)
        padded_embed = functional.gelu(self.audio_model.conv2d1(padded_feature))
        padded_embed = functional.gelu(self.audio_model.conv2d2(padded_embed))
        padded_embed = functional.gelu(self.audio_model.conv2d3(padded_embed))
        padded_embed = self.audio_model.conv_out(
            padded_embed.permute(0, 3, 1, 2)
            .contiguous()
            .view(self.num_chunks, self.max_valid_len, self.conv_out_features)
        )
        positional_embedding = (
            self.audio_model.positional_embedding.positional_embedding[: self.max_valid_len, :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = torch.cat(
            [padded_embed[idx, :valid_len, :] for idx, valid_len in enumerate(self.valid_lens)],
            dim=0,
        )

        audio_token_count = feat_extract_output_length_tensor(
            audio_data_seqlens.to(torch.long)
        ).reshape(())
        valid_key_mask = torch.arange(
            self.audio_seq_len,
            device=hidden_states.device,
            dtype=torch.long,
        ) < audio_token_count

        for encoder_layer in self.audio_model.layers:
            hidden_states = self._static_encoder_layer(
                encoder_layer,
                hidden_states,
                valid_key_mask,
            )

        hidden_states = self.audio_model.ln_post(hidden_states)
        hidden_states = self.audio_model.proj1(hidden_states)
        hidden_states = self.audio_model.act(hidden_states)
        hidden_states = self.audio_model.proj2(hidden_states)
        adapted = self.audio_adapter(hidden_states)
        return adapted * valid_key_mask.to(dtype=adapted.dtype).unsqueeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MOSS audio encoder+adapter to CoreML.")
    parser.add_argument("--source-weights", type=Path, default=DEFAULT_SOURCE_WEIGHTS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_EXTRACTED_WEIGHTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-tensors", type=Path, default=DEFAULT_REFERENCE_TENSORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--package-name", default="moss_audio_encoder_adapter_fixture.mlpackage")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Static mel-frame width for padded exports. Defaults to fixture length.",
    )
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--torch-check-only", action="store_true")
    parser.add_argument("--validate-predict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trace-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--wrapper",
        choices=["static-fixture", "static-padded", "dynamic"],
        default="static-fixture",
    )
    parser.add_argument(
        "--compute-precision",
        choices=["float16", "float32"],
        default="float16",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_audio_adapter_weights(*, source: Path, output: Path) -> dict[str, Any]:
    tensors = load_file(str(source), device="cpu")
    selected = {
        key: value
        for key, value in tensors.items()
        if key.startswith(COMPONENT_PREFIXES)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(selected, str(output))
    audio_model_tensors = [key for key in selected if key.startswith("model.audio_model.")]
    audio_adapter_tensors = [key for key in selected if key.startswith("model.audio_adapter.")]
    return {
        "source_weights": str(source),
        "output": str(output),
        "tensor_count": len(selected),
        "audio_model_tensors": len(audio_model_tensors),
        "audio_adapter_tensors": len(audio_adapter_tensors),
        "bytes": output.stat().st_size,
    }


def torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    return torch.float32


def coreml_compute_precision(name: str) -> Any:
    import coremltools as ct

    if name == "float32":
        return ct.precision.FLOAT32
    return ct.precision.FLOAT16


def build_module(
    *,
    config_path: Path,
    weights_path: Path,
    dtype: torch.dtype,
    frames: int,
    wrapper: str,
) -> AudioEncoderAdapter:
    config_data = load_json(config_path)
    audio_config = Qwen3OmniMoeAudioEncoderConfig(**config_data["audio_config"])
    audio_config._attn_implementation = "eager"
    language_config = config_data["language_config"]

    audio_model = Qwen3OmniMoeAudioEncoder(audio_config)
    audio_adapter = MossGatedMLP(
        input_size=int(audio_config.output_dim),
        hidden_size=int(config_data["adapter_hidden_size"]),
        output_size=int(language_config["hidden_size"]),
    )

    tensors = load_file(str(weights_path), device="cpu")
    audio_state = {
        key.removeprefix("model.audio_model."): value.to(dtype=dtype)
        for key, value in tensors.items()
        if key.startswith("model.audio_model.")
    }
    adapter_state = {
        key.removeprefix("model.audio_adapter."): value.to(dtype=dtype)
        for key, value in tensors.items()
        if key.startswith("model.audio_adapter.")
    }
    audio_model.load_state_dict(audio_state, strict=True)
    audio_adapter.load_state_dict(adapter_state, strict=True)

    if wrapper == "dynamic":
        module: nn.Module = AudioEncoderAdapter(
            audio_model=audio_model,
            audio_adapter=audio_adapter,
        )
    elif wrapper == "static-padded":
        module = StaticPaddedAudioEncoderAdapter(
            audio_model=audio_model,
            audio_adapter=audio_adapter,
            frames=frames,
        )
    else:
        module = StaticFixtureAudioEncoderAdapter(
            audio_model=audio_model,
            audio_adapter=audio_adapter,
            frames=frames,
        )
    module.to(dtype=dtype)
    module.eval()
    return module  # type: ignore[return-value]


def load_fixture_inputs(
    path: Path,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    tensors = np.load(path)
    audio_data = torch.from_numpy(tensors["audio_data"]).to(dtype=dtype)
    seqlens = torch.from_numpy(tensors["audio_data_seqlens"]).to(dtype=torch.int32)
    expected = tensors["audio_embeds"].astype(np.float32)
    return audio_data, seqlens, expected


def diff_stats(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def diff_stats_prefix(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape == expected.shape:
        return diff_stats(actual, expected)
    if actual.ndim != expected.ndim or actual.shape[0] < expected.shape[0]:
        raise ValueError(f"cannot prefix-compare shapes {actual.shape} and {expected.shape}")
    prefix_actual = actual[: expected.shape[0], ...]
    stats = diff_stats(prefix_actual, expected)
    stats["actual_full_shape"] = list(actual.shape)
    stats["compared_prefix_shape"] = list(prefix_actual.shape)
    return stats


def pad_audio_data(audio_data: torch.Tensor, frames: int) -> torch.Tensor:
    current_frames = int(audio_data.shape[-1])
    if current_frames == frames:
        return audio_data
    if current_frames > frames:
        return audio_data[..., :frames]
    return functional.pad(audio_data, (0, frames - current_frames))


def export_audio_encoder_adapter(
    *,
    weights: Path,
    config: Path,
    reference_tensors: Path,
    output_dir: Path,
    package_name: str,
    trace_dtype_name: str,
    compute_precision_name: str,
    wrapper: str,
    frames: int | None,
    validate_predict: bool,
    overwrite: bool,
) -> dict[str, Any]:
    import coremltools as ct

    dtype = torch_dtype(trace_dtype_name)
    audio_data, seqlens, expected_audio_embeds = load_fixture_inputs(reference_tensors, dtype=dtype)
    trace_frames = int(audio_data.shape[-1]) if frames is None else frames
    if trace_frames < int(audio_data.shape[-1]):
        raise ValueError(
            f"--frames {trace_frames} is shorter than fixture length {int(audio_data.shape[-1])}"
        )
    trace_audio_data = (
        pad_audio_data(audio_data, trace_frames) if wrapper == "static-padded" else audio_data
    )
    module = build_module(
        config_path=config,
        weights_path=weights,
        dtype=dtype,
        frames=trace_frames,
        wrapper=wrapper,
    )
    with torch.no_grad():
        torch_output = module(trace_audio_data, seqlens).detach().cpu().float().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / package_name
    if package_path.exists():
        if not overwrite:
            raise FileExistsError(f"{package_path} exists; pass --overwrite to replace it")
        shutil.rmtree(package_path)

    with torch.no_grad():
        traced = torch.jit.trace(module, (trace_audio_data, seqlens), strict=False)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="audio_data", shape=trace_audio_data.shape, dtype=np.float32),
            ct.TensorType(name="audio_data_seqlens", shape=seqlens.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="audio_embeddings")],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=coreml_compute_precision(compute_precision_name),
    )
    mlmodel.save(str(package_path))

    coreml_validation: dict[str, Any] | None = None
    if validate_predict:
        prediction = mlmodel.predict(
            {
                "audio_data": trace_audio_data.detach().cpu().float().numpy(),
                "audio_data_seqlens": seqlens.detach().cpu().numpy().astype(np.int32),
            }
        )
        output_key = (
            "audio_embeddings" if "audio_embeddings" in prediction else next(iter(prediction))
        )
        coreml_output = np.asarray(prediction[output_key])
        coreml_validation = {
            "output_key": output_key,
            "vs_torch": diff_stats(coreml_output, torch_output),
            "vs_reference_bf16": diff_stats_prefix(coreml_output, expected_audio_embeds),
        }

    return {
        "weights": str(weights),
        "config": str(config),
        "reference_tensors": str(reference_tensors),
        "output_package": str(package_path),
        "trace_dtype": trace_dtype_name,
        "compute_precision": compute_precision_name,
        "wrapper": wrapper,
        "audio_data_shape": list(trace_audio_data.shape),
        "audio_data_seqlens": seqlens.detach().cpu().tolist(),
        "torch_vs_reference_bf16": diff_stats_prefix(torch_output, expected_audio_embeds),
        "coreml_validation": coreml_validation,
    }


def check_torch_audio_encoder_adapter(
    *,
    weights: Path,
    config: Path,
    reference_tensors: Path,
    trace_dtype_name: str,
    wrapper: str,
    frames: int | None,
) -> dict[str, Any]:
    dtype = torch_dtype(trace_dtype_name)
    audio_data, seqlens, expected_audio_embeds = load_fixture_inputs(reference_tensors, dtype=dtype)
    trace_frames = int(audio_data.shape[-1]) if frames is None else frames
    if trace_frames < int(audio_data.shape[-1]):
        raise ValueError(
            f"--frames {trace_frames} is shorter than fixture length {int(audio_data.shape[-1])}"
        )
    trace_audio_data = (
        pad_audio_data(audio_data, trace_frames) if wrapper == "static-padded" else audio_data
    )
    module = build_module(
        config_path=config,
        weights_path=weights,
        dtype=dtype,
        frames=trace_frames,
        wrapper=wrapper,
    )
    with torch.no_grad():
        torch_output = module(trace_audio_data, seqlens).detach().cpu().float().numpy()
    return {
        "weights": str(weights),
        "config": str(config),
        "reference_tensors": str(reference_tensors),
        "trace_dtype": trace_dtype_name,
        "wrapper": wrapper,
        "audio_data_shape": list(trace_audio_data.shape),
        "audio_data_seqlens": seqlens.detach().cpu().tolist(),
        "torch_vs_reference_bf16": diff_stats_prefix(torch_output, expected_audio_embeds),
    }


def main() -> None:
    args = parse_args()
    if args.extract_only:
        manifest = extract_audio_adapter_weights(
            source=args.source_weights.resolve(),
            output=args.weights.resolve(),
        )
        write_json(args.weights.resolve().with_suffix(".json"), manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.torch_check_only:
        manifest = check_torch_audio_encoder_adapter(
            weights=args.weights.resolve(),
            config=args.config.resolve(),
            reference_tensors=args.reference_tensors.resolve(),
            trace_dtype_name=args.trace_dtype,
            wrapper=args.wrapper,
            frames=args.frames,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    manifest = export_audio_encoder_adapter(
        weights=args.weights.resolve(),
        config=args.config.resolve(),
        reference_tensors=args.reference_tensors.resolve(),
        output_dir=args.output_dir.resolve(),
        package_name=args.package_name,
        trace_dtype_name=args.trace_dtype,
        compute_precision_name=args.compute_precision,
        wrapper=args.wrapper,
        frames=args.frames,
        validate_predict=args.validate_predict,
        overwrite=args.overwrite,
    )
    manifest_path = Path(manifest["output_package"]).with_suffix(".json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
