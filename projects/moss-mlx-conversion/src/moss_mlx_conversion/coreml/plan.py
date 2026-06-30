from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moss_mlx_conversion import DEFAULT_MODEL_ID
from moss_mlx_conversion.config import MossModelConfig
from moss_mlx_conversion.dump import write_json
from moss_mlx_conversion.paths import ARTIFACTS_DIR, MLX_DIR

DEFAULT_CONFIG_PATH = MLX_DIR / "MOSS-Transcribe-preview-2B-bf16" / "config.json"
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "coreml" / "moss-coreml-plan.json"
DEFAULT_TEMPLATE_FIXED_TOKENS = 10
BYTES_PER_FP16 = 2


@dataclass(frozen=True)
class CoreMLPlanOptions:
    max_audio_seconds: float = 30.0
    prefill_seq_len: int = 512
    max_decode_len: int = 256
    template_fixed_prompt_tokens: int = DEFAULT_TEMPLATE_FIXED_TOKENS
    batch_size: int = 1
    cache_padding_multiple: int = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a private Mobius-style CoreML conversion plan for MOSS."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Converted config.json or upstream original_config.json.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--prefill-seq-len", type=int, default=512)
    parser.add_argument("--max-decode-len", type=int, default=256)
    parser.add_argument(
        "--template-fixed-prompt-tokens",
        type=int,
        default=DEFAULT_TEMPLATE_FIXED_TOKENS,
        help="Default MOSS chat template tokens outside the audio placeholder span.",
    )
    parser.add_argument("--cache-padding-multiple", type=int, default=128)
    return parser.parse_args()


def resolve_config_path(path: Path) -> Path:
    if path.is_dir():
        converted = path / "config.json"
        if converted.exists():
            return converted
        upstream = path / "original_config.json"
        if upstream.exists():
            return upstream
    return path


def load_moss_config(path: Path) -> tuple[MossModelConfig, Path]:
    config_path = resolve_config_path(path)
    return MossModelConfig.from_json(config_path), config_path


def feature_frames_for_seconds(
    *,
    seconds: float,
    sample_rate: int,
    hop_length: int,
) -> int:
    return math.ceil((seconds * sample_rate) / hop_length)


def moss_audio_tokens_for_frames(input_lengths: int) -> int:
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def fp16_kv_cache_bytes(
    *,
    layers: int,
    batch_size: int,
    kv_heads: int,
    cache_len: int,
    head_dim: int,
) -> int:
    return layers * 2 * batch_size * kv_heads * cache_len * head_dim * BYTES_PER_FP16


def build_coreml_plan(
    *,
    config: MossModelConfig,
    config_path: Path,
    options: CoreMLPlanOptions,
) -> dict[str, Any]:
    text = config.text_config
    audio = config.audio_config
    max_mel_frames = feature_frames_for_seconds(
        seconds=options.max_audio_seconds,
        sample_rate=config.sample_rate,
        hop_length=config.mel_hop_length,
    )
    max_audio_tokens = moss_audio_tokens_for_frames(max_mel_frames)
    max_prefill_prompt_tokens = max_audio_tokens + options.template_fixed_prompt_tokens
    raw_cache_len = options.prefill_seq_len + options.max_decode_len
    padded_cache_len = max(
        text.head_dim,
        ceil_to_multiple(raw_cache_len, options.cache_padding_multiple),
    )
    cache_shape = [
        options.batch_size,
        text.num_key_value_heads,
        padded_cache_len,
        text.head_dim,
    ]
    cache_bytes = fp16_kv_cache_bytes(
        layers=text.num_hidden_layers,
        batch_size=options.batch_size,
        kv_heads=text.num_key_value_heads,
        cache_len=padded_cache_len,
        head_dim=text.head_dim,
    )
    prefill_margin = options.prefill_seq_len - max_prefill_prompt_tokens
    warnings: list[str] = []
    if prefill_margin < 0:
        warnings.append(
            "max_prefill_prompt_tokens exceeds prefill_seq_len; raise prefill_seq_len or "
            "lower max_audio_seconds."
        )
    if options.prefill_seq_len < text.head_dim:
        warnings.append(
            "prefill_seq_len is below head_dim; keep cache padding enabled to avoid the "
            "Qwen-family CoreML cache-length bug observed in Mobius."
        )

    return {
        "source": {
            "model_id": DEFAULT_MODEL_ID,
            "config_path": str(config_path),
            "private_scope": True,
            "public_actions": "none",
        },
        "model": {
            "model_type": config.model_type,
            "sample_rate": config.sample_rate,
            "mel": {
                "bins": config.mel_dim,
                "n_fft": config.mel_n_fft,
                "hop_length": config.mel_hop_length,
            },
            "audio_encoder": {
                "model_type": audio.model_type,
                "layers": audio.encoder_layers,
                "hidden_size": audio.d_model,
                "attention_heads": audio.encoder_attention_heads,
                "ffn_dim": audio.encoder_ffn_dim,
                "output_dim": audio.output_dim,
                "n_window": audio.n_window,
                "n_window_infer": audio.n_window_infer,
            },
            "adapter": {
                "hidden_size": config.adapter_hidden_size,
                "input_output_dim": audio.output_dim,
            },
            "text_decoder": {
                "model_type": text.model_type,
                "layers": text.num_hidden_layers,
                "hidden_size": text.hidden_size,
                "intermediate_size": text.intermediate_size,
                "attention_heads": text.num_attention_heads,
                "key_value_heads": text.num_key_value_heads,
                "head_dim": text.head_dim,
                "vocab_size": text.vocab_size,
                "rope_theta": text.rope_theta,
                "tie_word_embeddings": text.tie_word_embeddings,
            },
            "special_tokens": {
                "audio_placeholder_id": config.audio_placeholder_id,
                "audio_start_token_id": config.audio_start_token_id,
                "audio_end_token_id": config.audio_end_token_id,
                "start_token_id": config.start_token_id,
                "end_token_id": config.end_token_id,
            },
        },
        "shape_defaults": {
            "batch_size": options.batch_size,
            "max_audio_seconds": options.max_audio_seconds,
            "prefill_seq_len": options.prefill_seq_len,
            "max_decode_len": options.max_decode_len,
            "template_fixed_prompt_tokens": options.template_fixed_prompt_tokens,
            "cache_padding_multiple": options.cache_padding_multiple,
        },
        "derived_shapes": {
            "max_mel_frames": max_mel_frames,
            "max_audio_tokens": max_audio_tokens,
            "max_prefill_prompt_tokens": max_prefill_prompt_tokens,
            "prefill_margin_tokens": prefill_margin,
            "raw_cache_len": raw_cache_len,
            "padded_cache_len": padded_cache_len,
            "kv_cache": {
                "shape_per_layer": cache_shape,
                "total_tensors": text.num_hidden_layers * 2,
                "total_bytes_fp16": cache_bytes,
                "total_mib_fp16": round(cache_bytes / (1024 * 1024), 2),
            },
            "component_inputs": {
                "mel": [options.batch_size, config.mel_dim, max_mel_frames],
                "audio_data_seqlens": [options.batch_size],
                "prompt_input_ids": [options.batch_size, options.prefill_seq_len],
                "prompt_audio_mask": [options.batch_size, options.prefill_seq_len],
                "prefill_inputs_embeds": [
                    options.batch_size,
                    options.prefill_seq_len,
                    text.hidden_size,
                ],
                "step_input_id": [options.batch_size, 1],
            },
        },
        "components": [
            {
                "name": "moss_mel_frontend",
                "format": "host_swift_or_python",
                "status": "reuse_current_python_first_then_port",
                "contract": "16 kHz mono audio to 128-bin Whisper log-mel frames.",
            },
            {
                "name": "moss_audio_encoder_adapter.mlpackage",
                "format": "CoreML",
                "status": "planned",
                "inputs": ["mel", "audio_data_seqlens"],
                "outputs": ["audio_placeholder_embeddings"],
                "notes": [
                    "Fuse Qwen3-Omni audio encoder and gated MOSS audio adapter.",
                    "Export as fixed max_mel_frames with explicit length input.",
                ],
            },
            {
                "name": "moss_token_embedding.mlpackage",
                "format": "CoreML",
                "status": "planned",
                "inputs": ["input_ids"],
                "outputs": ["token_embeddings"],
                "notes": [
                    "Host merges token embeddings with audio_placeholder_embeddings using "
                    "audio_input_mask."
                ],
            },
            {
                "name": "moss_decoder_prefill.mlpackage",
                "format": "CoreML",
                "status": "planned",
                "inputs": ["prefill_inputs_embeds", "position_ids", "attention_mask", "kv_cache"],
                "outputs": ["updated_kv_cache", "last_hidden_state"],
                "notes": [
                    "Fixed sequence prefill avoids one-token prompt replay.",
                    "Pad unused prompt slots and mask them out.",
                ],
            },
            {
                "name": "moss_decoder_step_cache_external.mlpackage",
                "format": "CoreML",
                "status": "planned",
                "inputs": ["step_input_id", "position_id", "attention_mask", "kv_cache"],
                "outputs": ["logits", "updated_kv_cache"],
                "notes": [
                    "Use host-managed KV cache by default, following the Cohere "
                    "cache-external direction in FluidAudio/Mobius.",
                    "Keep decoder and LM head at float32 until overflow parity is ruled out.",
                ],
            },
        ],
        "mobius_prior_art": [
            {
                "reference": "mobius/models/stt/qwen3-asr-0.6b/coreml",
                "reused_idea": (
                    "split audio encoder, embedding, prefill, one-token decode, and LM head."
                ),
            },
            {
                "reference": "mobius/models/stt/qwen3-asr-0.6b/coreml/QWEN3_ASR_COREML.md",
                "reused_idea": (
                    "decoder f32 fallback, RoPE layout checks, fixed prefill, and cache padding."
                ),
            },
            {
                "reference": "mobius/models/stt/cohere-transcribe-03-2026/coreml",
                "reused_idea": "host-managed cache-external decoder contract for broad OS support.",
            },
        ],
        "moss_specific_deltas": [
            "MOSS is audio-embedding injection into a decoder-only Qwen3 stack, not an "
            "encoder-decoder cross-attention model.",
            "The prompt length is driven mostly by audio placeholders; use audio_input_mask as "
            "the injection source of truth.",
            "The audio tower is Qwen3-Omni-MoE with 32 layers and 1280 hidden size, not the "
            "Qwen3-ASR 0.6B audio stack.",
            "Baseline is English-only with time markers disabled.",
        ],
        "validation_plan": [
            "Export PyTorch reference tensors for one fixture: mel, input_ids, audio_input_mask, "
            "audio embeddings, merged embeddings, first logits, and first generated tokens.",
            "Validate CoreML audio_encoder_adapter output against PyTorch before wiring decode.",
            "Validate token_embedding plus host scatter against PyTorch merged embeddings.",
            "Validate decoder_prefill last hidden state and cache tensors against PyTorch.",
            "Validate one-token decode logits and first 5 generated token IDs.",
            "Run the existing 20-row LibriSpeech clean smoke eval before any quantization.",
            "Profile with coreml-cli on macOS and only then try encoder-only INT8 or palette work.",
        ],
        "known_risks": [
            "Actual CoreML conversion and profiling require macOS/CoreML runtime.",
            "CoreML per-token MLModel.prediction overhead can dominate this decoder-heavy model.",
            "Qwen-family RoPE layout and cache-length behavior need explicit parity tests.",
            "MOSS may remain slower than CTC/TDT ASR even when the CoreML path is correct.",
        ],
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    config, config_path = load_moss_config(args.config.resolve())
    options = CoreMLPlanOptions(
        max_audio_seconds=args.max_audio_seconds,
        prefill_seq_len=args.prefill_seq_len,
        max_decode_len=args.max_decode_len,
        template_fixed_prompt_tokens=args.template_fixed_prompt_tokens,
        cache_padding_multiple=args.cache_padding_multiple,
    )
    plan = build_coreml_plan(config=config, config_path=config_path, options=options)
    write_json(args.output, plan)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "max_mel_frames": plan["derived_shapes"]["max_mel_frames"],
                "max_audio_tokens": plan["derived_shapes"]["max_audio_tokens"],
                "prefill_margin_tokens": plan["derived_shapes"]["prefill_margin_tokens"],
                "padded_cache_len": plan["derived_shapes"]["padded_cache_len"],
                "kv_cache_mib_fp16": plan["derived_shapes"]["kv_cache"]["total_mib_fp16"],
                "warnings": plan["warnings"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
