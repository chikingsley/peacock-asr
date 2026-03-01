#!/usr/bin/env python3
"""
Convert HuggingFace Transformers Kyutai STT weights to Candle/native Kyutai format.

This script converts fine-tuned transformers models to the format expected by
the Kyutai Rust/Candle inference server.

Usage:
    uv run python convert_transformers_to_candle.py \
        --input ./kyutai-finetuned \
        --output ./kyutai-finetuned-candle/model.safetensors

The input can be either:
- A directory containing model.safetensors and config.json
- A HuggingFace model ID (e.g., "kyutai/stt-1b-en_fr-trfs")

Note: The codec_model weights are NOT included in the output as the Candle
server loads them separately from kyutai/stt-*-candle/mimi-*.safetensors.
"""

import argparse
import os
from typing import Dict

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig


def inverse_rotary_permute(w: torch.Tensor, n_heads: int, hidden_size: int) -> torch.Tensor:
    """
    Reverse the rotary embedding permutation applied during forward conversion.

    The HuggingFace converter applies this permutation to Q and K:
        w.view(n_heads, dim1//n_heads//2, 2, dim2).transpose(1,2).reshape(dim1, dim2)

    This interleaves dimensions for rotary position encoding.
    We reverse by un-interleaving.
    """
    head_dim = hidden_size // n_heads
    # Reshape: [hidden, hidden] -> [n_heads, 2, head_dim//2, hidden]
    w = w.view(n_heads, 2, head_dim // 2, hidden_size)
    # Transpose back: [n_heads, 2, head_dim//2, hidden] -> [n_heads, head_dim//2, 2, hidden]
    w = w.transpose(1, 2)
    # Reshape back: [hidden, hidden]
    w = w.reshape(hidden_size, hidden_size)
    return w


def convert_transformers_to_candle(
    input_path: str,
    output_path: str,
    verbose: bool = True
) -> None:
    """
    Convert HuggingFace Transformers Kyutai STT weights to Candle format.

    The main conversions are:
    1. Split combined embedding into text_emb + emb.{0-31}
    2. Combine separate Q/K/V projections into in_proj_weight
    3. Rename layer components (mlp.fc1 -> gating.linear_in, etc.)
    4. Reshape norm weights from [hidden] to [1, 1, hidden]
    5. Skip codec_model weights (loaded separately by Candle)
    """
    # Determine if input is a directory or model ID
    if os.path.isdir(input_path):
        model_dir = input_path
        weights_path = os.path.join(input_path, "model.safetensors")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(input_path, "pytorch_model.bin")
    else:
        from huggingface_hub import hf_hub_download
        model_dir = input_path
        weights_path = hf_hub_download(input_path, "model.safetensors")

    # Load config
    if verbose:
        print(f"Loading config from {model_dir}...")
    config = AutoConfig.from_pretrained(model_dir)

    # Load weights
    if verbose:
        print(f"Loading weights from {weights_path}...")
    if weights_path.endswith(".bin"):
        state_dict = torch.load(weights_path, map_location="cpu")
    else:
        state_dict = load_file(weights_path)

    if verbose:
        print(f"Loaded {len(state_dict)} tensors")

    # Get model dimensions from config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_layers = config.num_hidden_layers
    num_codebooks = config.num_codebooks
    codebook_vocab_size = config.codebook_vocab_size
    text_vocab_size = config.vocab_size

    if verbose:
        print(f"\nModel config:")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num layers: {num_layers}")
        print(f"  Num heads: {num_heads}")
        print(f"  Num codebooks: {num_codebooks}")
        print(f"  Codebook vocab size: {codebook_vocab_size}")
        print(f"  Text vocab size: {text_vocab_size}")

    # Output state dict
    candle_state_dict: Dict[str, torch.Tensor] = {}

    # Step 1: Split combined embedding
    if verbose:
        print("\nStep 1: Splitting combined embedding...")

    embed_key = "model.embed_tokens.embed_tokens.weight"
    if embed_key in state_dict:
        combined_emb = state_dict[embed_key]
        if verbose:
            print(f"  Combined embedding shape: {combined_emb.shape}")

        # Extract text embedding (first text_vocab_size rows)
        text_emb = combined_emb[:text_vocab_size]
        candle_state_dict["text_emb.weight"] = text_emb
        if verbose:
            print(f"  text_emb.weight: {text_emb.shape}")

        # Extract audio embeddings (next num_codebooks * codebook_vocab_size rows)
        audio_start = text_vocab_size
        for i in range(num_codebooks):
            start_idx = audio_start + i * codebook_vocab_size
            end_idx = start_idx + codebook_vocab_size
            emb_i = combined_emb[start_idx:end_idx]
            candle_state_dict[f"emb.{i}.weight"] = emb_i
        if verbose:
            print(f"  emb.{{0-{num_codebooks-1}}}.weight: ({codebook_vocab_size}, {hidden_size})")

    # Step 2: Convert lm_head
    if verbose:
        print("\nStep 2: Converting lm_head...")

    lm_head_key = "lm_head.weight"
    if lm_head_key in state_dict:
        lm_head = state_dict[lm_head_key]
        # Candle text_linear has 8000 rows (excludes BOS), transformers has 8001
        # Take first 8000 rows (or vocab_size - 1)
        output_vocab_size = text_vocab_size - 1
        text_linear = lm_head[:output_vocab_size]
        candle_state_dict["text_linear.weight"] = text_linear
        if verbose:
            print(f"  lm_head.weight {lm_head.shape} -> text_linear.weight {text_linear.shape}")

    # Step 3: Convert output norm
    if verbose:
        print("\nStep 3: Converting output norm...")

    out_norm_key = "model.norm.weight"
    if out_norm_key in state_dict:
        norm_weight = state_dict[out_norm_key]
        # Reshape from [hidden] to [1, 1, hidden] for RMSNorm alpha
        out_norm_alpha = norm_weight.view(1, 1, -1)
        candle_state_dict["out_norm.alpha"] = out_norm_alpha
        if verbose:
            print(f"  model.norm.weight {norm_weight.shape} -> out_norm.alpha {out_norm_alpha.shape}")

    # Step 4: Convert transformer layers
    if verbose:
        print(f"\nStep 4: Converting {num_layers} transformer layers...")

    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        out_prefix = f"transformer.layers.{layer_idx}"

        # Convert layer norms (reshape to [1, 1, hidden])
        for norm_name, out_name in [("input_layernorm", "norm1"), ("post_attention_layernorm", "norm2")]:
            key = f"{prefix}.{norm_name}.weight"
            if key in state_dict:
                norm_weight = state_dict[key]
                candle_state_dict[f"{out_prefix}.{out_name}.alpha"] = norm_weight.view(1, 1, -1)

        # Convert MLP (rename fc1/fc2 to gating.linear_in/linear_out)
        for fc_name, out_name in [("fc1", "linear_in"), ("fc2", "linear_out")]:
            key = f"{prefix}.mlp.{fc_name}.weight"
            if key in state_dict:
                candle_state_dict[f"{out_prefix}.gating.{out_name}.weight"] = state_dict[key]

        # Combine Q/K/V projections into in_proj_weight
        q_key = f"{prefix}.self_attn.q_proj.linear.weight"
        k_key = f"{prefix}.self_attn.k_proj.linear.weight"
        v_key = f"{prefix}.self_attn.v_proj.linear.weight"

        if q_key in state_dict and k_key in state_dict and v_key in state_dict:
            q = state_dict[q_key]
            k = state_dict[k_key]
            v = state_dict[v_key]

            # Apply inverse rotary permutation to Q and K
            # The forward conversion applies: w.view(n_heads, dim1//n_heads//2, 2, dim2).transpose(1,2).reshape(dim1, dim2)
            # We need to reverse this
            q = inverse_rotary_permute(q, num_heads, hidden_size)
            k = inverse_rotary_permute(k, num_heads, hidden_size)

            in_proj_weight = torch.cat([q, k, v], dim=0)
            candle_state_dict[f"{out_prefix}.self_attn.in_proj_weight"] = in_proj_weight

        # Convert output projection
        o_key = f"{prefix}.self_attn.o_proj.linear.weight"
        if o_key in state_dict:
            candle_state_dict[f"{out_prefix}.self_attn.out_proj.weight"] = state_dict[o_key]

    if verbose:
        print(f"  Converted {num_layers} layers")

    # Step 5: Copy extra_heads (VAD heads) from original Candle model
    # These are not in the transformers model but may be needed by the Candle server
    if verbose:
        print("\nStep 5: Copying extra_heads from original Candle model...")

    try:
        from huggingface_hub import hf_hub_download

        # Determine original Candle model based on input model
        # Map transformers model IDs to their Candle equivalents
        candle_model_map = {
            "kyutai/stt-1b-en_fr-trfs": "kyutai/stt-1b-en_fr-candle",
            "kyutai/stt-2.6b-en-trfs": "kyutai/stt-2.6b-en-candle",
        }

        # Try to find matching Candle model
        candle_model_id = candle_model_map.get(model_dir)
        if candle_model_id is None and "-trfs" in model_dir:
            # Try automatic conversion
            candle_model_id = model_dir.replace("-trfs", "-candle")

        if candle_model_id:
            try:
                candle_weights_path = hf_hub_download(candle_model_id, "model.safetensors")
                candle_weights = load_file(candle_weights_path)

                # Copy extra_heads
                extra_heads_copied = 0
                for key in candle_weights:
                    if key.startswith("extra_heads"):
                        candle_state_dict[key] = candle_weights[key]
                        extra_heads_copied += 1

                if verbose:
                    print(f"  Copied {extra_heads_copied} extra_heads from {candle_model_id}")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load Candle model {candle_model_id}: {e}")
                    print("  extra_heads will not be included (may cause issues with VAD)")
        else:
            if verbose:
                print("  Warning: Could not determine matching Candle model")
                print("  extra_heads will not be included (may cause issues with VAD)")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not copy extra_heads: {e}")

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Conversion complete!")
        print(f"  Input tensors: {len(state_dict)}")
        print(f"  Output tensors: {len(candle_state_dict)}")
        print(f"  Skipped codec_model: {sum(1 for k in state_dict if k.startswith('codec_model'))}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save in safetensors format
    if verbose:
        print(f"\nSaving to {output_path}...")
    save_file(candle_state_dict, output_path)

    if verbose:
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Transformers Kyutai STT weights to Candle format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to transformers model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for Candle-format safetensors file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    convert_transformers_to_candle(
        args.input,
        args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
