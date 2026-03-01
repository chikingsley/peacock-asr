#!/usr/bin/env python3
"""
Validate Transformers → Candle weight conversion.

This script validates that the conversion produces correct weights by:
1. Comparing weight keys and shapes against original Candle model
2. Comparing actual weight values (for round-trip validation)
3. Optionally running WER tests using the moshi library

Usage:
    # Basic key/shape comparison (run from fine-tuning directory)
    uv run python tests/test_conversion.py \
        --original kyutai/stt-1b-en_fr-candle \
        --converted ./converted_model.safetensors

    # Full validation with WER test
    uv run python tests/test_conversion.py \
        --original kyutai/stt-1b-en_fr-candle \
        --converted ./converted_model.safetensors \
        --audio ../delayed-streams-modeling/audio/bria.mp3
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open


def download_model_weights(model_id: str, filename: str = "model.safetensors") -> str:
    """Download model weights from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model_id, filename)


def get_weight_info(path: str) -> Dict[str, Tuple[torch.Size, torch.dtype]]:
    """Get weight keys, shapes, and dtypes from a safetensors file."""
    result = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            result[key] = (tensor.shape, tensor.dtype)
    return result


def compare_keys_and_shapes(
    original_path: str,
    converted_path: str,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Compare weight keys and shapes between original and converted models.

    Returns a dict with:
        - keys_match: bool
        - shapes_match: bool
        - missing_keys: set of keys in original but not converted
        - extra_keys: set of keys in converted but not original
        - shape_mismatches: list of (key, original_shape, converted_shape)
    """
    results = {
        "keys_match": True,
        "shapes_match": True,
        "missing_keys": set(),
        "extra_keys": set(),
        "shape_mismatches": [],
    }

    original_info = get_weight_info(original_path)
    converted_info = get_weight_info(converted_path)

    original_keys = set(original_info.keys())
    converted_keys = set(converted_info.keys())

    # Check for missing/extra keys
    results["missing_keys"] = original_keys - converted_keys
    results["extra_keys"] = converted_keys - original_keys

    if results["missing_keys"] or results["extra_keys"]:
        results["keys_match"] = False

    # Check shapes for common keys
    common_keys = original_keys & converted_keys
    for key in sorted(common_keys):
        orig_shape, orig_dtype = original_info[key]
        conv_shape, conv_dtype = converted_info[key]

        if orig_shape != conv_shape:
            results["shapes_match"] = False
            results["shape_mismatches"].append((key, orig_shape, conv_shape))

    if verbose:
        print(f"\n{'='*60}")
        print("KEY AND SHAPE COMPARISON")
        print(f"{'='*60}")
        print(f"Original keys:  {len(original_keys)}")
        print(f"Converted keys: {len(converted_keys)}")
        print(f"Common keys:    {len(common_keys)}")

        if results["missing_keys"]:
            print(f"\n❌ Missing keys ({len(results['missing_keys'])}):")
            for key in sorted(results["missing_keys"])[:10]:
                print(f"   - {key}")
            if len(results["missing_keys"]) > 10:
                print(f"   ... and {len(results['missing_keys']) - 10} more")

        if results["extra_keys"]:
            print(f"\n❌ Extra keys ({len(results['extra_keys'])}):")
            for key in sorted(results["extra_keys"])[:10]:
                print(f"   - {key}")
            if len(results["extra_keys"]) > 10:
                print(f"   ... and {len(results['extra_keys']) - 10} more")

        if results["shape_mismatches"]:
            print(f"\n❌ Shape mismatches ({len(results['shape_mismatches'])}):")
            for key, orig_shape, conv_shape in results["shape_mismatches"][:10]:
                print(f"   {key}: {orig_shape} vs {conv_shape}")
            if len(results["shape_mismatches"]) > 10:
                print(f"   ... and {len(results['shape_mismatches']) - 10} more")

        if results["keys_match"] and results["shapes_match"]:
            print("\n✓ All keys and shapes match!")

    return results


def compare_weight_values(
    original_path: str,
    converted_path: str,
    atol: float = 1e-6,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Compare actual weight values between original and converted models.

    This is useful for round-trip validation (Candle -> Transformers -> Candle).

    Returns a dict with:
        - values_match: bool
        - max_diff: maximum absolute difference across all tensors
        - mismatches: list of (key, max_diff)
    """
    results = {
        "values_match": True,
        "max_diff": 0.0,
        "mismatches": [],
    }

    with safe_open(original_path, framework="pt") as orig:
        with safe_open(converted_path, framework="pt") as conv:
            common_keys = set(orig.keys()) & set(conv.keys())

            for key in sorted(common_keys):
                orig_tensor = orig.get_tensor(key)
                conv_tensor = conv.get_tensor(key)

                if orig_tensor.shape != conv_tensor.shape:
                    continue  # Shape mismatch handled elsewhere

                # Convert to same dtype for comparison
                orig_float = orig_tensor.float()
                conv_float = conv_tensor.float()

                diff = (orig_float - conv_float).abs()
                max_diff = diff.max().item()

                results["max_diff"] = max(results["max_diff"], max_diff)

                if not torch.allclose(orig_float, conv_float, atol=atol):
                    results["values_match"] = False
                    results["mismatches"].append((key, max_diff))

    if verbose:
        print(f"\n{'='*60}")
        print("VALUE COMPARISON")
        print(f"{'='*60}")
        print(f"Tolerance: {atol}")
        print(f"Max diff:  {results['max_diff']:.2e}")

        if results["mismatches"]:
            print(f"\n❌ Value mismatches ({len(results['mismatches'])}):")
            for key, diff in sorted(results["mismatches"], key=lambda x: -x[1])[:10]:
                print(f"   {key}: max diff = {diff:.2e}")
            if len(results["mismatches"]) > 10:
                print(f"   ... and {len(results['mismatches']) - 10} more")
        else:
            print("\n✓ All weight values match within tolerance!")

    return results


def run_wer_test(
    model_weights: str,
    audio_file: str,
    mimi_weights: Optional[str] = None,
    tokenizer: Optional[str] = None,
    verbose: bool = True
) -> Optional[str]:
    """
    Run STT inference using moshi library and return transcript.

    This requires the moshi library and runs the stt_from_file_pytorch.py script.
    """
    import subprocess

    # Find the script
    script_dir = Path(__file__).parent.parent / "delayed-streams-modeling" / "scripts"
    script_path = script_dir / "stt_from_file_pytorch.py"

    if not script_path.exists():
        print(f"Warning: Could not find {script_path}")
        return None

    # Build command
    cmd = [
        "uv", "run", "python", str(script_path),
        "--moshi-weight", model_weights,
    ]

    if mimi_weights:
        cmd.extend(["--mimi-weight", mimi_weights])
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])

    cmd.append(audio_file)

    if verbose:
        print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"Error running WER test: {result.stderr}")
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("WER test timed out")
        return None
    except Exception as e:
        print(f"Error running WER test: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate Transformers → Candle weight conversion"
    )
    parser.add_argument(
        "--original",
        required=True,
        help="Path to original Candle weights (file or HuggingFace model ID)"
    )
    parser.add_argument(
        "--converted",
        required=True,
        help="Path to converted weights to validate"
    )
    parser.add_argument(
        "--audio",
        help="Optional audio file for WER test"
    )
    parser.add_argument(
        "--mimi-weight",
        help="Path to mimi codec weights (for WER test)"
    )
    parser.add_argument(
        "--tokenizer",
        help="Path to tokenizer (for WER test)"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for value comparison"
    )
    parser.add_argument(
        "--skip-values",
        action="store_true",
        help="Skip value comparison (only check keys/shapes)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Resolve original weights path
    original_path = args.original
    if not os.path.exists(original_path):
        # Try to download from HuggingFace
        if verbose:
            print(f"Downloading original weights from {args.original}...")
        try:
            original_path = download_model_weights(args.original)
        except Exception as e:
            print(f"Error downloading original weights: {e}")
            sys.exit(1)

    converted_path = args.converted
    if not os.path.exists(converted_path):
        print(f"Error: Converted weights not found at {converted_path}")
        sys.exit(1)

    # Run comparisons
    all_passed = True

    # 1. Key and shape comparison
    key_results = compare_keys_and_shapes(original_path, converted_path, verbose)
    if not (key_results["keys_match"] and key_results["shapes_match"]):
        all_passed = False

    # 2. Value comparison (optional)
    if not args.skip_values and key_results["keys_match"] and key_results["shapes_match"]:
        value_results = compare_weight_values(
            original_path, converted_path, args.atol, verbose
        )
        if not value_results["values_match"]:
            all_passed = False

    # 3. WER test (optional)
    if args.audio:
        print(f"\n{'='*60}")
        print("WER TEST")
        print(f"{'='*60}")

        # Run with converted weights
        transcript = run_wer_test(
            converted_path,
            args.audio,
            args.mimi_weight,
            args.tokenizer,
            verbose
        )

        if transcript:
            print(f"\nTranscript: {transcript}")
        else:
            print("\n⚠ WER test could not be completed")

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if all_passed:
        print("✓ Conversion validation PASSED")
        sys.exit(0)
    else:
        print("✗ Conversion validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
