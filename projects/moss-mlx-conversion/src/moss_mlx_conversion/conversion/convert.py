from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from moss_mlx_conversion import DEFAULT_MODEL_ID
from moss_mlx_conversion.config import MossModelConfig
from moss_mlx_conversion.conversion.weights import map_source_key
from moss_mlx_conversion.dump import ensure_dir, write_json
from moss_mlx_conversion.paths import CACHE_DIR
from moss_mlx_conversion.reference.download import METADATA_PATTERNS, WEIGHT_PATTERNS

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template_default.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MOSS safetensors to MLX key layout.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/mlx/MOSS-Transcribe-preview-2B-bf16"),
    )
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--include-tied-lm-head", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_snapshot(
    model_id: str,
    *,
    revision: str,
    snapshot_dir: Path | None,
    local_files_only: bool,
) -> Path:
    if snapshot_dir is not None:
        return snapshot_dir
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=CACHE_DIR / "huggingface",
            allow_patterns=[*METADATA_PATTERNS, *WEIGHT_PATTERNS],
            local_files_only=local_files_only,
        )
    )


def convert_tensor(source_key: str, tensor: torch.Tensor, *, dtype: str) -> torch.Tensor:
    if source_key.startswith("model.audio_model.conv2d") and source_key.endswith(".weight"):
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
    if dtype == "fp32":
        return tensor.float().contiguous()
    return tensor.contiguous()


def copy_metadata(snapshot_dir: Path, output_dir: Path, config: MossModelConfig) -> None:
    for filename in TOKENIZER_FILES:
        source = snapshot_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)

    shutil.copy2(snapshot_dir / "config.json", output_dir / "original_config.json")
    (output_dir / "config.json").write_text(
        json.dumps(config.to_mlx_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def convert_weights(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    dtype: str,
    include_tied_lm_head: bool,
) -> dict[str, Any]:
    safetensor_paths = sorted(snapshot_dir.glob("*.safetensors"))
    if not safetensor_paths:
        raise FileNotFoundError(f"No safetensors found in {snapshot_dir}")

    converted: dict[str, torch.Tensor] = {}
    mapping: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    for path in safetensor_paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            source_keys = handle.keys()
            for source_key in source_keys:
                destination_key = map_source_key(source_key)
                if destination_key is None:
                    skipped[source_key] = "unrecognized prefix"
                    continue
                tensor = handle.get_tensor(source_key)
                converted_tensor = convert_tensor(source_key, tensor, dtype=dtype)
                converted[destination_key] = converted_tensor
                mapping[source_key] = {
                    "destination": destination_key,
                    "source_shape": list(tensor.shape),
                    "destination_shape": list(converted_tensor.shape),
                    "dtype": str(converted_tensor.dtype),
                    "file": path.name,
                }

    generated: dict[str, str] = {}
    if include_tied_lm_head:
        embed = converted.get("model.embed_tokens.weight")
        if embed is None:
            raise KeyError("Cannot generate lm_head.weight without model.embed_tokens.weight")
        converted["lm_head.weight"] = embed
        generated["lm_head.weight"] = "model.embed_tokens.weight"

    weight_path = output_dir / "weights.safetensors"
    save_file(converted, weight_path)
    return {
        "weight_path": str(weight_path),
        "source_tensor_count": len(mapping) + len(skipped),
        "saved_tensor_count": len(converted),
        "skipped_source_tensor_count": len(skipped),
        "generated_tensors": generated,
        "skipped_source_tensors": skipped,
        "mapping": mapping,
    }


def main() -> None:
    args = parse_args()
    snapshot_dir = resolve_snapshot(
        args.model_id,
        revision=args.revision,
        snapshot_dir=args.snapshot_dir,
        local_files_only=args.local_files_only,
    )
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    ensure_dir(output_dir)

    config = MossModelConfig.from_json(snapshot_dir / "config.json")
    copy_metadata(snapshot_dir, output_dir, config)
    report = convert_weights(
        snapshot_dir,
        output_dir,
        dtype=args.dtype,
        include_tied_lm_head=args.include_tied_lm_head,
    )
    report.update(
        {
            "model_id": args.model_id,
            "revision": args.revision,
            "snapshot_dir": str(snapshot_dir),
            "output_dir": str(output_dir),
            "dtype": args.dtype,
            "include_tied_lm_head": args.include_tied_lm_head,
        }
    )
    write_json(output_dir / "conversion-report.json", report)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "weight_path": report["weight_path"],
                "source_tensor_count": report["source_tensor_count"],
                "saved_tensor_count": report["saved_tensor_count"],
                "skipped_source_tensor_count": report["skipped_source_tensor_count"],
                "generated_tensors": report["generated_tensors"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
