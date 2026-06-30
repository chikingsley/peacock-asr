from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from moss_mlx_conversion.config import MossModelConfig
from moss_mlx_conversion.conversion.convert import TOKENIZER_FILES
from moss_mlx_conversion.dump import ensure_dir, write_json
from moss_mlx_conversion.mlx_compat import mx, require_mlx
from moss_mlx_conversion.model.moss import MossMLXModel
from moss_mlx_conversion.paths import MLX_DIR
from moss_mlx_conversion.runtime.quantization import (
    DEFAULT_QUANTIZATION_MODE,
    make_scope_predicate,
)

QUANTIZATION_SCOPES = ["text-decoder", "audio-adapter", "audio-encoder", "text-and-adapter", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a converted MOSS MLX artifact.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=MLX_DIR / "MOSS-Transcribe-preview-2B-bf16",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bits", type=int, choices=[4, 8], required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
        default=DEFAULT_QUANTIZATION_MODE,
    )
    parser.add_argument("--scope", choices=QUANTIZATION_SCOPES, default="text-decoder")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_output_dir(source_dir: Path, *, bits: int, group_size: int, scope: str) -> Path:
    return source_dir.parent / f"{source_dir.name}-{scope}-{bits}bit-g{group_size}"


def copy_sidecar_files(source_dir: Path, output_dir: Path) -> None:
    for filename in TOKENIZER_FILES:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)
    for filename in ["original_config.json", "conversion-report.json", "mapping-report.json"]:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)


def quantize_artifact(
    *,
    source_dir: Path,
    output_dir: Path,
    bits: int,
    group_size: int,
    mode: str,
    scope: str,
    overwrite: bool,
) -> dict[str, Any]:
    require_mlx()
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    started = time.perf_counter()
    config_path = source_dir / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config = MossModelConfig.from_moss_dict(config_data)
    model = MossMLXModel(config)
    weights = mx.load(str(source_dir / "weights.safetensors"))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())

    quantization = {
        "bits": bits,
        "group_size": group_size,
        "mode": mode,
        "scope": scope,
    }
    quantize_started = time.perf_counter()
    quantize_model = __import__("mlx_lm.utils", fromlist=["quantize_model"]).quantize_model
    model, quantized_config = quantize_model(
        model,
        {**config_data, "quantization": quantization},
        group_size,
        bits,
        mode=mode,
        quant_predicate=make_scope_predicate(scope=scope, group_size=group_size),
    )
    quantize_elapsed_sec = time.perf_counter() - quantize_started
    quantized_config["quantization"] = {
        **quantized_config["quantization"],
        "scope": scope,
    }
    quantized_config["quantization_config"] = quantized_config["quantization"]

    save_started = time.perf_counter()
    tree_flatten = __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten
    quantized_weights = dict(tree_flatten(model.parameters()))
    weight_path = output_dir / "weights.safetensors"
    mx.save_safetensors(str(weight_path), quantized_weights, metadata={"format": "mlx"})
    save_elapsed_sec = time.perf_counter() - save_started

    copy_sidecar_files(source_dir, output_dir)
    (output_dir / "config.json").write_text(
        json.dumps(quantized_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "weight_path": str(weight_path),
        "bits": bits,
        "group_size": group_size,
        "mode": mode,
        "scope": scope,
        "saved_tensor_count": len(quantized_weights),
        "weight_bytes": weight_path.stat().st_size,
        "quantize_elapsed_sec": quantize_elapsed_sec,
        "save_elapsed_sec": save_elapsed_sec,
        "elapsed_sec": time.perf_counter() - started,
    }
    write_json(output_dir / "quantization-report.json", report)
    return report


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = default_output_dir(
            args.source_dir,
            bits=args.bits,
            group_size=args.group_size,
            scope=args.scope,
        )
    report = quantize_artifact(
        source_dir=args.source_dir,
        output_dir=output_dir,
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
        scope=args.scope,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
