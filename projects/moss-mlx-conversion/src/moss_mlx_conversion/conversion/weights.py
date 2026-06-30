from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from safetensors import safe_open

from moss_mlx_conversion import DEFAULT_MODEL_ID
from moss_mlx_conversion.dump import ensure_dir, write_json
from moss_mlx_conversion.paths import CACHE_DIR
from moss_mlx_conversion.reference.download import METADATA_PATTERNS, WEIGHT_PATTERNS

DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "BOOL": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MOSS safetensors and write map report.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mlx/mapping-report.json"))
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


def map_source_key(source_key: str) -> str | None:
    if source_key.startswith("model.language_model."):
        return f"model.{source_key.removeprefix('model.language_model.')}"
    if source_key.startswith("model.audio_model."):
        return source_key.removeprefix("model.")
    if source_key.startswith("model.audio_adapter."):
        return source_key.removeprefix("model.")
    return None


def category_for_key(source_key: str) -> str:
    if source_key.startswith("model.language_model."):
        return "language_model"
    if source_key.startswith("model.audio_model."):
        return "audio_model"
    if source_key.startswith("model.audio_adapter."):
        return "audio_adapter"
    return "unknown"


def tensor_bytes(shape: list[int], dtype: str) -> int:
    return math.prod(shape) * DTYPE_BYTES.get(dtype, 0)


def inspect_safetensors(snapshot_dir: Path) -> dict[str, Any]:
    safetensor_paths = sorted(snapshot_dir.glob("*.safetensors"))
    if not safetensor_paths:
        raise FileNotFoundError(f"No safetensors found in {snapshot_dir}")

    tensors: dict[str, dict[str, Any]] = {}
    for path in safetensor_paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = handle.keys()
            for key in keys:
                tensor_slice = handle.get_slice(key)
                shape = list(tensor_slice.get_shape())
                dtype = tensor_slice.get_dtype()
                tensors[key] = {
                    "file": path.name,
                    "shape": shape,
                    "dtype": dtype,
                    "parameter_count": math.prod(shape),
                    "bytes": tensor_bytes(shape, dtype),
                }
    return tensors


def build_mapping_report(
    *,
    model_id: str,
    revision: str,
    snapshot_dir: Path,
) -> dict[str, Any]:
    tensors = inspect_safetensors(snapshot_dir)
    mapped: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    totals_by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensors": 0, "parameters": 0, "bytes": 0}
    )

    for source_key, info in tensors.items():
        category = category_for_key(source_key)
        destination_key = map_source_key(source_key)
        categories[category][info["dtype"]] += 1
        totals_by_category[category]["tensors"] += 1
        totals_by_category[category]["parameters"] += info["parameter_count"]
        totals_by_category[category]["bytes"] += info["bytes"]
        if destination_key is None:
            skipped[source_key] = "unrecognized prefix"
            continue
        mapped[source_key] = {
            "destination": destination_key,
            **info,
        }

    generated: dict[str, dict[str, str]] = {}
    embed_source = "model.language_model.embed_tokens.weight"
    destination_keys = {entry["destination"] for entry in mapped.values()}
    if embed_source in mapped and "lm_head.weight" not in destination_keys:
        generated["lm_head.weight"] = {
            "source": embed_source,
            "reason": (
                "MOSS ties lm_head to language_model.embed_tokens.weight and does not "
                "store a separate lm_head tensor."
            ),
        }

    prefix_counts = Counter(category_for_key(key) for key in tensors)
    dtype_counts = Counter(info["dtype"] for info in tensors.values())
    total_parameters = sum(info["parameter_count"] for info in tensors.values())
    total_bytes = sum(info["bytes"] for info in tensors.values())

    return {
        "model_id": model_id,
        "revision": revision,
        "snapshot_dir": str(snapshot_dir),
        "safetensors": [path.name for path in sorted(snapshot_dir.glob("*.safetensors"))],
        "source_tensor_count": len(tensors),
        "mapped_source_tensor_count": len(mapped),
        "skipped_source_tensor_count": len(skipped),
        "generated_destination_tensor_count": len(generated),
        "destination_tensor_count": len(mapped) + len(generated),
        "total_parameters": total_parameters,
        "total_bytes": total_bytes,
        "dtype_counts": dict(dtype_counts),
        "prefix_counts": dict(prefix_counts),
        "totals_by_category": dict(totals_by_category),
        "dtype_counts_by_category": {
            category: dict(counter) for category, counter in categories.items()
        },
        "generated_destination_tensors": generated,
        "skipped_source_tensors": skipped,
        "mapping": mapped,
    }


def main() -> None:
    args = parse_args()
    snapshot_dir = resolve_snapshot(
        args.model_id,
        revision=args.revision,
        snapshot_dir=args.snapshot_dir,
        local_files_only=args.local_files_only,
    )
    report = build_mapping_report(
        model_id=args.model_id,
        revision=args.revision,
        snapshot_dir=snapshot_dir,
    )
    output = args.output
    if not output.is_absolute():
        output = Path.cwd() / output
    ensure_dir(output.parent)
    write_json(output, report)
    summary = {
        key: report[key]
        for key in [
            "source_tensor_count",
            "mapped_source_tensor_count",
            "skipped_source_tensor_count",
            "generated_destination_tensor_count",
            "destination_tensor_count",
            "total_parameters",
            "total_bytes",
            "dtype_counts",
            "prefix_counts",
        ]
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"mapping report: {output}")


if __name__ == "__main__":
    main()
