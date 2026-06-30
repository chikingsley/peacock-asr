from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def tensor_to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    tensor = value.detach().cpu()
    if tensor.dtype in {torch.bfloat16, torch.float16}:
        tensor = tensor.float()
    return tensor.numpy()


def array_digest(value: torch.Tensor | np.ndarray) -> str:
    array = np.ascontiguousarray(tensor_to_numpy(value))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def tensor_stats(value: torch.Tensor | np.ndarray, *, sample_size: int = 12) -> dict[str, Any]:
    array = tensor_to_numpy(value)
    flat = array.reshape(-1)
    stats: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(value.dtype if isinstance(value, torch.Tensor) else array.dtype),
        "sha256_16": array_digest(array),
        "sample": flat[:sample_size].tolist(),
    }
    if flat.size == 0:
        return stats

    if np.issubdtype(array.dtype, np.bool_):
        stats["true_count"] = int(array.sum())
        stats["false_count"] = int(array.size - array.sum())
        return stats

    numeric = array.astype(np.float64, copy=False)
    stats.update(
        {
            "min": float(np.nanmin(numeric)),
            "max": float(np.nanmax(numeric)),
            "mean": float(np.nanmean(numeric)),
            "std": float(np.nanstd(numeric)),
            "nan_count": int(np.isnan(numeric).sum()),
        }
    )
    return stats


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_npz(path: Path, **arrays: torch.Tensor | np.ndarray) -> None:
    ensure_dir(path.parent)
    converted = {key: tensor_to_numpy(value) for key, value in arrays.items()}
    np.savez_compressed(path, **converted)  # ty: ignore[invalid-argument-type]


def topk_summary(logits: torch.Tensor, *, k: int = 10) -> dict[str, list[float] | list[int]]:
    values, indices = torch.topk(logits.detach().float().cpu(), k=k, dim=-1)
    return {
        "indices": indices.reshape(-1).tolist(),
        "values": values.reshape(-1).tolist(),
    }
