"""Persistent prepared-posterior bundles for scoring benchmarks and runtime.

Each split is stored as a single posterior matrix plus an index of row offsets.
That keeps the number of open files bounded while still allowing memmap-backed
views for CPU workers.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

PreparedItem = tuple[np.ndarray, list[int], list[str], list[float]]
PreparedBundle = dict[str, list[PreparedItem]]
FORMAT_VERSION = 2


def save_prepared_bundle(
    bundle_dir: Path,
    bundle: PreparedBundle,
    meta: dict[str, object],
) -> None:
    """Persist a prepared bundle as JSON metadata plus split-level `.npy` files."""
    tmp_dir = bundle_dir.parent / f".{bundle_dir.name}.tmp-{uuid4().hex}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    meta_payload = {
        **meta,
        "format_version": FORMAT_VERSION,
    }
    (tmp_dir / "meta.json").write_text(
        json.dumps(meta_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for split_name, prepared in bundle.items():
        split_dir = tmp_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        if prepared:
            posterior_dtype = prepared[0][0].dtype
            vocab_dim = prepared[0][0].shape[1]
            concatenated = np.concatenate(
                [np.asarray(posteriors) for posteriors, _, _, _ in prepared],
                axis=0,
            )
        else:
            posterior_dtype = np.float32
            vocab_dim = 0
            concatenated = np.empty((0, 0), dtype=posterior_dtype)

        np.save(split_dir / "posteriors.npy", concatenated, allow_pickle=False)

        index_rows: list[dict[str, object]] = []
        start_row = 0
        for posteriors, phone_indices, valid_phones, valid_scores in prepared:
            end_row = start_row + int(posteriors.shape[0])
            index_rows.append(
                {
                    "start_row": start_row,
                    "end_row": end_row,
                    "phone_indices": phone_indices,
                    "valid_phones": valid_phones,
                    "valid_scores": valid_scores,
                }
            )
            start_row = end_row

        (split_dir / "index.json").write_text(
            json.dumps(
                {
                    "posterior_dtype": str(np.dtype(posterior_dtype).name),
                    "vocab_dim": vocab_dim,
                    "rows": index_rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    tmp_dir.rename(bundle_dir)


def load_prepared_bundle(
    bundle_dir: Path,
    *,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "c",
) -> tuple[dict[str, object], PreparedBundle] | None:
    """Load a prepared bundle. Returns None when the bundle is missing."""
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        return None

    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("format_version", 0)) != FORMAT_VERSION:
        return None

    bundle: PreparedBundle = {}
    for split_dir in sorted(path for path in bundle_dir.iterdir() if path.is_dir()):
        prepared = _load_split_prepared(split_dir, mmap_mode=mmap_mode)
        if prepared is None:
            return None
        bundle[split_dir.name] = prepared

    return meta, bundle


def _load_split_prepared(
    split_dir: Path,
    *,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None,
) -> list[PreparedItem] | None:
    index_path = split_dir / "index.json"
    posterior_path = split_dir / "posteriors.npy"
    if not index_path.exists() or not posterior_path.exists():
        return None

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    index_rows = index_payload.get("rows")
    if not isinstance(index_rows, list):
        return None

    posteriors_all = np.load(
        posterior_path,
        allow_pickle=False,
        mmap_mode=mmap_mode,
    )
    prepared: list[PreparedItem] = []
    for row in index_rows:
        prepared_item = _load_prepared_row(row, posteriors_all)
        if prepared_item is None:
            return None
        prepared.append(prepared_item)
    return prepared


def _load_prepared_row(
    row: object,
    posteriors_all: np.ndarray,
) -> PreparedItem | None:
    row_dict = _normalize_prepared_row(row)
    if row_dict is None:
        return None

    start_row = row_dict.get("start_row")
    end_row = row_dict.get("end_row")
    if not isinstance(start_row, int) or not isinstance(end_row, int):
        return None

    phone_indices = _coerce_int_list(row_dict.get("phone_indices"))
    valid_phones = _coerce_str_list(row_dict.get("valid_phones"))
    valid_scores = _coerce_float_list(row_dict.get("valid_scores"))
    if phone_indices is None or valid_phones is None or valid_scores is None:
        return None

    return (
        posteriors_all[start_row:end_row],
        phone_indices,
        valid_phones,
        valid_scores,
    )


def _normalize_prepared_row(row: object) -> dict[str, object] | None:
    if not isinstance(row, dict) or "posterior_file" in row:
        return None

    row_dict: dict[str, object] = {}
    for key, value in row.items():
        if not isinstance(key, str):
            return None
        row_dict[key] = value
    return row_dict


def _coerce_int_list(value: object) -> list[int] | None:
    if not isinstance(value, list):
        return None
    coerced: list[int] = []
    for item in value:
        if not isinstance(item, int):
            return None
        coerced.append(item)
    return coerced


def _coerce_str_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    coerced: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        coerced.append(item)
    return coerced


def _coerce_float_list(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None
    coerced: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            return None
        coerced.append(float(item))
    return coerced
