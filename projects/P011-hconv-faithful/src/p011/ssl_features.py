"""Canonical SSL feature metadata and helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

type SSLModelKey = Literal["w2v_300m", "hubert", "wavlm"]

SSL_MODEL_KEYS: tuple[SSLModelKey, ...] = ("w2v_300m", "hubert", "wavlm")
SSL_FEATURE_DIM = 1024
SSL_LAST_LAYER_FILES: dict[SSLModelKey, str] = {
    "hubert": "hubert_feat_v2.npy",
    "w2v_300m": "w2v_300m_feat_v2.npy",
    "wavlm": "wavlm_feat_v2.npy",
}


def parse_ssl_model_keys(value: str | Sequence[str] | None) -> tuple[SSLModelKey, ...]:
    """Normalize a user-provided SSL model selection into canonical repo order."""
    if value is None:
        return SSL_MODEL_KEYS

    if isinstance(value, str):
        requested = [part.strip().lower() for part in value.split(",") if part.strip()]
    else:
        requested = [str(part).strip().lower() for part in value if str(part).strip()]

    if not requested:
        raise ValueError("ssl_models must include at least one SSL model")

    unknown = sorted({part for part in requested if part not in SSL_MODEL_KEYS})
    if unknown:
        valid = ", ".join(SSL_MODEL_KEYS)
        raise ValueError(f"Unknown SSL model(s): {', '.join(unknown)}. Valid choices: {valid}")

    requested_set = set(requested)
    normalized = tuple(cast(SSLModelKey, key) for key in SSL_MODEL_KEYS if key in requested_set)
    if not normalized:
        raise ValueError("ssl_models must include at least one valid SSL model")
    return normalized


def ssl_feature_dim(model_keys: Sequence[SSLModelKey]) -> int:
    """Return the concatenated SSL feature width for the selected model keys."""
    if not model_keys:
        raise ValueError("Expected at least one SSL model key")
    return SSL_FEATURE_DIM * len(model_keys)
