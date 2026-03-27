"""Frame-level SSL feature store metadata and utilities.

The paper-faithful HConv path operates on frame-level hidden states before any
phone pooling. Extracted features are stored on disk as per-utterance ``.npy``
shards with a JSON manifest per split/model pair.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from p011.ssl_features import SSL_MODEL_KEYS, SSLModelKey

SSL_MODELS: dict[SSLModelKey, str] = {
    "w2v_300m": "facebook/wav2vec2-large",
    "hubert": "facebook/hubert-large-ll60k",
    "wavlm": "microsoft/wavlm-large",
}

FRAME_RATE_HZ = 50
FRAME_STORE_VERSION = 1
FRAME_STORE_DIRNAME = "ssl_frame_store_v1"


@dataclass(frozen=True)
class FrameStoreEntry:
    utterance_id: str
    speaker_dir: str
    rel_path: str
    num_frames: int
    num_layers: int
    feat_dim: int


@dataclass(frozen=True)
class FrameStoreManifest:
    version: int
    split: str
    model_key: SSLModelKey
    frame_rate_hz: int
    entries: tuple[FrameStoreEntry, ...]


def frame_store_root(features_dir: Path) -> Path:
    """Return the top-level frame store directory."""
    return features_dir / FRAME_STORE_DIRNAME


def frame_store_split_dir(features_dir: Path, split: str, model_key: SSLModelKey) -> Path:
    """Return the directory holding shards for one split/model pair."""
    return frame_store_root(features_dir) / split / model_key


def manifest_path(features_dir: Path, split: str, model_key: SSLModelKey) -> Path:
    """Return the manifest path for one split/model pair."""
    return frame_store_split_dir(features_dir, split, model_key) / "manifest.json"


def write_manifest(features_dir: Path, manifest: FrameStoreManifest) -> Path:
    """Write a manifest JSON file and return its path."""
    path = manifest_path(features_dir, manifest.split, manifest.model_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": manifest.version,
        "split": manifest.split,
        "model_key": manifest.model_key,
        "frame_rate_hz": manifest.frame_rate_hz,
        "entries": [asdict(entry) for entry in manifest.entries],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(features_dir: Path, split: str, model_key: SSLModelKey) -> FrameStoreManifest:
    """Load a manifest from disk."""
    raw = json.loads(manifest_path(features_dir, split, model_key).read_text(encoding="utf-8"))
    entries = tuple(FrameStoreEntry(**entry) for entry in raw["entries"])
    return FrameStoreManifest(
        version=int(raw["version"]),
        split=str(raw["split"]),
        model_key=model_key,
        frame_rate_hz=int(raw["frame_rate_hz"]),
        entries=entries,
    )


def shard_path(features_dir: Path, split: str, model_key: SSLModelKey, entry: FrameStoreEntry) -> Path:
    """Resolve one shard path from a manifest entry."""
    return frame_store_split_dir(features_dir, split, model_key) / entry.rel_path


__all__ = [
    "FRAME_RATE_HZ",
    "FRAME_STORE_DIRNAME",
    "FRAME_STORE_VERSION",
    "SSL_MODELS",
    "FrameStoreEntry",
    "FrameStoreManifest",
    "frame_store_root",
    "frame_store_split_dir",
    "load_manifest",
    "manifest_path",
    "shard_path",
    "write_manifest",
    "SSL_MODEL_KEYS",
]
