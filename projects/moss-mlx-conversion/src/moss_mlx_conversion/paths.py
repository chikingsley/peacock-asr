from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
REFERENCE_DIR = ARTIFACTS_DIR / "reference"
MLX_DIR = ARTIFACTS_DIR / "mlx"
