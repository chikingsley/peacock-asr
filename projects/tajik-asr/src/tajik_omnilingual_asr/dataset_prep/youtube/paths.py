from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARTIFACT_DIR = (
    ROOT / "src/tajik_omnilingual_asr/dataset_prep/artifacts/youtube_learning_tajik_v0"
)
DEFAULT_CHANNEL_URL = "https://www.youtube.com/@learningtajik/videos"
DEFAULT_LANGUAGE = "tgk"
DEFAULT_CAPTION_LANGUAGES = "en-orig,en"
