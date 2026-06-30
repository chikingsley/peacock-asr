from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from moss_mlx_conversion import DEFAULT_MODEL_ID
from moss_mlx_conversion.paths import CACHE_DIR

METADATA_PATTERNS = [
    "README.md",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "modeling_Moss.py",
    "processing_Moss.py",
    "chat_template_default.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
]

WEIGHT_PATTERNS = [
    "*.safetensors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MOSS model metadata or weights.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR / "huggingface")
    parser.add_argument("--include-weights", action="store_true")
    parser.add_argument("--local-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allow_patterns = [*METADATA_PATTERNS]
    if args.include_weights:
        allow_patterns.extend(WEIGHT_PATTERNS)

    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        allow_patterns=allow_patterns,
    )
    print(path)


if __name__ == "__main__":
    main()
