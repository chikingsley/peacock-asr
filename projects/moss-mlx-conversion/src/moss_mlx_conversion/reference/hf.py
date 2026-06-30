from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from moss_mlx_conversion import DEFAULT_MODEL_ID
from moss_mlx_conversion.paths import CACHE_DIR


def hf_cache_dir() -> Path:
    path = CACHE_DIR / "huggingface"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    revision: str = "main",
    local_files_only: bool = False,
) -> Any:
    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        cache_dir=hf_cache_dir(),
        local_files_only=local_files_only,
    )


def load_remote_processor_classes(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    revision: str = "main",
    local_files_only: bool = False,
) -> tuple[type[Any], type[Any]]:
    processor_cls = get_class_from_dynamic_module(
        "processing_Moss.MossProcessor",
        model_id,
        cache_dir=hf_cache_dir(),
        revision=revision,
        local_files_only=local_files_only,
    )
    mel_config_cls = get_class_from_dynamic_module(
        "processing_Moss.MelConfig",
        model_id,
        cache_dir=hf_cache_dir(),
        revision=revision,
        local_files_only=local_files_only,
    )
    return processor_cls, mel_config_cls


def download_template(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    revision: str = "main",
    local_files_only: bool = False,
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=model_id,
            filename="chat_template_default.py",
            revision=revision,
            cache_dir=hf_cache_dir(),
            local_files_only=local_files_only,
        )
    )
