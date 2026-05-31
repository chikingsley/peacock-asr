"""Typed fairseq2 asset cards for omni CTC fine-tunes, registered in-process.

fairseq2 resolves model/tokenizer/dataset by *name* through its asset store, populated
from every package that exposes a ``fairseq2.extension`` entry point. These Pydantic
models give the three card kinds we use a validated shape (paths checked, keys typed),
and :func:`register_cards` wraps fairseq2's in-memory registration. A project keeps its
``setup_fairseq2_extension`` thin:

    from omni_finetune_core.assets import ModelCard, register_cards
    CARDS = [TokenizerCard(...), ModelCard(...), MixtureParquetDatasetCard(...)]
    def setup_fairseq2_extension(container): register_cards(container, "my_proj", CARDS)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fairseq2.composition.assets import register_in_memory_assets
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fairseq2.runtime.dependency import DependencyContainer


class _Card(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str

    def to_card_dict(self) -> dict[str, object]:
        raise NotImplementedError


class TokenizerCard(_Card):
    tokenizer: Path
    tokenizer_family: str = "char_tokenizer"

    def to_card_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tokenizer_family": self.tokenizer_family,
            "tokenizer": str(self.tokenizer),
        }


class ModelCard(_Card):
    checkpoint: Path
    tokenizer_ref: str
    model_family: str = "wav2vec2_asr"
    model_arch: str = "300m_v2"

    def to_card_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_family": self.model_family,
            "model_arch": self.model_arch,
            "checkpoint": str(self.checkpoint),
            "tokenizer_ref": self.tokenizer_ref,
        }


class MixtureParquetDatasetCard(_Card):
    data: Path
    tokenizer_ref: str
    dataset_family: str = "mixture_parquet_asr_dataset"

    def to_card_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dataset_family": self.dataset_family,
            "dataset_config": {"data": str(self.data)},
            "tokenizer_ref": self.tokenizer_ref,
        }


def register_cards(
    container: DependencyContainer, source_name: str, cards: Sequence[_Card]
) -> None:
    """Register typed cards into fairseq2's in-memory asset store under ``source_name``."""
    register_in_memory_assets(container, source_name, [c.to_card_dict() for c in cards])
