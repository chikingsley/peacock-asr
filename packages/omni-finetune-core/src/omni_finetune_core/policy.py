"""Typed fine-tune policy contracts shared by project configs and reviews."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

HeadKind = Literal["ctc", "tdt", "rnnt", "llm"]
MetricName = Literal["wer", "cer", "val_loss", "loss"]
RegimeKind = Literal["preset", "cold_base", "warm_restart", "ad_hoc"]


class _PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MetricPolicy(_PolicyModel):
    """Best-checkpoint metric contract."""

    primary: MetricName = "wer"
    mode: Literal["min"] = "min"


class HeadPolicy(_PolicyModel):
    """Model-head policy: which validation metric is authoritative for checkpoints."""

    kind: HeadKind = "ctc"
    monitor: MetricPolicy = Field(default_factory=MetricPolicy)

    @model_validator(mode="after")
    def _metric_matches_head(self) -> HeadPolicy:
        if self.kind == "ctc" and self.monitor.primary != "wer":
            msg = "CTC heads must monitor validation WER"
            raise ValueError(msg)
        if self.kind in {"tdt", "rnnt"} and self.monitor.primary != "val_loss":
            msg = "TDT/RNNT heads must monitor validation loss"
            raise ValueError(msg)
        if self.kind == "llm" and self.monitor.primary not in {"wer", "loss"}:
            msg = "LLM heads must monitor WER or loss"
            raise ValueError(msg)
        return self


class RecipePolicy(_PolicyModel):
    """Training recipe identity and minimum reproducibility fields."""

    regime: RegimeKind = "preset"
    model_card: str | None = None
    dataset_card: str | None = None
    tokenizer_card: str | None = None
    num_steps: int | None = Field(default=None, gt=0)
    learning_rate: float | None = Field(default=None, gt=0)


class EvalPolicy(_PolicyModel):
    """Evaluation contract used for model comparisons and release gates."""

    split: Literal["dev", "test"] = "test"
    max_duration_seconds: float = Field(default=40.0, gt=0, le=40.0)
    fp32_wer_gate: bool = True


class FinetunePolicy(_PolicyModel):
    """One reviewable contract for recipe, head, metric, and eval decisions."""

    recipe: RecipePolicy = Field(default_factory=RecipePolicy)
    head: HeadPolicy = Field(default_factory=HeadPolicy)
    eval: EvalPolicy = Field(default_factory=EvalPolicy)
