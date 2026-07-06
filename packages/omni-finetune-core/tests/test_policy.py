"""Fine-tune policy contracts: recipe/head/metric/eval validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omni_finetune_core.policy import (
    EvalPolicy,
    FinetunePolicy,
    HeadPolicy,
    MetricPolicy,
    RecipePolicy,
)


def test_default_policy_is_ctc_wer_test_split():
    policy = FinetunePolicy()

    assert policy.recipe.regime == "preset"
    assert policy.head.kind == "ctc"
    assert policy.head.monitor.primary == "wer"
    assert policy.eval.split == "test"
    assert policy.eval.max_duration_seconds == 40.0


def test_transducer_heads_must_monitor_validation_loss():
    policy = HeadPolicy(kind="tdt", monitor=MetricPolicy(primary="val_loss"))
    assert policy.monitor.primary == "val_loss"

    with pytest.raises(ValidationError, match="TDT/RNNT"):
        HeadPolicy(kind="rnnt", monitor=MetricPolicy(primary="wer"))


def test_ctc_heads_must_monitor_wer():
    with pytest.raises(ValidationError, match="CTC"):
        HeadPolicy(kind="ctc", monitor=MetricPolicy(primary="val_loss"))


def test_eval_policy_enforces_omni_audio_cap():
    with pytest.raises(ValidationError, match="less than or equal to 40"):
        EvalPolicy(max_duration_seconds=41.0)


def test_recipe_policy_rejects_invalid_budget_fields():
    with pytest.raises(ValidationError):
        RecipePolicy(num_steps=0)
    with pytest.raises(ValidationError):
        RecipePolicy(learning_rate=0)


def test_policy_models_reject_extra_fields():
    with pytest.raises(ValidationError, match="Extra inputs"):
        RecipePolicy(regime="preset", surprise=True)  # type: ignore[call-arg]
