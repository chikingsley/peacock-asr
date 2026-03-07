from __future__ import annotations

import pytest

from p004_training_from_scratch.vast.models import OfferSearchSpec
from p004_training_from_scratch.vast.query import build_env_flags, build_offer_query


def test_build_offer_query_quotes_gpu_name_and_merges_clauses() -> None:
    query = build_offer_query(
        OfferSearchSpec(
            gpu_name="RTX 4090",
            num_gpus=1,
            query_clauses=("cpu_cores>=16", "cpu_ram>=64000"),
        )
    )

    assert query == ('gpu_name="RTX 4090" num_gpus=1 cpu_cores>=16 cpu_ram>=64000')


def test_build_env_flags_joins_multiple_pairs() -> None:
    env = build_env_flags(["WANDB_MODE=offline", "HF_HOME=/workspace/.hf"])
    assert env == "-e WANDB_MODE=offline -e HF_HOME=/workspace/.hf"


def test_build_env_flags_rejects_invalid_pair() -> None:
    with pytest.raises(ValueError, match="Invalid env pair"):
        build_env_flags(["NOT_VALID"])
