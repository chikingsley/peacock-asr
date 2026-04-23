from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
ScoreRange = Annotated[list[float], Field(min_length=2, max_length=2)]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ScenarioName(StrEnum):
    FREE_SPEAKING = "free_speaking"
    READ_ALOUD = "read_aloud"


class PaperTable(StrEnum):
    TABLE_1 = "table_1"
    TABLE_2 = "table_2"


class FreeSpeakingAssignmentRule(StrEnum):
    IGNORE = "ignore"
    ALIGNED_SEGMENT = "aligned_segment"
    ZERO = "zero"


class PaperReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    doi: str
    acl_url: AnyHttpUrl
    arxiv_url: AnyHttpUrl
    official_repo_url: AnyHttpUrl
    official_repo_checked_on: date
    official_repo_public_state: str


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden_units: PositiveInt = 24
    phone_encoder_blocks: PositiveInt = 3
    word_encoder_blocks: PositiveInt = 2
    utterance_encoder_blocks: PositiveInt = 1
    mhsa_heads: PositiveInt = 1
    attention_pool_heads: PositiveInt = 3
    cnn_kernel_size: PositiveInt = 3
    fusion_weight_conv: Probability = 0.3
    word_embedding_backend: str = "modernbert"
    ssl_feature_dim: PositiveInt = 3072
    projected_ssl_dim: PositiveInt = 24
    ssl_models: list[str] = [
        "facebook/wav2vec2-large-960h",
        "facebook/hubert-large-ls960-ft",
        "microsoft/wavlm-large",
    ]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer: str = "adam"
    learning_rate: PositiveFloat = 0.001
    batch_size: PositiveInt = 25
    epochs: PositiveInt = 100
    trials: PositiveInt = 5
    selection_metric: str = "phone_mse"
    word_score_range: ScoreRange = [0.0, 2.0]
    utterance_score_range: ScoreRange = [0.0, 2.0]
    use_cono_regularizer: bool = True
    use_curriculum_learning: bool = True


class FreeSpeakingAssignmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deletion: FreeSpeakingAssignmentRule = FreeSpeakingAssignmentRule.IGNORE
    substitution: FreeSpeakingAssignmentRule = FreeSpeakingAssignmentRule.ALIGNED_SEGMENT
    insertion: FreeSpeakingAssignmentRule = FreeSpeakingAssignmentRule.ZERO


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus: str = "Speechocean762"
    sampling_rate_hz: PositiveInt = 16_000
    asr_model: str = "openai/whisper-large-v3"
    g2p_model: str = "g2p_en"
    free_speaking_assignment: FreeSpeakingAssignmentConfig = Field(
        default_factory=FreeSpeakingAssignmentConfig
    )


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ScenarioName
    paper_table: PaperTable
    requires_reference_text: bool
    notes: str


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper: PaperReference
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    scenario: ScenarioConfig

    @model_validator(mode="after")
    def validate_paper_consistency(self) -> ExperimentConfig:
        if self.scenario.name is ScenarioName.FREE_SPEAKING:
            if not self.training.use_curriculum_learning:
                raise ValueError("free_speaking config must keep curriculum learning enabled.")
            if not self.training.use_cono_regularizer:
                raise ValueError("free_speaking config must keep CONO enabled.")
            if self.scenario.paper_table is not PaperTable.TABLE_1:
                raise ValueError("free_speaking config must map to Table 1.")

        if self.scenario.name is ScenarioName.READ_ALOUD:
            if self.training.use_curriculum_learning:
                raise ValueError("read_aloud config must disable curriculum learning.")
            if self.training.use_cono_regularizer:
                raise ValueError("read_aloud config must disable CONO regularization.")
            if self.scenario.paper_table is not PaperTable.TABLE_2:
                raise ValueError("read_aloud config must map to Table 2.")

        if self.model.hidden_units != self.model.projected_ssl_dim:
            raise ValueError("projected_ssl_dim should match hidden_units for the paper-faithful scaffold.")

        return self


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(dict(existing), value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Config file must contain a mapping at the top level: {path}")
    return loaded


def load_experiment_config(base_path: Path, override_path: Path | None = None) -> ExperimentConfig:
    base_config = _read_yaml(base_path)
    merged = base_config if override_path is None else _deep_merge(base_config, _read_yaml(override_path))
    return ExperimentConfig.model_validate(merged)


def load_named_config(scenario: ScenarioName | str, config_dir: Path = CONFIG_DIR) -> ExperimentConfig:
    scenario_name = ScenarioName(scenario)
    override_name = {
        ScenarioName.FREE_SPEAKING: "hippo_free_speaking.yaml",
        ScenarioName.READ_ALOUD: "hippo_read_aloud.yaml",
    }[scenario_name]
    return load_experiment_config(config_dir / "base_config.yaml", config_dir / override_name)


class TrainingRunSettings(BaseSettings):
    """Runtime configuration for ``p014 train`` (env + CLI overrides).

    Distinct from :class:`ExperimentConfig`, which is the YAML-pinned paper
    spec. This one tracks knobs that change per run: output directories,
    seeds, CONO toggles, trackio targets, etc. Every field is overridable via
    the ``P014_`` env prefix or through :func:`p014.cli.main`.
    """

    model_config = SettingsConfigDict(env_prefix="P014_", env_file=".env", extra="ignore")

    output_dir: Path = Path("artifacts")
    cache_dir: Path = Path(".cache/p014")
    run_name: str = "hippo_read_aloud"
    device: str = "auto"
    epochs: PositiveInt = 100
    batch_size: PositiveInt = 25
    learning_rate: PositiveFloat = 1e-3
    seeds: list[int] = [22, 33, 44, 55, 66]
    max_train_examples: PositiveInt | None = None
    max_test_examples: PositiveInt | None = None
    num_workers: NonNegativeInt = 0
    modernbert_model: str = "answerdotai/ModernBERT-base"

    use_cono: bool = False
    lambda_phone: PositiveFloat = 1.0
    lambda_word: PositiveFloat = 1.0
    lambda_utterance: PositiveFloat = 1.0
    lambda_cono: PositiveFloat = 0.1
    lambda_diversity: PositiveFloat = 1.0
    lambda_tightness: PositiveFloat = 1.0

    scenario: Literal["read_aloud", "free_speaking"] = "read_aloud"
    use_curriculum: bool = False
    curriculum_total_iters: PositiveInt | None = None

    trackio_project: str = "p014-hippo"
    trackio_space: str | None = None
