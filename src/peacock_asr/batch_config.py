"""Pydantic schema and resolution helpers for batch YAML configs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from pathlib import Path

BatchMode = Literal["scalar", "feats", "gopt"]


class BatchDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: BatchMode | None = None
    repeats: int | None = Field(default=None, ge=1)
    device: str | None = None
    limit: int | None = Field(default=None, ge=0)
    workers: int | None = Field(default=None, ge=0)
    no_cache: bool | None = None
    verbose: bool | None = None
    seed: int | None = None
    seeds: list[int] | None = None

    @field_validator("seeds")
    @classmethod
    def _validate_seeds_non_empty(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) == 0:
            msg = "'seeds' must not be empty."
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_seed_fields(self) -> BatchDefaults:
        if self.seed is not None and self.seeds is not None:
            msg = "Provide only one of 'seed' or 'seeds'."
            raise ValueError(msg)
        return self


class BatchJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    backend: str
    mode: BatchMode | None = None
    repeats: int | None = Field(default=None, ge=1)
    device: str | None = None
    limit: int | None = Field(default=None, ge=0)
    workers: int | None = Field(default=None, ge=0)
    no_cache: bool | None = None
    verbose: bool | None = None
    seed: int | None = None
    seeds: list[int] | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if stripped == "":
            msg = "Job 'id' must not be empty when provided."
            raise ValueError(msg)
        return stripped

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            msg = "Job 'backend' must not be empty."
            raise ValueError(msg)
        return stripped

    @field_validator("seeds")
    @classmethod
    def _validate_seeds_non_empty(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) == 0:
            msg = "'seeds' must not be empty."
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_seed_fields(self) -> BatchJob:
        if self.seed is not None and self.seeds is not None:
            msg = "Job may define only one of 'seed' or 'seeds'."
            raise ValueError(msg)
        return self


class BatchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "batch"
    defaults: BatchDefaults = Field(default_factory=BatchDefaults)
    jobs: list[BatchJob]

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            msg = "Batch 'name' must not be empty."
            raise ValueError(msg)
        return stripped

    @field_validator("jobs")
    @classmethod
    def _validate_jobs_non_empty(cls, value: list[BatchJob]) -> list[BatchJob]:
        if len(value) == 0:
            msg = "Batch config must define at least one job."
            raise ValueError(msg)
        return value


class BatchCliDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str | None = None
    limit: int = Field(ge=0)
    workers: int = Field(ge=0)
    no_cache: bool
    verbose: bool


class BatchResolvedJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    backend: str
    mode: BatchMode
    repeats: int = Field(ge=1)
    device: str | None = None
    limit: int = Field(ge=0)
    workers: int = Field(ge=0)
    no_cache: bool
    verbose: bool
    seeds: list[int | None]

    @model_validator(mode="after")
    def _validate_seed_count(self) -> BatchResolvedJob:
        if len(self.seeds) != self.repeats:
            msg = (
                f"Resolved job '{self.job_id}' has {len(self.seeds)} seeds "
                f"for repeats={self.repeats}."
            )
            raise ValueError(msg)
        return self


def _coalesce[T](job_value: T | None, defaults_value: T | None, cli_value: T) -> T:
    if job_value is not None:
        return job_value
    if defaults_value is not None:
        return defaults_value
    return cli_value


def _resolve_seed_plan(
    *,
    job: BatchJob,
    defaults: BatchDefaults,
    repeats: int,
    job_id: str,
) -> tuple[int, list[int | None]]:
    seeds: list[int] | None
    base_seed: int | None

    if job.seeds is not None:
        seeds = job.seeds
        base_seed = None
    elif job.seed is not None:
        seeds = None
        base_seed = job.seed
    elif defaults.seeds is not None:
        seeds = defaults.seeds
        base_seed = None
    else:
        seeds = None
        base_seed = defaults.seed

    if seeds is not None:
        if repeats == 1:
            repeats = len(seeds)
        elif len(seeds) != repeats:
            msg = (
                f"Job '{job_id}' seeds length ({len(seeds)}) must match "
                f"repeats ({repeats})."
            )
            raise ValueError(msg)
        return repeats, [int(seed) for seed in seeds]

    if base_seed is not None:
        return repeats, [base_seed + offset for offset in range(repeats)]

    return repeats, [None] * repeats


def load_batch_spec(path: Path) -> BatchSpec:
    if not path.exists():
        msg = f"Batch config not found: {path}"
        raise FileNotFoundError(msg)

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        msg = "Batch config root must be a mapping."
        raise TypeError(msg)

    try:
        return BatchSpec.model_validate(data)
    except ValidationError as e:
        msg = f"Invalid batch config '{path}':\n{e}"
        raise ValueError(msg) from e


def resolve_batch_jobs(
    spec: BatchSpec,
    *,
    cli_defaults: BatchCliDefaults,
) -> list[BatchResolvedJob]:
    resolved: list[BatchResolvedJob] = []
    for index, job in enumerate(spec.jobs, start=1):
        job_id = job.id or f"job_{index}"
        mode = job.mode or spec.defaults.mode or "scalar"
        repeats = job.repeats or spec.defaults.repeats or 1
        repeats, seeds = _resolve_seed_plan(
            job=job, defaults=spec.defaults, repeats=repeats, job_id=job_id
        )

        resolved.append(
            BatchResolvedJob(
                job_id=job_id,
                backend=job.backend,
                mode=mode,
                repeats=repeats,
                device=_coalesce(job.device, spec.defaults.device, cli_defaults.device),
                limit=_coalesce(job.limit, spec.defaults.limit, cli_defaults.limit),
                workers=_coalesce(
                    job.workers, spec.defaults.workers, cli_defaults.workers
                ),
                no_cache=_coalesce(
                    job.no_cache, spec.defaults.no_cache, cli_defaults.no_cache
                ),
                verbose=_coalesce(
                    job.verbose, spec.defaults.verbose, cli_defaults.verbose
                ),
                seeds=seeds,
            )
        )
    return resolved
