"""Tests for batch YAML schema validation and resolution."""

from __future__ import annotations

import textwrap

import pytest

from p001_gop.batch_config import (
    BatchCliDefaults,
    load_batch_spec,
    resolve_batch_jobs,
)


def _write_batch_yaml(tmp_path, content: str):
    path = tmp_path / "batch.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_load_and_resolve_minimal_config_uses_cli_defaults(tmp_path):
    config_path = _write_batch_yaml(
        tmp_path,
        """
        jobs:
          - backend: original
        """,
    )
    spec = load_batch_spec(config_path)
    jobs = resolve_batch_jobs(
        spec,
        cli_defaults=BatchCliDefaults(
            device="cuda",
            limit=25,
            workers=8,
            no_cache=True,
            verbose=False,
            score_variant="gop_sf",
            score_alpha=0.5,
        ),
    )

    assert spec.name == "batch"
    assert len(jobs) == 1
    assert jobs[0].job_id == "job_1"
    assert jobs[0].backend == "original"
    assert jobs[0].mode == "scalar"
    assert jobs[0].repeats == 1
    assert jobs[0].device == "cuda"
    assert jobs[0].limit == 25
    assert jobs[0].workers == 8
    assert jobs[0].no_cache is True
    assert jobs[0].verbose is False
    assert jobs[0].score_variant == "gop_sf"
    assert jobs[0].score_alpha == 0.5
    assert jobs[0].seeds == [None]


def test_resolve_seeds_expand_repeats_when_repeats_default(tmp_path):
    config_path = _write_batch_yaml(
        tmp_path,
        """
        jobs:
          - id: seeded
            backend: xlsr-espeak
            mode: gopt
            seeds: [10, 11, 12]
        """,
    )
    spec = load_batch_spec(config_path)
    jobs = resolve_batch_jobs(
        spec,
        cli_defaults=BatchCliDefaults(
            device=None,
            limit=0,
            workers=0,
            no_cache=False,
            verbose=False,
            score_variant="gop_sf",
            score_alpha=0.5,
        ),
    )

    assert len(jobs) == 1
    assert jobs[0].repeats == 3
    assert jobs[0].seeds == [10, 11, 12]


def test_rejects_mode_aliases_not_in_schema(tmp_path):
    config_path = _write_batch_yaml(
        tmp_path,
        """
        jobs:
          - backend: original
            mode: svr-feats
        """,
    )

    with pytest.raises(ValueError, match="mode"):
        load_batch_spec(config_path)


def test_rejects_unknown_fields(tmp_path):
    config_path = _write_batch_yaml(
        tmp_path,
        """
        jobs:
          - backend: original
            extra_field: true
        """,
    )

    with pytest.raises(ValueError, match="extra_field"):
        load_batch_spec(config_path)


def test_rejects_seed_and_seeds_together(tmp_path):
    config_path = _write_batch_yaml(
        tmp_path,
        """
        jobs:
          - backend: original
            seed: 10
            seeds: [10, 11]
        """,
    )

    with pytest.raises(ValueError, match="seed"):
        load_batch_spec(config_path)


def test_rejects_seed_length_mismatch_on_resolution(tmp_path):
    config_path = _write_batch_yaml(
        tmp_path,
        """
        jobs:
          - id: mismatch
            backend: original
            repeats: 2
            seeds: [100, 101, 102]
        """,
    )
    spec = load_batch_spec(config_path)

    with pytest.raises(ValueError, match="mismatch"):
        resolve_batch_jobs(
            spec,
            cli_defaults=BatchCliDefaults(
                device=None,
                limit=0,
                workers=0,
                no_cache=False,
                verbose=False,
                score_variant="gop_sf",
                score_alpha=0.5,
            ),
        )


def test_resolve_job_can_override_score_variant_and_alpha(tmp_path):
    config_path = _write_batch_yaml(
        tmp_path,
        """
        defaults:
          score_variant: gop_sf
          score_alpha: 0.5
        jobs:
          - id: margin
            backend: original
            score_variant: logit_margin
            score_alpha: 0.25
        """,
    )
    spec = load_batch_spec(config_path)
    jobs = resolve_batch_jobs(
        spec,
        cli_defaults=BatchCliDefaults(
            device=None,
            limit=0,
            workers=0,
            no_cache=False,
            verbose=False,
            score_variant="gop_sf",
            score_alpha=0.5,
        ),
    )

    assert jobs[0].score_variant == "logit_margin"
    assert jobs[0].score_alpha == 0.25


def test_rejects_score_alpha_out_of_range(tmp_path):
    config_path = _write_batch_yaml(
        tmp_path,
        """
        jobs:
          - backend: original
            score_alpha: 1.5
        """,
    )

    with pytest.raises(ValueError, match="score_alpha"):
        load_batch_spec(config_path)
