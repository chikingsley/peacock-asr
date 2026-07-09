"""FinetuneProject CLI: presets are typed configs; the regime path fails fast and budgets steps."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any, cast

import pytest

from omni_finetune_core import train as train_mod
from omni_finetune_core.presets import gpu_max_finetune
from omni_finetune_core.project import (
    FinetuneProject,
    TrainingPreset,
    _load_manifest_test,
    _regime_config,
    _safe_output_label,
    _write_eval_predictions,
    build_train_parser,
)

if TYPE_CHECKING:
    from pathlib import Path


def dig(d: object, *keys: str) -> Any:
    """Walk nested recipe-dict keys (typed as object) for assertions."""
    for key in keys:
        d = cast("dict[str, object]", d)[key]
    return d


@pytest.fixture
def project(tmp_path):
    summary_dir = tmp_path / "datasets" / "v0"
    summary_dir.mkdir(parents=True)
    (summary_dir / "language_distribution_0.tsv").write_text(
        "corpus\tlanguage\thours\nfleurs\txx\t490.0\n"  # weighted: hours here are FAKE
    )
    (summary_dir / "export_summary.json").write_text(json.dumps({"hours": 145.3}))
    return FinetuneProject(
        name="testlang",
        language="xxx_Test",
        root=tmp_path,
        presets={
            "v0-300m": TrainingPreset(
                config=lambda: gpu_max_finetune(
                    model="m",
                    dataset="d",
                    tokenizer="t",
                    dataset_summary_path="s.tsv",
                    num_steps=100,
                ),
                output_dir=tmp_path / "runs" / "v0",
            )
        },
        model_card="m",
        dataset_card="d",
        tokenizer_card="t",
        dataset_summary_path=summary_dir / "language_distribution_0.tsv",
        fragment_cache_dir=tmp_path / "cache",
        default_output_dir=tmp_path / "runs" / "generic",
    )


def test_corpus_hours_reads_true_hours_not_the_weighted_tsv(project):
    # The TSV says 490 (a sampling WEIGHT); the truth lives in export_summary.json.
    assert project.corpus_hours() == 145.3


def test_corpus_hours_fails_fast_without_summary(tmp_path):
    bare = FinetuneProject(
        name="t",
        language="x",
        root=tmp_path,
        dataset_summary_path=tmp_path / "nonexistent.tsv",
    )
    with pytest.raises(SystemExit, match="export summary not found"):
        bare.corpus_hours()


def test_preset_config_is_typed_and_threads_cache_dir():
    cfg = gpu_max_finetune(
        model="m",
        dataset="d",
        tokenizer="t",
        dataset_summary_path="s.tsv",
        num_steps=20_000,
        lr=5e-6,
        validate_every=500,
        fragment_cache_dir="/data/cache/fragments",
    )
    d = cfg.to_recipe_dict()
    assert (
        dig(d, "dataset", "mixture_parquet_storage_config", "fragment_loading", "cache_dir")
        == "/data/cache/fragments"
    )
    assert dig(d, "regime", "num_steps") == 20_000
    assert dig(d, "optimizer", "config", "lr") == 5e-6
    assert dig(d, "regime", "score_metric") == "wer"


def test_cache_dir_omitted_when_unset():
    cfg = gpu_max_finetune(
        model="m", dataset="d", tokenizer="t", dataset_summary_path="s.tsv", num_steps=10
    )
    loading = dig(
        cfg.to_recipe_dict(), "dataset", "mixture_parquet_storage_config", "fragment_loading"
    )
    assert "cache_dir" not in loading  # exclude_none keeps the YAML clean


def test_regime_config_builds_all_three(project):
    for regime in ("gpu_max", "1b", "warm_restart"):
        d = _regime_config(project, regime, 1_000, None).to_recipe_dict()
        cache_dir = dig(
            d, "dataset", "mixture_parquet_storage_config", "fragment_loading", "cache_dir"
        )
        assert cache_dir.endswith("cache")
        assert dig(d, "regime", "num_steps") == 1_000


def test_regime_config_fails_fast_on_missing_cards(tmp_path):
    bare = FinetuneProject(name="t", language="x", root=tmp_path)
    with pytest.raises(SystemExit, match="--regime needs"):
        _regime_config(bare, "gpu_max", 100, None)


def test_train_parser_offers_presets_and_regimes(project):
    parser = build_train_parser(project)
    args = parser.parse_args(["--preset", "v0-300m"])
    assert args.preset == "v0-300m"
    args = parser.parse_args(["--regime", "warm_restart", "--lr", "2e-6"])
    assert args.regime == "warm_restart"
    assert args.lr == 2e-6
    with pytest.raises(SystemExit):
        parser.parse_args(["--preset", "nonexistent"])


def test_run_recipe_restores_sys_argv(tmp_path, monkeypatch):
    outer_argv = ["outer-command", "--still-here"]
    monkeypatch.setattr(sys, "argv", outer_argv[:])
    seen: dict[str, object] = {}

    def fake_run_module(module: str, *, run_name: str) -> None:
        seen["module"] = module
        seen["run_name"] = run_name
        seen["argv"] = sys.argv[:]

    monkeypatch.setattr(train_mod.runpy, "run_module", fake_run_module)

    train_mod.run_recipe(tmp_path / "config.yaml", tmp_path / "run", extra_args=["--dry-run"])

    assert seen == {
        "module": train_mod.RECIPE_MODULE,
        "run_name": "__main__",
        "argv": [
            train_mod.RECIPE_MODULE,
            str(tmp_path / "run"),
            "--config-file",
            str(tmp_path / "config.yaml"),
            "--dry-run",
        ],
    }
    assert sys.argv == outer_argv


def test_load_manifest_test_reads_encoded_audio_and_filters_duration(tmp_path: Path) -> None:
    first = tmp_path / "first.flac"
    second = tmp_path / "second.flac"
    first.write_bytes(b"fLaC-one")
    second.write_bytes(b"fLaC-two")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "audio_filepath": str(first),
                        "text": "one",
                        "duration": 1.0,
                        "corpus": "youtube-one",
                    }
                ),
                json.dumps(
                    {
                        "audio_filepath": str(second),
                        "text": "two",
                        "duration": 41.0,
                        "corpus": "youtube-two",
                    }
                ),
            ]
        )
        + "\n"
    )

    audio, refs, corpora, excluded = _load_manifest_test(manifest, limit=0, max_dur=40.0)

    assert [item.tobytes() for item in audio] == [b"fLaC-one"]
    assert refs == ["one"]
    assert corpora == ["youtube-one"]
    assert excluded == 1


def test_eval_prediction_artifacts_are_joinable_by_row_index(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"

    _write_eval_predictions(
        output,
        ["Raw Ref"],
        ["Raw Hyp"],
        ["raw ref"],
        ["raw hyp"],
        ["youtube-demo"],
    )

    assert json.loads(output.read_text()) == {
        "row_index": 0,
        "corpus": "youtube-demo",
        "text": "Raw Ref",
        "hypothesis": "Raw Hyp",
        "normalized_text": "raw ref",
        "normalized_hypothesis": "raw hyp",
    }
    assert _safe_output_label("v3/card name") == "v3_card_name"
