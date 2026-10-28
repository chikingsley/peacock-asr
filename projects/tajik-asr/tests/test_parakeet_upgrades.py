import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from parakeet_finetune_core.provenance import fingerprint

from tajik_asr.parakeet import upgrades


def test_refinement_selects_on_development_before_evaluating_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text("training input")
    dev = tmp_path / "dev.jsonl"
    project = SimpleNamespace(
        runs=tmp_path / "runs", default_train_manifest=train, default_validation_manifest=dev
    )
    monkeypatch.setattr(upgrades, "PROJECT", project)
    upgrades.write_json(tmp_path / "plan.json", {"input": fingerprint(train)})
    upgrades.write_json(tmp_path / "runtime-trials.json", {"selected": upgrades.DEFAULT_RUNTIME})

    def candidate_models(run: Path, _output: Path) -> dict:
        name = (
            "balanced"
            if "balanced" in run.name
            else "control"
            if "control" in run.name
            else "baseline"
        )
        return {name: tmp_path / f"{name}.nemo"}

    commands, evaluations = [], []

    def command(_output: Path, _name: str, arguments: list[str]) -> int:
        commands.append(arguments)
        return 0

    def evaluate(_output: Path, _name: str, model: Path, manifest: Path) -> dict:
        evaluations.append((model, manifest))
        score = (
            {"baseline": 20, "control": 19, "balanced": 18}[model.stem] if manifest == dev else 99
        )
        return {"normalized": {"wer_percent": score}}

    monkeypatch.setattr(upgrades, "candidates", candidate_models)
    monkeypatch.setattr(upgrades, "command", command)
    monkeypatch.setattr(upgrades, "training_result", lambda *_: {})
    monkeypatch.setattr(upgrades, "evaluate", evaluate)
    result = upgrades.refine(tmp_path)
    assert result["selected"] == "balanced"
    assert all(manifest == dev for _, manifest in evaluations[:3])
    assert all(model.stem == "balanced" for model, _ in evaluations[3:])
    assert all("--keep-tokenizer" in args for args in commands)
    assert any(str(value).endswith("gold=0.12") for value in commands[1])
    saved = json.loads((tmp_path / "selection.json").read_text())
    assert saved["selected"] == "balanced"


def test_asr_restoration_runs_when_refinement_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "completed-at.txt").touch()
    monkeypatch.setattr(upgrades, "BASELINE_JOB", tmp_path)
    commands = []

    def run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(args)
        return SimpleNamespace(stdout="inactive", returncode=0)

    def fail(_output: Path) -> dict:
        raise ValueError("test refinement failure")

    monkeypatch.setattr(upgrades.subprocess, "run", run)
    monkeypatch.setattr(upgrades, "refine", fail)
    with pytest.raises(ValueError, match="test refinement failure"):
        upgrades.wait_and_refine(tmp_path)
    assert commands[-1] == [str(tmp_path / "restore-asr.sh")]
    assert not (tmp_path / "gpu-lease").exists()
