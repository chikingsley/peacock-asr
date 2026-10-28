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


def test_continuation_restores_checkpoint_and_uses_upgraded_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text("audited training input")
    checkpoint = tmp_path / "source.ckpt"
    checkpoint.write_bytes(b"saved optimizer and model state")
    project = SimpleNamespace(runs=tmp_path / "runs", default_train_manifest=train)
    monkeypatch.setattr(upgrades, "PROJECT", project)
    upgrades.write_json(tmp_path / "plan.json", {"input": fingerprint(train)})
    runtime = {"loss_mode": "unfused", "precision": "bf16-mixed"}
    upgrades.write_json(tmp_path / "runtime-trials.json", {"selected": runtime})
    commands = []

    def command(_output: Path, _name: str, arguments: list[str]) -> int:
        assert (
            project.runs / "upgraded/checkpoints/last.ckpt"
        ).read_bytes() == checkpoint.read_bytes()
        commands.append(arguments)
        return 0

    monkeypatch.setattr(upgrades, "command", command)
    monkeypatch.setattr(upgrades, "training_result", lambda *_: {"steps": 100000})
    monkeypatch.setattr(upgrades, "candidates", lambda *_: {"best": tmp_path / "best.nemo"})
    monkeypatch.setattr(upgrades, "select_and_evaluate", lambda *_: {"selected": "best"})
    result = upgrades.continue_training(tmp_path, checkpoint, "upgraded")
    assert result["selected"] == "best"
    assert len(commands) == 1
    args = commands[0]
    assert "--resume" in args
    assert (
        "--keep-tokenizer" not in args
    )  # Original base + same replacement vocabulary precede resume.
    assert args[args.index("--loss-mode") + 1] == "unfused"
    assert args[args.index("--max-steps") + 1] == "100000"
    assert str(tmp_path / "manifests/gold") + "=0.12" in args
    report = json.loads((tmp_path / "continuation.json").read_text())
    assert report["resume_checkpoint"] == fingerprint(checkpoint)
    assert report["status"] == "complete"
    with pytest.raises(FileExistsError, match="Preserve"):
        upgrades.continue_training(tmp_path, checkpoint, "upgraded")


def test_asr_restoration_runs_when_continuation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(upgrades, "BASELINE_JOB", tmp_path)
    commands = []

    def run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(args)
        return SimpleNamespace(returncode=0)

    def fail(*_args: object) -> dict:
        raise ValueError("test continuation failure")

    monkeypatch.setattr(upgrades.subprocess, "run", run)
    monkeypatch.setattr(upgrades, "continue_training", fail)
    with pytest.raises(ValueError, match="test continuation failure"):
        upgrades.continue_with_asr_paused(tmp_path, tmp_path / "saved.ckpt", "upgraded")
    assert commands[-1] == [str(tmp_path / "restore-asr.sh")]
    assert not (tmp_path / "gpu-lease").exists()
