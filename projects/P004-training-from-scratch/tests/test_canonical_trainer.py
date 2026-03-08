from __future__ import annotations

from pathlib import Path

from p004_training_from_scratch.canonical import trainer


def test_frozen_stable_train_config_keeps_production_baseline() -> None:
    config = trainer.FROZEN_STABLE_TRAIN_CONFIG

    assert config.model_type == "conformer"
    assert config.attention_backend == "mha"
    assert config.enable_compile is False
    assert config.train_limit == 24
    assert config.dev_limit == 8
    assert config.epochs == 3
    assert config.batch_size == 4


def test_build_stable_train_config_returns_replaced_copy() -> None:
    config = trainer.build_stable_train_config(epochs=5, batch_size=6)

    assert config.epochs == 5
    assert config.batch_size == 6
    assert trainer.FROZEN_STABLE_TRAIN_CONFIG.epochs == 3
    assert trainer.FROZEN_STABLE_TRAIN_CONFIG.batch_size == 4


def test_run_canonical_train_delegates_to_smoke_runner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_canonical_train_smoke(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"success": True, "output_dir": str(kwargs["output_dir"])}

    monkeypatch.setattr(
        trainer,
        "run_canonical_train_smoke",
        fake_run_canonical_train_smoke,
    )

    config = trainer.build_stable_train_config(epochs=5, enable_compile=True)
    result = trainer.run_canonical_train(output_dir=tmp_path / "run", config=config)

    assert result["success"] is True
    assert len(calls) == 1
    assert calls[0]["epochs"] == 5
    assert calls[0]["enable_compile"] is True
    assert calls[0]["output_dir"] == tmp_path / "run"
