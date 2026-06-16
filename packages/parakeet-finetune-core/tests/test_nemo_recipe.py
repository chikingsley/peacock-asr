from __future__ import annotations

from parakeet_finetune_core.nemo_recipe import build_command, build_parser, build_script_args
from parakeet_finetune_core.project import ParakeetProject


def test_nemo_recipe_command_uses_runpy_and_hydra_overrides(tmp_path):
    nemo_root = tmp_path / "nemo"
    script = nemo_root / "examples/asr/speech_to_text_finetune.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder\n", encoding="utf-8")
    project = ParakeetProject(
        name="tajik",
        language="tgk_Cyrl",
        root=tmp_path,
        nemo_root=nemo_root,
        default_tdt_model="nvidia/parakeet-tdt_ctc-110m",
    )
    parser = build_parser(project)
    args = parser.parse_args(
        [
            "--train-manifest",
            str(tmp_path / "train.jsonl"),
            "--validation-manifest",
            str(tmp_path / "dev.jsonl"),
            "--tokenizer-dir",
            str(tmp_path / "tok"),
            "--batch-size",
            "5",
            "--dry-run",
        ]
    )

    script_path, script_args = build_script_args(args)
    command = build_command(args)

    assert script_path == script
    assert command[0] == "runpy"
    assert str(script) in command
    assert all("python" not in item for item in command)
    assert f"--config-path={nemo_root / 'examples/asr/conf/asr_finetune'}" in script_args
    assert f"model.train_ds.manifest_filepath={(tmp_path / 'train.jsonl').resolve()}" in script_args
    assert "model.validation_ds.batch_size=5" in script_args
    assert "model.tokenizer.update_tokenizer=true" in script_args
    assert "+init_from_pretrained_model=nvidia/parakeet-tdt_ctc-110m" in script_args


def test_nemo_recipe_early_stopping_overrides_are_explicit(tmp_path):
    nemo_root = tmp_path / "nemo"
    script = nemo_root / "examples/asr/speech_to_text_finetune.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder\n", encoding="utf-8")
    project = ParakeetProject(
        name="farsi",
        language="fas_Arab",
        root=tmp_path,
        nemo_root=nemo_root,
    )
    args = build_parser(project).parse_args(
        [
            "--train-manifest",
            str(tmp_path / "train.jsonl"),
            "--validation-manifest",
            str(tmp_path / "dev.jsonl"),
            "--tokenizer-dir",
            str(tmp_path / "tok"),
            "--early-stopping",
            "--early-stopping-monitor",
            "val_loss",
            "--early-stopping-mode",
            "min",
            "--early-stopping-check-on-train-epoch-end",
        ]
    )

    _, script_args = build_script_args(args)

    assert "+exp_manager.create_early_stopping_callback=true" in script_args
    assert "+exp_manager.early_stopping_callback_params.monitor=val_loss" in script_args
    assert "+exp_manager.early_stopping_callback_params.mode=min" in script_args
    assert (
        "+exp_manager.early_stopping_callback_params.check_on_train_epoch_end=true"
        in script_args
    )
