import json
from types import SimpleNamespace

from parakeet_finetune_core.project import ParakeetProject
from parakeet_finetune_core.provenance import record_run


def test_run_history_preserves_input_versions_across_invocations(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("first version")
    validation = tmp_path / "dev.jsonl"
    validation.write_text("held-out")
    args = SimpleNamespace(
        dry_run=False,
        prepare_only=False,
        train_source=[],
        train_manifest=train,
        validation_manifest=validation,
        model_name="upstream/model",
        tokenizer_dir=None,
        exp_dir=tmp_path,
        name="run",
        seed=123,
    )
    project = ParakeetProject(name="example", language="example", root=tmp_path)
    record_run(project, args)
    train.write_text("second version")
    record_run(project, args)
    records = [
        json.loads(line)
        for line in (tmp_path / "run/run_invocations.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[0]["arguments"]["seed"] == 123
    assert records[0]["inputs"][1]["sha256"] != records[1]["inputs"][1]["sha256"]
