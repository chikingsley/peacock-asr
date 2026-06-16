from __future__ import annotations

from pathlib import Path

from parakeet_finetune_core.ctc import build_parser as build_ctc_parser
from parakeet_finetune_core.project import ParakeetProject
from parakeet_finetune_core.tokenizer import build_command, build_parser, tokenizer_data_root


def test_project_defaults_are_rooted_under_language_project(tmp_path):
    project = ParakeetProject(name="tajik", language="tgk_Cyrl", root=tmp_path)

    assert project.data == tmp_path / "data"
    assert project.models == tmp_path / "models" / "parakeet"
    assert project.tokenizers == tmp_path / "tokenizers" / "parakeet"
    assert project.runs == tmp_path / "runs" / "parakeet"


def test_tokenizer_command_uses_project_defaults(tmp_path):
    nemo_root = tmp_path / "nemo"
    script = nemo_root / "scripts/tokenizers/process_asr_text_tokenizer.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    project = ParakeetProject(
        name="farsi",
        language="fas_Arab",
        root=tmp_path,
        nemo_root=nemo_root,
        default_tokenizer_name="fa_spe_bpe_v1024",
    )

    parser = build_parser(project)
    args = parser.parse_args(["--manifest", str(manifest), "--dry-run"])

    assert tokenizer_data_root(project, args) == project.tokenizers / "fa_spe_bpe_v1024"
    command = build_command(project, args)
    assert str(script) in command
    assert "--spe_byte_fallback" in command
    assert "--no_lower_case" in command


def test_ctc_parser_allows_language_project_defaults(tmp_path):
    project = ParakeetProject(
        name="test",
        language="x",
        root=tmp_path,
        default_ctc_model=Path("/models/ctc.nemo"),
        default_train_manifest=Path("/data/train.jsonl"),
        default_validation_manifest=Path("/data/dev.jsonl"),
        default_tokenizer_dir=Path("/tok"),
    )

    args = build_ctc_parser(project).parse_args([])

    assert args.init_from_nemo_model == Path("/models/ctc.nemo")
    assert args.train_manifest == Path("/data/train.jsonl")
    assert args.validation_manifest == Path("/data/dev.jsonl")
    assert args.tokenizer_dir == Path("/tok")
