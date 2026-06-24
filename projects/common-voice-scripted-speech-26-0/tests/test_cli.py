"""Import + ``--help`` smoke tests for every ``cv26`` subcommand (the project standard)."""

from __future__ import annotations

import importlib

import pytest

from cv26 import cli

SUBCOMMANDS = ["download", "process", "queue", "card"]
MODULES = [
    "config",
    "manifest",
    "download",
    "convert",
    "upload",
    "pipeline",
    "queue",
    "card",
    "cli",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    """Every package module imports cleanly."""
    importlib.import_module(f"cv26.{name}")


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_subcommand_parses(name: str) -> None:
    """Each subcommand parses with no extra args and binds a callable handler."""
    args = cli.build_parser().parse_args([name])
    assert callable(args.func)


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_subcommand_help(name: str) -> None:
    """``cv26 <cmd> --help`` exits 0 (argparse raises SystemExit(0))."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([name, "--help"])
    assert exc.value.code == 0
