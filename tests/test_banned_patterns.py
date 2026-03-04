"""Enforce banned code patterns across the codebase.

These tests grep src/ and training/ for patterns we've banned.
Fails fast in CI so nobody (human or AI) can sneak them in.
"""

from __future__ import annotations

import re
from pathlib import Path

# Directories to check (relative to repo root)
CHECKED_DIRS = ["src", "training"]
REPO_ROOT = Path(__file__).parent.parent


def _collect_python_files() -> list[Path]:
    files = []
    for d in CHECKED_DIRS:
        files.extend((REPO_ROOT / d).rglob("*.py"))
    return files


def test_no_os_environ_get():
    """Ban os.environ.get — use pydantic BaseSettings + dotenv instead."""
    pattern = re.compile(r"os\.environ\.get\s*\(")
    violations = []
    for path in _collect_python_files():
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line) and not line.strip().startswith("#"):
                violations.append(f"  {path.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not violations, (
        "os.environ.get() is banned. Use pydantic BaseSettings instead.\n"
        + "\n".join(violations)
    )


def test_no_os_environ_read():
    """Ban os.environ['KEY'] reads — use pydantic BaseSettings + dotenv instead.

    Allows os.environ['KEY'] = 'value' (setting env vars for libraries).
    """
    # Match os.environ["KEY"] but NOT os.environ["KEY"] = ...
    pattern = re.compile(r"os\.environ\[")
    assignment = re.compile(r"os\.environ\[.*\]\s*=")
    violations = []
    for path in _collect_python_files():
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if pattern.search(line) and not assignment.search(line):
                violations.append(f"  {path.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not violations, (
        "os.environ['KEY'] reads are banned. Use pydantic BaseSettings instead.\n"
        + "\n".join(violations)
    )
