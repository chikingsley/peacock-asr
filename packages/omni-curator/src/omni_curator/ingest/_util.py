"""Small shared helpers for the ingest loaders."""

from __future__ import annotations


def slug(value: str) -> str:
    """Filesystem path / id fragment -> a safe underscore-joined id token."""
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")
