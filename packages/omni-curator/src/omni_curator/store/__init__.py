"""store: persist curated samples in SQLite."""

from __future__ import annotations

from omni_curator.store.sqlite import CuratorStore, open_store

__all__ = ["CuratorStore", "open_store"]
