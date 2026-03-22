"""Shared Hatchet client setup for local development."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from hatchet_sdk import Hatchet

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV = Path(__file__).resolve().with_name(".env")

load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv(LOCAL_ENV, override=False)

hatchet = Hatchet(debug=True)
