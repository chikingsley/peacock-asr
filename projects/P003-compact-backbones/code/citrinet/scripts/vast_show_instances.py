#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "python-dotenv>=1.2.1",
#   "vastai-sdk>=0.5.1",
# ]
# ///
# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from citrinet_vast.cli.vast_show_instances import main


if __name__ == "__main__":
    raise SystemExit(main())
