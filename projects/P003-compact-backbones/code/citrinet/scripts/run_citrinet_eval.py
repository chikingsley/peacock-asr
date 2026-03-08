#!/usr/bin/env python3
"""Run the standard P003 evaluation CLI from the Citrinet environment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
CODE_ROOT = REPO_ROOT / "projects/P003-compact-backbones/code"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def main() -> None:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(REPO_ROOT / "projects/P003-compact-backbones/.env")
    except ImportError:
        pass

    from p003_compact.cli import main as cli_main  # noqa: PLC0415

    cli_main()


if __name__ == "__main__":
    main()
