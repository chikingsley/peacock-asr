from __future__ import annotations

import json
from pathlib import Path

from p004_training_from_scratch.machine_manifest import capture_machine_manifest


def test_capture_machine_manifest_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "machine_manifest.json"
    payload = capture_machine_manifest(output=output)
    assert output.is_file()
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["tracking"]["wandb_project"] == payload["tracking"]["wandb_project"]
    assert "WANDB_API_KEY" not in written["public_env"]
    assert "VAST_API_KEY" not in written["public_env"]
