from __future__ import annotations

import json
from pathlib import Path

from p014 import cli as cli_module


def test_extract_ssl_command_invokes_extractor(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    called: dict[str, object] = {}

    def fake_extract_ssl_utterance_for_split(
        *,
        split: str,
        cache_dir: Path,
        dataset_id: str,
        device,
        max_examples: int | None = None,
    ) -> Path:
        called["split"] = split
        called["cache_dir"] = cache_dir
        called["dataset_id"] = dataset_id
        called["device"] = device
        called["max_examples"] = max_examples
        output_path = cache_dir / "features" / "ssl_utt" / f"{split}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return output_path

    monkeypatch.setattr(
        cli_module,
        "extract_ssl_utterance_for_split",
        fake_extract_ssl_utterance_for_split,
    )

    rc = cli_module.main(
        [
            "extract-ssl",
            "--split",
            "test",
            "--cache-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--max-examples",
            "7",
            "--json",
        ]
    )

    assert rc == 0
    assert called["split"] == "test"
    assert called["cache_dir"] == tmp_path
    assert called["dataset_id"] == "mispeech/speechocean762"
    assert str(called["device"]) == "cpu"
    assert called["max_examples"] == 7

    payload = json.loads(capsys.readouterr().out)
    assert payload["split"] == "test"
    assert payload["cache_path"].endswith("features/ssl_utt/test.pt")
