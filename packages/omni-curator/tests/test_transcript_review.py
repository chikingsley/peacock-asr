import json
import sqlite3
from pathlib import Path

import pytest

from omni_curator.audit.transcript_review import MARKER, _state, prepare_review


def test_prepare_review_creates_resumable_bundle(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-placeholder")
    manifest = tmp_path / "items.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "item_id": "clip-1",
                "session_id": "session-1",
                "audio_path": str(audio),
                "transcript": "hello world",
                "words": [
                    {"text": "hello", "start": 0.0, "end": 0.4},
                    {"text": "world", "start": 0.5, "end": 0.9},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "review"
    result = prepare_review(manifest, output)
    assert result == {"output_dir": str(output), "items": 1, "sessions": 1}
    assert (output / MARKER).is_file()
    public = json.loads((output / "review_items.json").read_text(encoding="utf-8"))
    assert "audio_source" not in public["items"][0]
    assert (output / public["items"][0]["audio"]).resolve() == audio
    with sqlite3.connect(output / "review.sqlite") as connection:
        connection.execute(
            "INSERT INTO reviews(item_id,verdict,reviewed_at) VALUES(?,?,?)",
            ("clip-1", "accepted", 1.0),
        )
    assert _state(output)["summary"]["reviewed"] == 1


def test_prepare_requires_word_alignment(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    manifest = tmp_path / "items.jsonl"
    manifest.write_text(
        json.dumps({"item_id": "clip", "audio_path": str(audio), "transcript": "hello"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aligned words"):
        prepare_review(manifest, tmp_path / "review")
