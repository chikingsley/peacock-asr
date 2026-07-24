import json
import sqlite3
from pathlib import Path

import pytest

from english_asr.dictation import _dev_session_ids, finalize_review


def _write_review_fixture(root: Path) -> Path:
    freeze = root / "review/frozen-v1"
    reviewer = freeze / "reviewer"
    reviewer.mkdir(parents=True)
    audio = freeze / "clip.flac"
    audio.write_bytes(b"audio")
    aligned = {
        "item_id": "clip-1",
        "session_id": "session-1",
        "audio_path": str(audio),
        "duration": 1.25,
        "transcript": "Cleaned output.",
        "metadata": {"teacher": {"model": "ARK"}},
    }
    (freeze / "review-aligned.jsonl").write_text(json.dumps(aligned) + "\n", encoding="utf-8")
    schema = """
        CREATE TABLE reviews (
            item_id TEXT PRIMARY KEY, verdict TEXT NOT NULL,
            correction TEXT, reviewed_at REAL NOT NULL
        );
        CREATE TABLE markers (
            id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL,
            kind TEXT NOT NULL, audio_time REAL NOT NULL, created_at REAL NOT NULL
        );
    """
    database = reviewer / "review.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(schema)
        connection.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?)",
            ("clip-1", "accepted", None, 1.0),
        )
    backup = reviewer / "manual.sqlite"
    with sqlite3.connect(backup) as connection:
        connection.execute("CREATE TABLE reviews (item_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO reviews VALUES (?)", ("clip-1",))
    return backup


def test_finalize_review_exports_product_gold(tmp_path: Path) -> None:
    backup = _write_review_fixture(tmp_path)
    result = finalize_review(tmp_path, manual_review_backup=backup)
    manifest = Path(str(result["manifest"]))
    row = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["text"] == "Cleaned output."
    assert row["review"]["surface"] == "ideal-pasted-dictation-v1"
    assert row["review"]["manually_spot_checked"] is True
    assert result["manual_spot_check_rows"] == 1
    assert result["bulk_accepted_rows"] == 0


def test_finalize_review_requires_complete_decisions(tmp_path: Path) -> None:
    backup = _write_review_fixture(tmp_path)
    database = tmp_path / "review/frozen-v1/reviewer/review.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM reviews")
    with pytest.raises(ValueError, match="review is incomplete"):
        finalize_review(tmp_path, manual_review_backup=backup)


def test_dev_sessions_are_deterministic_and_disjoint() -> None:
    rows = [
        {"session_id": f"session-{index}", "duration": 10.0}
        for index in range(20)
        for _ in range(2)
    ]
    first = _dev_session_ids(rows, fraction=0.1, seed=7)
    second = _dev_session_ids(list(reversed(rows)), fraction=0.1, seed=7)
    assert first == second
    assert len(first) == 2
    assert first < {str(row["session_id"]) for row in rows}
