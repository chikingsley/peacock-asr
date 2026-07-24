"""Prepare and serve a playback-synchronous transcript review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MARKER = ".omni-transcript-review"
MARKER_CONTENT = '{"kind":"omni-transcript-review","version":1}\n'
ISSUE_KINDS = {"word", "formatting", "boundary", "private", "unsure"}
MAX_REQUEST_BYTES = 64 * 1024
SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    item_id TEXT PRIMARY KEY,
    verdict TEXT NOT NULL,
    correction TEXT,
    reviewed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS markers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    audio_time REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS markers_item_id ON markers(item_id);
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _validate_item(raw: dict[str, Any], sequence: int) -> dict[str, Any]:
    item_id = str(raw["item_id"])
    source = Path(str(raw["audio_path"])).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"audio does not exist for {item_id}: {source}")
    transcript = str(raw["transcript"]).strip()
    if not transcript:
        raise ValueError(f"empty transcript for {item_id}")
    words = raw.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError(f"aligned words are required for {item_id}")
    normalized_words: list[dict[str, object]] = []
    previous_end = 0.0
    for position, word in enumerate(words):
        text = str(word["text"]).strip()
        start = float(word["start"])
        end = float(word["end"])
        if not text or start < 0 or end <= start or start < previous_end - 0.05:
            raise ValueError(f"invalid word timing {position} for {item_id}")
        normalized_words.append({"text": text, "start": start, "end": end})
        previous_end = end
    suffix = source.suffix.lower() or ".audio"
    audio_name = f"{hashlib.sha256(item_id.encode()).hexdigest()[:20]}{suffix}"
    return {
        "item_id": item_id,
        "sequence": sequence,
        "audio_source": str(source),
        "audio": f"audio/{audio_name}",
        "transcript": transcript,
        "words": normalized_words,
        "session_id": str(raw.get("session_id", item_id)),
        "duration": float(raw.get("duration", previous_end)),
        "metadata": raw.get("metadata", {}),
    }


def prepare_review(
    manifest: Path, output_dir: Path, *, overwrite: bool = False
) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"review directory exists: {output_dir}")
        if not (output_dir / MARKER).is_file():
            raise ValueError(f"refusing to replace unmarked directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / MARKER).write_text(MARKER_CONTENT, encoding="utf-8")
    (output_dir / "audio").mkdir()
    items = [_validate_item(row, index + 1) for index, row in enumerate(_read_jsonl(manifest))]
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("item_id values must be unique")
    for item in items:
        destination = output_dir / str(item["audio"])
        destination.symlink_to(str(item["audio_source"]))
    public_items = [
        {key: value for key, value in item.items() if key != "audio_source"} for item in items
    ]
    (output_dir / "review_items.json").write_text(
        json.dumps({"version": 1, "items": public_items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    template = Path(__file__).with_name("transcript_review.html")
    shutil.copyfile(template, output_dir / "index.html")
    with sqlite3.connect(output_dir / "review.sqlite") as connection:
        connection.executescript(SCHEMA)
    return {
        "output_dir": str(output_dir),
        "items": len(items),
        "sessions": len({item["session_id"] for item in items}),
    }


def _manifest(review_dir: Path) -> dict[str, Any]:
    return json.loads((review_dir / "review_items.json").read_text(encoding="utf-8"))


def _state(review_dir: Path) -> dict[str, Any]:
    with sqlite3.connect(review_dir / "review.sqlite") as connection:
        connection.row_factory = sqlite3.Row
        reviews = {
            str(row["item_id"]): dict(row)
            for row in connection.execute("SELECT * FROM reviews ORDER BY reviewed_at")
        }
        markers: dict[str, list[dict[str, object]]] = {}
        for row in connection.execute("SELECT * FROM markers ORDER BY item_id, audio_time, id"):
            markers.setdefault(str(row["item_id"]), []).append(dict(row))
    items = _manifest(review_dir)["items"]
    return {
        "items": items,
        "reviews": reviews,
        "markers": markers,
        "summary": {
            "total": len(items),
            "reviewed": len(reviews),
            "remaining": len(items) - len(reviews),
            "markers": sum(len(rows) for rows in markers.values()),
        },
    }


class ReviewHandler(SimpleHTTPRequestHandler):
    review_dir: Path

    def __init__(self, *args: Any, review_dir: Path, **kwargs: Any) -> None:
        self.review_dir = review_dir
        super().__init__(*args, directory=str(review_dir), **kwargs)

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _payload(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        if size <= 0 or size > MAX_REQUEST_BYTES:
            raise ValueError("invalid request size")
        value = json.loads(self.rfile.read(size))
        item_ids = {str(item["item_id"]) for item in _manifest(self.review_dir)["items"]}
        if str(value.get("item_id")) not in item_ids:
            raise ValueError("unknown item_id")
        return value

    @staticmethod
    def _issue_kind(payload: dict[str, Any]) -> str:
        kind = str(payload["kind"])
        if kind not in ISSUE_KINDS:
            raise ValueError(f"invalid marker kind: {kind}")
        return kind

    @staticmethod
    def _verdict(payload: dict[str, Any]) -> str:
        verdict = str(payload["verdict"])
        if verdict not in {"accepted", "exact", "issues"}:
            raise ValueError(f"invalid verdict: {verdict}")
        return verdict

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/api/state", "/api/export.json"}:
            self._json(_state(self.review_dir))
            return
        if path == "/api/summary":
            self._json(_state(self.review_dir)["summary"])
            return
        super().do_GET()

    def do_POST(self) -> None:
        try:
            payload = self._payload()
            item_id = str(payload["item_id"])
            path = urlparse(self.path).path
            with sqlite3.connect(self.review_dir / "review.sqlite") as connection:
                if path == "/api/marker":
                    kind = self._issue_kind(payload)
                    audio_time = max(0.0, float(payload["audio_time"]))
                    cursor = connection.execute(
                        "INSERT INTO markers(item_id,kind,audio_time,created_at) VALUES(?,?,?,?)",
                        (item_id, kind, audio_time, time.time()),
                    )
                    self._json({"ok": True, "marker_id": cursor.lastrowid})
                    return
                if path == "/api/review":
                    verdict = self._verdict(payload)
                    correction = str(payload.get("correction", "")).strip() or None
                    insert_review = (
                        "INSERT INTO reviews(item_id,verdict,correction,reviewed_at) "
                        "VALUES(?,?,?,?) "
                        "ON CONFLICT(item_id) DO UPDATE SET verdict=excluded.verdict, "
                        "correction=excluded.correction, reviewed_at=excluded.reviewed_at"
                    )
                    connection.execute(
                        insert_review,
                        (item_id, verdict, correction, time.time()),
                    )
                    self._json({"ok": True})
                    return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/marker", "/api/review"}:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._payload()
            with sqlite3.connect(self.review_dir / "review.sqlite") as connection:
                if path == "/api/marker":
                    marker_id = int(payload["marker_id"])
                    connection.execute(
                        "DELETE FROM markers WHERE id=? AND item_id=?",
                        (marker_id, str(payload["item_id"])),
                    )
                else:
                    connection.execute(
                        "DELETE FROM reviews WHERE item_id=?", (str(payload["item_id"]),)
                    )
            self._json({"ok": True})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        print(f"transcript-review: {self.address_string()} {format % args}", flush=True)


def serve(review_dir: Path, *, host: str, port: int) -> None:
    review_dir = review_dir.expanduser().resolve()
    if not (review_dir / MARKER).is_file():
        raise FileNotFoundError(f"not a prepared transcript review: {review_dir}")

    def handler(*args: Any, **kwargs: Any) -> ReviewHandler:
        return ReviewHandler(*args, review_dir=review_dir, **kwargs)

    server = ThreadingHTTPServer((host, port), handler)
    print(f"Transcript review: http://{host}:{port}/ ({review_dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and serve aligned transcript review")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")
    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--review-dir", type=Path, required=True)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8766)
    summary = sub.add_parser("summary")
    summary.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_review(args.manifest, args.output_dir, overwrite=args.overwrite)
        print(json.dumps(result, indent=2))
    elif args.command == "serve":
        serve(args.review_dir, host=args.host, port=args.port)
    else:
        print(json.dumps(_state(args.review_dir)["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
