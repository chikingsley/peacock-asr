"""Prepare and serve a blinded listening review for VAD disagreement regions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from omni_curator.create.vad import SAMPLE_RATE
from omni_curator.process.audio import load_16k_mono

if TYPE_CHECKING:
    from collections.abc import Sequence

Interval = tuple[float, float]
LABELS = ("speech", "non_speech", "clipped", "unsure")
DIRECTIONS = {
    "marble_only": ("marblenet", "silero"),
    "silero_only": ("silero", "marblenet"),
}
TIERS = ("clean", "noisy")
REVIEW_MARKER = ".omni-vad-review"
REVIEW_MARKER_CONTENT = '{"kind":"omni-vad-review","version":1}\n'
REVIEW_CHILDREN = {
    REVIEW_MARKER,
    "audio",
    "index.html",
    "review.sqlite",
    "review_items.json",
}
SHORT_REGION_SECONDS = 0.35
MEDIUM_REGION_SECONDS = 0.75
MAX_REQUEST_BYTES = 8192
SCHEMA = """
CREATE TABLE IF NOT EXISTS votes (
    item_id       TEXT PRIMARY KEY,
    label         TEXT NOT NULL,
    replay_count  INTEGER NOT NULL DEFAULT 0,
    reviewed_at   REAL NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class Candidate:
    source_id: str
    source_path: str
    tier: str
    direction: str
    favored_engine: str
    opposed_engine: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2

    @property
    def candidate_id(self) -> str:
        raw = (f"{self.source_id}\0{self.direction}\0{self.start:.9f}\0{self.end:.9f}").encode()
        return hashlib.sha256(raw).hexdigest()[:20]


def _union_intervals(intervals: Sequence[Sequence[float]]) -> list[Interval]:
    merged: list[list[float]] = []
    for raw_start, raw_end in sorted((float(start), float(end)) for start, end in intervals):
        if raw_end <= raw_start:
            continue
        if merged and raw_start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], raw_end)
        else:
            merged.append([raw_start, raw_end])
    return [(start, end) for start, end in merged]


def _subtract_intervals(
    included: Sequence[Sequence[float]], excluded: Sequence[Sequence[float]]
) -> list[Interval]:
    out: list[Interval] = []
    excluded_union = _union_intervals(excluded)
    for start, end in _union_intervals(included):
        cursor = start
        for other_start, other_end in excluded_union:
            if other_end <= cursor:
                continue
            if other_start >= end:
                break
            if other_start > cursor:
                out.append((cursor, min(other_start, end)))
            cursor = max(cursor, other_end)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return out


def _split_candidate(start: float, end: float, *, maximum: float) -> list[Interval]:
    duration = end - start
    if duration <= maximum:
        return [(start, end)]
    count = math.ceil(duration / maximum)
    step = duration / count
    bounds = [start + index * step for index in range(count)] + [end]
    return [(bounds[index], bounds[index + 1]) for index in range(count)]


def load_candidates(
    intervals_path: Path,
    *,
    min_region_seconds: float = 0.18,
    max_region_seconds: float = 4.0,
) -> list[Candidate]:
    """Derive MarbleNet-only and Silero-only regions from one corrected pilot JSONL."""
    records = [
        json.loads(line)
        for line in intervals_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_key = {(str(row["source_id"]), str(row["engine"])): row for row in records}
    source_ids = sorted({str(row["source_id"]) for row in records})
    candidates: list[Candidate] = []
    for source_id in source_ids:
        missing = [
            engine for engine in ("marblenet", "silero") if (source_id, engine) not in by_key
        ]
        if missing:
            raise ValueError(f"source {source_id} is missing VAD engines: {missing}")
        marble = by_key[source_id, "marblenet"]
        silero = by_key[source_id, "silero"]
        if marble["path"] != silero["path"] or marble["tier"] != silero["tier"]:
            raise ValueError(f"source metadata differs across engines: {source_id}")
        for direction, (favored, opposed) in DIRECTIONS.items():
            regions = _subtract_intervals(
                by_key[source_id, favored]["intervals"],
                by_key[source_id, opposed]["intervals"],
            )
            for start, end in regions:
                for piece_start, piece_end in _split_candidate(
                    start, end, maximum=max_region_seconds
                ):
                    if piece_end - piece_start < min_region_seconds:
                        continue
                    candidates.append(
                        Candidate(
                            source_id=source_id,
                            source_path=str(marble["path"]),
                            tier=str(marble["tier"]),
                            direction=direction,
                            favored_engine=favored,
                            opposed_engine=opposed,
                            start=piece_start,
                            end=piece_end,
                        )
                    )
    return candidates


def _duration_bucket(candidate: Candidate) -> str:
    if candidate.duration < SHORT_REGION_SECONDS:
        return "short"
    if candidate.duration < MEDIUM_REGION_SECONDS:
        return "medium"
    return "long"


def _choose_source_balanced(
    pool: Sequence[Candidate],
    count: int,
    *,
    selected: list[Candidate],
    rng: random.Random,
) -> list[Candidate]:
    remaining = [candidate for candidate in pool if candidate not in selected]
    random_rank = {candidate.candidate_id: rng.random() for candidate in remaining}
    chosen: list[Candidate] = []
    while remaining and len(chosen) < count:
        all_selected = [*selected, *chosen]
        source_counts = Counter(candidate.source_id for candidate in all_selected)
        spaced = [
            candidate
            for candidate in remaining
            if all(
                other.source_id != candidate.source_id
                or abs(other.center - candidate.center) >= 1.0
                for other in all_selected
            )
        ]
        eligible = spaced or remaining
        candidate = min(
            eligible,
            key=lambda item: (source_counts[item.source_id], random_rank[item.candidate_id]),
        )
        chosen.append(candidate)
        remaining.remove(candidate)
    return chosen


def sample_candidates(
    candidates: Sequence[Candidate], *, total_items: int, seed: int
) -> list[Candidate]:
    """Select equal direction/tier cells, source-balanced and duration-stratified."""
    cell_count = len(DIRECTIONS) * len(TIERS)
    if total_items <= 0 or total_items % cell_count:
        raise ValueError(f"items must be a positive multiple of {cell_count} for exact balance")
    quota = total_items // cell_count
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    selected: list[Candidate] = []
    for direction in DIRECTIONS:
        for tier in TIERS:
            cell = [
                candidate
                for candidate in candidates
                if candidate.direction == direction and candidate.tier == tier
            ]
            if len(cell) < quota:
                raise ValueError(
                    f"not enough {direction}/{tier} candidates: need {quota}, have {len(cell)}"
                )
            bucket_targets = {
                "long": min(
                    sum(_duration_bucket(item) == "long" for item in cell), round(quota * 0.2)
                ),
                "medium": min(
                    sum(_duration_bucket(item) == "medium" for item in cell), round(quota * 0.3)
                ),
            }
            bucket_targets["short"] = quota - sum(bucket_targets.values())
            cell_selected: list[Candidate] = []
            for bucket in ("long", "medium", "short"):
                pool = [item for item in cell if _duration_bucket(item) == bucket]
                cell_selected.extend(
                    _choose_source_balanced(
                        pool,
                        bucket_targets[bucket],
                        selected=cell_selected,
                        rng=rng,
                    )
                )
            if len(cell_selected) < quota:
                cell_selected.extend(
                    _choose_source_balanced(
                        cell,
                        quota - len(cell_selected),
                        selected=cell_selected,
                        rng=rng,
                    )
                )
            if len(cell_selected) != quota:
                raise RuntimeError(
                    f"could not fill {direction}/{tier}: {len(cell_selected)} of {quota}"
                )
            selected.extend(cell_selected)
    rng.shuffle(selected)
    return selected


def _prepare_review_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"review output is not empty: {output_dir}; pass --overwrite")
        marker = output_dir / REVIEW_MARKER
        if not marker.is_file() or marker.read_text(encoding="utf-8") != REVIEW_MARKER_CONTENT:
            raise PermissionError(f"refusing to overwrite unmarked review directory: {output_dir}")
        unexpected = sorted(
            child.name for child in output_dir.iterdir() if child.name not in REVIEW_CHILDREN
        )
        if unexpected:
            raise PermissionError(f"refusing to overwrite unknown review files: {unexpected}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REVIEW_MARKER).write_text(REVIEW_MARKER_CONTENT, encoding="utf-8")
    (output_dir / "audio").mkdir()


def _marked_audio(
    audio: Any,
    *,
    start: float,
    end: float,
    context_seconds: float,
) -> Any:
    import numpy as np

    start_sample = max(0, round(start * SAMPLE_RATE))
    end_sample = min(len(audio), round(end * SAMPLE_RATE))
    context = round(context_seconds * SAMPLE_RATE)
    before = audio[max(0, start_sample - context) : start_sample]
    target = audio[start_sample:end_sample]
    after = audio[end_sample : min(len(audio), end_sample + context)]
    tone_samples = round(0.06 * SAMPLE_RATE)
    gap_samples = round(0.04 * SAMPLE_RATE)
    positions = np.arange(tone_samples, dtype=np.float32)
    tone = (0.22 * np.sin(2 * np.pi * 900.0 * positions / SAMPLE_RATE)).astype(np.float32)
    gap = np.zeros(gap_samples, dtype=np.float32)
    return np.concatenate((before, tone, gap, target, gap, tone, after)).astype(
        np.float32, copy=False
    )


def _isolated_audio(audio: Any, *, start: float, end: float) -> tuple[Any, int]:
    import numpy as np

    start_sample = max(0, round(start * SAMPLE_RATE))
    end_sample = min(len(audio), round(end * SAMPLE_RATE))
    target = audio[start_sample:end_sample]
    duration = len(target) / SAMPLE_RATE
    repetitions = (
        3 if duration < SHORT_REGION_SECONDS else 2 if duration < MEDIUM_REGION_SECONDS else 1
    )
    silence = np.zeros(round(0.25 * SAMPLE_RATE), dtype=np.float32)
    pieces: list[Any] = [silence]
    for repeat in range(repetitions):
        pieces.append(target)
        if repeat + 1 < repetitions:
            pieces.append(silence)
    pieces.append(silence)
    return np.concatenate(pieces).astype(np.float32, copy=False), repetitions


def prepare_review(
    *,
    intervals_path: Path,
    output_dir: Path,
    total_items: int = 160,
    seed: int = 20260709,
    context_seconds: float = 1.0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate isolated/context FLACs, blinded order, UI, and an empty vote database."""
    import soundfile as sf

    intervals_path = intervals_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _prepare_review_dir(output_dir, overwrite=overwrite)
    candidates = load_candidates(intervals_path)
    selected = sample_candidates(candidates, total_items=total_items, seed=seed)
    by_source: dict[str, list[tuple[int, Candidate]]] = defaultdict(list)
    for sequence, candidate in enumerate(selected, start=1):
        by_source[candidate.source_id].append((sequence, candidate))
    items: list[dict[str, object]] = []
    for _source_id, source_items in sorted(by_source.items()):
        source_path = Path(source_items[0][1].source_path)
        audio = load_16k_mono(source_path)
        for sequence, candidate in source_items:
            isolated, repetitions = _isolated_audio(
                audio,
                start=candidate.start,
                end=candidate.end,
            )
            context = _marked_audio(
                audio,
                start=candidate.start,
                end=candidate.end,
                context_seconds=context_seconds,
            )
            stem = f"{sequence:04d}_{candidate.candidate_id}.flac"
            relative_audio = Path("audio") / f"target_{stem}"
            relative_context = Path("audio") / f"context_{stem}"
            for relative_path, review_audio in (
                (relative_audio, isolated),
                (relative_context, context),
            ):
                sf.write(
                    str(output_dir / relative_path),
                    review_audio,
                    SAMPLE_RATE,
                    format="FLAC",
                    subtype="PCM_16",
                )
            item = {
                **asdict(candidate),
                "item_id": candidate.candidate_id,
                "sequence": sequence,
                "audio": relative_audio.as_posix(),
                "context_audio": relative_context.as_posix(),
                "repetitions": repetitions,
                "target_duration": candidate.duration,
                "review_duration": len(isolated) / SAMPLE_RATE,
                "context_duration": len(context) / SAMPLE_RATE,
                "duration_bucket": _duration_bucket(candidate),
            }
            items.append(item)
    items.sort(key=lambda item: int(item["sequence"]))
    manifest = {
        "schema_version": 2,
        "created_at": time.time(),
        "intervals_path": str(intervals_path),
        "intervals_sha256": hashlib.sha256(intervals_path.read_bytes()).hexdigest(),
        "seed": seed,
        "context_seconds": context_seconds,
        "item_count": len(items),
        "labels": {
            "speech": "The isolated disputed region contains clear speech.",
            "non_speech": "The isolated disputed region contains no speech.",
            "clipped": "The region contains a speech fragment or cut-off word/syllable.",
            "unsure": "Mixed or ambiguous; cannot make a confident binary judgment.",
        },
        "items": items,
    }
    (output_dir / "review_items.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    template = Path(__file__).with_suffix(".html")
    (output_dir / "index.html").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    with sqlite3.connect(output_dir / "review.sqlite") as connection:
        connection.executescript(SCHEMA)
    counts = Counter((str(item["direction"]), str(item["tier"])) for item in items)
    return {
        "output_dir": str(output_dir),
        "items": len(items),
        "cells": {
            f"{direction}/{tier}": count for (direction, tier), count in sorted(counts.items())
        },
        "sources": len({str(item["source_id"]) for item in items}),
        "audio_bytes": sum(
            (output_dir / str(item[path_key])).stat().st_size
            for item in items
            for path_key in ("audio", "context_audio")
        ),
    }


def _load_manifest(review_dir: Path) -> dict[str, Any]:
    return json.loads((review_dir / "review_items.json").read_text(encoding="utf-8"))


def _load_votes(review_dir: Path) -> dict[str, dict[str, object]]:
    with sqlite3.connect(review_dir / "review.sqlite") as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT item_id, label, replay_count, reviewed_at FROM votes"
        ).fetchall()
    return {str(row["item_id"]): dict(row) for row in rows}


def review_export(review_dir: Path) -> list[dict[str, object]]:
    manifest = _load_manifest(review_dir)
    votes = _load_votes(review_dir)
    return [
        {**item, **votes[str(item["item_id"])]}
        for item in manifest["items"]
        if str(item["item_id"]) in votes
    ]


def review_summary(review_dir: Path) -> dict[str, object]:
    manifest = _load_manifest(review_dir)
    rows = review_export(review_dir)
    cells: dict[str, Counter[str]] = defaultdict(Counter)
    engine_support: Counter[str] = Counter()
    for row in rows:
        cells[f"{row['direction']}/{row['tier']}"][str(row["label"])] += 1
        if row["label"] in {"speech", "clipped"}:
            engine_support[str(row["favored_engine"])] += 1
        elif row["label"] == "non_speech":
            engine_support[str(row["opposed_engine"])] += 1
    return {
        "total": int(manifest["item_count"]),
        "reviewed": len(rows),
        "remaining": int(manifest["item_count"]) - len(rows),
        "complete": len(rows) == int(manifest["item_count"]),
        "cells": {cell: dict(counts) for cell, counts in sorted(cells.items())},
        "engine_support": dict(engine_support),
        "unsure": sum(row["label"] == "unsure" for row in rows),
    }


class ReviewHandler(SimpleHTTPRequestHandler):
    """Serve the static reviewer and a tiny SQLite-backed voting API."""

    review_dir: Path

    def __init__(self, *args: Any, review_dir: Path, **kwargs: Any) -> None:
        self.review_dir = review_dir
        super().__init__(*args, directory=str(review_dir), **kwargs)

    def _send_json(self, value: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            manifest = _load_manifest(self.review_dir)
            votes = _load_votes(self.review_dir)
            public_items = [
                {
                    "item_id": item["item_id"],
                    "sequence": item["sequence"],
                    "audio": item["audio"],
                    "context_audio": item["context_audio"],
                    "repetitions": item["repetitions"],
                    "target_duration": item["target_duration"],
                    "review_duration": item["review_duration"],
                }
                for item in manifest["items"]
            ]
            self._send_json(
                {
                    "items": public_items,
                    "votes": votes,
                    "summary": review_summary(self.review_dir),
                }
            )
            return
        if path == "/api/summary":
            self._send_json(review_summary(self.review_dir))
            return
        if path == "/api/export.json":
            self._send_json(review_export(self.review_dir))
            return
        if path == "/api/export.csv":
            rows = review_export(self.review_dir)
            fields = sorted({key for row in rows for key in row})
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            raw = stream.getvalue().encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="vad-review.csv"')
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        super().do_GET()

    def _parse_vote(self) -> tuple[str, str, int]:
        size = int(self.headers.get("Content-Length", "0"))
        if size <= 0 or size > MAX_REQUEST_BYTES:
            raise ValueError("invalid request size")
        payload = json.loads(self.rfile.read(size))
        item_id = str(payload["item_id"])
        label = str(payload["label"])
        replay_count = max(0, int(payload.get("replay_count", 0)))
        if label not in LABELS:
            raise ValueError(f"invalid label: {label}")
        manifest = _load_manifest(self.review_dir)
        valid_ids = {str(item["item_id"]) for item in manifest["items"]}
        if item_id not in valid_ids:
            raise ValueError(f"unknown item: {item_id}")
        return item_id, label, replay_count

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/vote":
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            item_id, label, replay_count = self._parse_vote()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        with sqlite3.connect(self.review_dir / "review.sqlite") as connection:
            connection.execute(
                "INSERT INTO votes(item_id,label,replay_count,reviewed_at) VALUES(?,?,?,?) "
                "ON CONFLICT(item_id) DO UPDATE SET label=excluded.label, "
                "replay_count=excluded.replay_count, reviewed_at=excluded.reviewed_at",
                (item_id, label, replay_count, time.time()),
            )
        self._send_json({"ok": True, "summary": review_summary(self.review_dir)})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        print(f"review: {self.address_string()} {format % args}", flush=True)


def serve_review(review_dir: Path, *, host: str, port: int) -> None:
    review_dir = review_dir.expanduser().resolve()
    if not (review_dir / REVIEW_MARKER).is_file():
        raise FileNotFoundError(f"not a prepared VAD review directory: {review_dir}")

    def handler(*args: Any, **kwargs: Any) -> ReviewHandler:
        return ReviewHandler(*args, review_dir=review_dir, **kwargs)

    server = ThreadingHTTPServer((host, port), handler)
    print(f"VAD review: http://{host}:{port}/ ({review_dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and serve a blinded VAD listening review")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--intervals", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--items", type=int, default=160)
    prepare.add_argument("--seed", type=int, default=20260709)
    prepare.add_argument("--context-seconds", type=float, default=1.0)
    prepare.add_argument("--overwrite", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--review-dir", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    summary = sub.add_parser("summary")
    summary.add_argument("--review-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_review(
            intervals_path=args.intervals,
            output_dir=args.output_dir,
            total_items=args.items,
            seed=args.seed,
            context_seconds=args.context_seconds,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "serve":
        serve_review(args.review_dir, host=args.host, port=args.port)
        return 0
    print(json.dumps(review_summary(args.review_dir), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
