"""Build the canonical owned-dictation session ledger from app database snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import local
from typing import Any

import httpx
from omni_curator.create.vad import (
    VadPolicy,
    build_vad_policy,
    load_vad_engine,
    segment_audio_with,
    segmentation_metadata,
)
from omni_curator.process.audio import load_16k_mono, to_16k_flac, write_clip_16k

DEFAULT_IMPORT = (
    Path("/mnt/tiny-2t/peacock-asr/english-asr/data/owned-dictation") / "mac-import-20260716"
)
SHORT_SESSION_SECONDS = 60
MEDIUM_SESSION_SECONDS = 180
CTM_MIN_FIELDS = 5
REVIEW_GOLD_VERSION = 1
TRAINING_DERIVATIVE_VERSION = 1
DEFAULT_TRAINING_DERIVATIVE = "owned-ark-25h-v1"
MAX_CONSECUTIVE_LABEL_FAILURES = 12


def _duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe is required to inventory dictation audio")
    result = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _macwhisper_rows(root: Path) -> list[dict[str, Any]]:
    database = root / "raw/databases/macwhisper.sqlite"
    audio_root = root / "raw/audio/macwhisper"
    query = """
        SELECT hex(d.id) AS id, d.dateCreated AS created_at,
               d.transcribedText AS transcript, d.targetAppBundleID AS target_app,
               m.filename AS filename
        FROM dictation d JOIN mediafile m ON m.id = d.mediaFileID
        WHERE d.transcriptionDidSucceed = 1 AND d.dateDeleted IS NULL
        ORDER BY d.dateCreated, id
    """
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query).fetchall()
    return [
        {
            "session_id": f"macwhisper-{str(row['id']).lower()}",
            "source": "macwhisper",
            "created_at": str(row["created_at"]),
            "audio_path": str((audio_root / str(row["filename"])).resolve()),
            "original_transcript": str(row["transcript"] or "").strip(),
            "target_app": row["target_app"],
        }
        for row in rows
    ]


def _timbervox_rows(root: Path) -> list[dict[str, Any]]:
    database = root / "raw/databases/timbervox.sqlite"
    audio_root = root / "raw/audio/timbervox-native"
    query = """
        SELECT id, text, createdAt AS created_at, durationSeconds AS stored_duration,
               audioPath, model, sourceApplicationBundleIdentifier AS target_app
        FROM transcripts
        WHERE status = 'succeeded' AND importSource IS NULL AND audioPath IS NOT NULL
              AND audioPath NOT LIKE '%/Imported/MacWhisper/%'
        ORDER BY createdAt, id
    """
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query).fetchall()
    return [
        {
            "session_id": f"timbervox-{row['id']}",
            "source": "timbervox-native",
            "created_at": str(row["created_at"]),
            "audio_path": str((audio_root / Path(str(row["audioPath"])).name).resolve()),
            "original_transcript": str(row["text"]).strip(),
            "target_app": row["target_app"],
            "original_model": row["model"],
            "stored_duration": float(row["stored_duration"]),
        }
        for row in rows
    ]


def build_ledger(root: Path, *, output: Path | None = None) -> dict[str, object]:
    root = root.expanduser().resolve()
    output = (output or root / "ledgers/sessions.jsonl").expanduser().resolve()
    rows = [*_macwhisper_rows(root), *_timbervox_rows(root)]
    missing = [row["audio_path"] for row in rows if not Path(str(row["audio_path"])).is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"{len(missing)} ledger audio files are missing: {preview}")
    for row in rows:
        path = Path(str(row["audio_path"]))
        row["duration"] = _duration(path)
        row["audio_sha256"] = _sha256(path)
        row["privacy_state"] = "unreviewed"
    rows.sort(key=lambda row: (str(row["created_at"]), str(row["session_id"])))
    hashes: dict[str, list[str]] = {}
    for row in rows:
        hashes.setdefault(str(row["audio_sha256"]), []).append(str(row["session_id"]))
    duplicate_ids = {
        session_id
        for session_ids in hashes.values()
        if len(session_ids) > 1
        for session_id in session_ids[1:]
    }
    for row in rows:
        row["duplicate"] = str(row["session_id"]) in duplicate_ids
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    unique_rows = [row for row in rows if not row["duplicate"]]
    return {
        "output": str(output),
        "sessions": len(rows),
        "unique_sessions": len(unique_rows),
        "duplicates": len(rows) - len(unique_rows),
        "hours": sum(float(row["duration"]) for row in unique_rows) / 3600,
        "sources": {
            source: sum(row["source"] == source and not row["duplicate"] for row in rows)
            for source in sorted({str(row["source"]) for row in rows})
        },
    }


def _duration_bucket(seconds: float) -> str:
    if seconds < SHORT_SESSION_SECONDS:
        return "short"
    if seconds < MEDIUM_SESSION_SECONDS:
        return "medium"
    return "long"


def _select_sessions(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise ValueError(f"session count must be between 1 and {len(rows)}")
    rng = random.Random(seed)  # noqa: S311 - reproducible held-out sampling
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["source"]), _duration_bucket(float(row["duration"])))
        cells.setdefault(key, []).append(row)
    for cell in cells.values():
        rng.shuffle(cell)
    selected: list[dict[str, Any]] = []
    keys = sorted(cells)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if cells[key] and len(selected) < count:
                selected.append(cells[key].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda row: str(row["session_id"]))


def _sample_segments(
    candidates: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count <= 0 or count > len(candidates):
        raise ValueError(f"segment count must be between 1 and {len(candidates)}")
    rng = random.Random(seed)  # noqa: S311 - reproducible review sampling
    random_rank = {str(row["item_id"]): rng.random() for row in candidates}
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while len(selected) < count:
        session_counts: dict[str, int] = {}
        for row in selected:
            session = str(row["session_id"])
            session_counts[session] = session_counts.get(session, 0) + 1
        row = min(
            remaining,
            key=lambda item: (
                session_counts.get(str(item["session_id"]), 0),
                -float(item["duration"]),
                random_rank[str(item["item_id"])],
            ),
        )
        selected.append(row)
        remaining.remove(row)
    return selected


def freeze_review(
    root: Path,
    *,
    session_count: int,
    segment_count: int,
    seed: int,
    overwrite: bool = False,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    ledger = root / "ledgers/sessions.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    eligible = [
        row for row in rows if not row.get("duplicate") and row["privacy_state"] != "exclude"
    ]
    sessions = _select_sessions(eligible, session_count, seed)
    output = root / "review/frozen-v1"
    marker = output / ".english-dictation-review-freeze"
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"frozen review exists: {output}")
        if not marker.is_file():
            raise ValueError(f"refusing to replace unmarked directory: {output}")
        shutil.rmtree(output)
    (output / "segments").mkdir(parents=True)
    (output / "sessions-16k").mkdir()
    marker.write_text('{"kind":"english-dictation-review-freeze","version":1}\n', encoding="utf-8")
    policy = build_vad_policy(
        engine="silero",
        profile="conservative-v1",
        max_speech_s=30,
        threshold=0.5,
        silero_backend="onnx",
    )
    engine = load_vad_engine(policy, device="cpu")
    candidates: list[dict[str, Any]] = []
    session_audio_paths: dict[str, Path] = {}
    try:
        for session in sessions:
            session_id = str(session["session_id"])
            canonical_audio = output / "sessions-16k" / f"{session_id}.flac"
            to_16k_flac(Path(str(session["audio_path"])), canonical_audio)
            session_audio_paths[session_id] = canonical_audio
            audio = load_16k_mono(canonical_audio)
            result = segment_audio_with(engine, audio, policy=policy)
            vad_metadata = segmentation_metadata(policy, engine, result)
            for position, (start, end) in enumerate(result.intervals):
                item_id = f"{session['session_id']}-{position:04d}"
                candidates.append(
                    {
                        "item_id": item_id,
                        "session_id": session["session_id"],
                        "source": session["source"],
                        "source_audio_path": session["audio_path"],
                        "start": round(start, 6),
                        "end": round(end, 6),
                        "duration": round(end - start, 6),
                        "created_at": session["created_at"],
                        "target_app": session.get("target_app"),
                        "vad": vad_metadata,
                    }
                )
    finally:
        engine.close()
    sampled = _sample_segments(candidates, segment_count, seed)
    (output / "candidate_segments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    by_session = {str(row["session_id"]): row for row in sessions}
    sampled_by_session: dict[str, list[dict[str, Any]]] = {}
    for row in sampled:
        sampled_by_session.setdefault(str(row["session_id"]), []).append(row)
    for session_id, session_rows in sampled_by_session.items():
        session = by_session[session_id]
        audio = load_16k_mono(session_audio_paths[session_id])
        for row in session_rows:
            destination = output / "segments" / f"{row['item_id']}.flac"
            write_clip_16k(audio, destination, float(row["start"]), float(row["end"]))
            row["audio_path"] = str(destination)
    (output / "heldout_sessions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sessions),
        encoding="utf-8",
    )
    manifest = output / "review_segments.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sampled),
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "manifest": str(manifest),
        "heldout_sessions": len(sessions),
        "candidate_segments": len(candidates),
        "review_segments": len(sampled),
        "review_hours": sum(float(row["duration"]) for row in sampled) / 3600,
        "represented_sessions": len({str(row["session_id"]) for row in sampled}),
        "vad_policy": policy.as_dict(),
    }


def _teacher_transcript(payload: dict[str, Any]) -> str:
    transcript = str(payload["text"]).strip()
    if not transcript:
        raise ValueError("empty teacher transcript")
    return transcript


def label_review(
    root: Path,
    *,
    url: str,
    model: str,
    revision: str,
    timeout: float,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    freeze = root / "review/frozen-v1"
    rows = [
        json.loads(line)
        for line in (freeze / "review_segments.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    raw_dir = freeze / "teacher/ark/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=url.rstrip("/"), timeout=timeout) as client:
        models_response = client.get("/v1/models")
        service_models = models_response.json() if models_response.is_success else None
        for row in rows:
            raw_path = raw_dir / f"{row['item_id']}.json"
            if raw_path.is_file():
                previous = json.loads(raw_path.read_text(encoding="utf-8"))
                if previous.get("status") == "ok":
                    continue
            audio_path = Path(str(row["audio_path"]))
            try:
                with audio_path.open("rb") as audio:
                    response = client.post(
                        "/v1/audio/transcriptions",
                        data={"model": model},
                        files={"file": (audio_path.name, audio, "audio/flac")},
                    )
                response.raise_for_status()
                payload = response.json()
                transcript = _teacher_transcript(payload)
                record = {
                    "status": "ok",
                    "transcript": transcript,
                    "response": payload,
                    "labeled_at": datetime.now(UTC).isoformat(),
                }
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                record = {
                    "status": "failed",
                    "error": str(exc),
                    "labeled_at": datetime.now(UTC).isoformat(),
                }
            raw_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    labeled: list[dict[str, Any]] = []
    for row in rows:
        raw_path = raw_dir / f"{row['item_id']}.json"
        record = json.loads(raw_path.read_text(encoding="utf-8"))
        if record["status"] != "ok":
            continue
        labeled.append(
            {
                **row,
                "transcript": record["transcript"],
                "teacher": {
                    "model": model,
                    "revision": revision,
                    "url": url,
                    "service_models": service_models,
                    "labeled_at": record["labeled_at"],
                    "raw_response": str(raw_path),
                },
            }
        )
    output = freeze / "teacher/ark/labeled.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in labeled),
        encoding="utf-8",
    )
    alignment = freeze / "teacher/ark/alignment-input.jsonl"
    alignment.write_text(
        "".join(
            json.dumps(
                {
                    "audio_filepath": row["audio_path"],
                    "text": row["transcript"],
                    "duration": row["duration"],
                    "item_id": row["item_id"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for row in labeled
        ),
        encoding="utf-8",
    )
    failures = len(rows) - len(labeled)
    return {
        "input_rows": len(rows),
        "labeled_rows": len(labeled),
        "failures": failures,
        "output": str(output),
        "alignment_input": str(alignment),
    }


def assemble_review(root: Path, *, ctm_dir: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    freeze = root / "review/frozen-v1"
    labeled_path = freeze / "teacher/ark/labeled.jsonl"
    rows = [
        json.loads(line) for line in labeled_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assembled: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        ctm = ctm_dir.expanduser().resolve() / "words" / f"{Path(row['audio_path']).stem}.ctm"
        if not ctm.is_file():
            missing.append(str(row["item_id"]))
            continue
        words: list[dict[str, object]] = []
        for line in ctm.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < CTM_MIN_FIELDS:
                continue
            start = float(fields[2])
            words.append({"text": fields[4], "start": start, "end": start + float(fields[3])})
        if not words:
            missing.append(str(row["item_id"]))
            continue
        assembled.append(
            {
                "item_id": row["item_id"],
                "session_id": row["session_id"],
                "audio_path": row["audio_path"],
                "duration": row["duration"],
                "transcript": row["transcript"],
                "words": words,
                "metadata": {
                    "source": row["source"],
                    "created_at": row["created_at"],
                    "target_app": row.get("target_app"),
                    "teacher": row["teacher"],
                    "vad": row["vad"],
                    "ctm": str(ctm),
                },
            }
        )
    if missing:
        raise FileNotFoundError(f"missing word alignment for {len(missing)} rows: {missing[:5]}")
    output = freeze / "review-aligned.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in assembled),
        encoding="utf-8",
    )
    return {"rows": len(assembled), "output": str(output)}


def finalize_review(root: Path, *, manual_review_backup: Path | None = None) -> dict[str, object]:
    """Freeze complete transcript-review decisions into the product-output gold manifest."""
    root = root.expanduser().resolve()
    freeze = root / "review/frozen-v1"
    aligned_path = freeze / "review-aligned.jsonl"
    reviewer = freeze / "reviewer"
    database = reviewer / "review.sqlite"
    rows = [
        json.loads(line) for line in aligned_path.read_text(encoding="utf-8").splitlines() if line
    ]
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        reviews = {
            str(row["item_id"]): dict(row)
            for row in connection.execute("SELECT * FROM reviews ORDER BY item_id")
        }
        markers: dict[str, list[dict[str, Any]]] = {}
        for marker in connection.execute("SELECT * FROM markers ORDER BY item_id, audio_time, id"):
            markers.setdefault(str(marker["item_id"]), []).append(dict(marker))
    item_ids = {str(row["item_id"]) for row in rows}
    if set(reviews) != item_ids:
        missing = sorted(item_ids - set(reviews))
        extra = sorted(set(reviews) - item_ids)
        raise ValueError(
            f"review is incomplete or mismatched: missing={len(missing)} extra={len(extra)}"
        )
    unexpected_markers = sorted(set(markers) - item_ids)
    if unexpected_markers:
        raise ValueError(f"markers reference unknown items: {unexpected_markers[:5]}")

    manual_review_backup = (
        manual_review_backup.expanduser().resolve()
        if manual_review_backup is not None
        else reviewer / "review.sqlite.pre-bulk-accept-20260717"
    )
    manual_ids: list[str] = []
    if manual_review_backup.is_file():
        with sqlite3.connect(manual_review_backup) as connection:
            manual_ids = sorted(
                str(row[0]) for row in connection.execute("SELECT item_id FROM reviews")
            )
    if not set(manual_ids).issubset(item_ids):
        raise ValueError("manual review backup contains rows outside the frozen sample")

    gold: list[dict[str, Any]] = []
    for row in rows:
        item_id = str(row["item_id"])
        review = reviews[item_id]
        verdict = str(review["verdict"])
        item_markers = markers.get(item_id, [])
        if verdict not in {"accepted", "exact", "issues"}:
            raise ValueError(f"invalid review verdict for {item_id}: {verdict}")
        if verdict in {"accepted", "exact"} and item_markers:
            raise ValueError(f"accepted item retains issue markers: {item_id}")
        correction = str(review.get("correction") or "").strip()
        if verdict == "issues" and not correction:
            raise ValueError(f"issue row has no corrected product output: {item_id}")
        text = correction if verdict == "issues" else str(row["transcript"]).strip()
        gold.append(
            {
                "audio_filepath": str(row["audio_path"]),
                "text": text,
                "duration": float(row["duration"]),
                "item_id": item_id,
                "session_id": str(row["session_id"]),
                "review": {
                    "surface": "ideal-pasted-dictation-v1",
                    "verdict": "accepted" if verdict == "exact" else verdict,
                    "reviewed_at": float(review["reviewed_at"]),
                    "manually_spot_checked": item_id in manual_ids,
                    "markers": item_markers,
                },
                "metadata": row.get("metadata", {}),
            }
        )

    output = freeze / "gold-v1"
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in gold),
        encoding="utf-8",
    )
    decision = {
        "version": REVIEW_GOLD_VERSION,
        "surface": "ideal-pasted-dictation-v1",
        "basis": (
            "User accepted the complete ARK output as product-ready gold after a bounded manual "
            "spot check; this is not a claim of verbatim human transcription."
        ),
        "rows": len(gold),
        "hours": sum(float(row["duration"]) for row in gold) / 3600,
        "manual_spot_check_rows": len(manual_ids),
        "bulk_accepted_rows": len(gold) - len(manual_ids),
        "aligned_manifest": str(aligned_path),
        "aligned_sha256": _sha256(aligned_path),
        "review_database": str(database),
        "manual_review_backup": str(manual_review_backup),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "teacher_model": "AutoArk-AI/ARK-ASR-3B",
        "teacher_revision": "1e28271b79edc97635783bea65abc89195a09ed3",
    }
    decision_path = output / "decision.json"
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**decision, "decision": str(decision_path)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _dev_session_ids(rows: list[dict[str, Any]], *, fraction: float, seed: int) -> set[str]:
    if not 0 < fraction < 1:
        raise ValueError("dev fraction must be between zero and one")
    durations: dict[str, float] = {}
    for row in rows:
        session_id = str(row["session_id"])
        durations[session_id] = durations.get(session_id, 0.0) + float(row["duration"])
    ranked = sorted(
        durations,
        key=lambda session_id: hashlib.sha256(f"{seed}:{session_id}".encode()).hexdigest(),
    )
    target = sum(durations.values()) * fraction
    selected: set[str] = set()
    selected_duration = 0.0
    for session_id in ranked:
        if selected and selected_duration >= target:
            break
        selected.add(session_id)
        selected_duration += durations[session_id]
    if selected == set(durations) and len(selected) > 1:
        selected.remove(ranked[-1])
    return selected


def _segment_owned_session(  # noqa: PLR0913
    session: dict[str, Any],
    *,
    sequence: int,
    output: Path,
    scratch: Path,
    policy: VadPolicy,
    worker_state: local,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    engine = getattr(worker_state, "vad_engine", None)
    if engine is None:
        engine = load_vad_engine(policy, device="cpu")
        worker_state.vad_engine = engine
    session_id = str(session["session_id"])
    canonical = scratch / f"{session_id}.flac"
    try:
        to_16k_flac(Path(str(session["audio_path"])), canonical)
        audio = load_16k_mono(canonical)
        result = segment_audio_with(engine, audio, policy=policy)
        vad_metadata = segmentation_metadata(policy, engine, result)
        session_rows: list[dict[str, Any]] = []
        for position, (start, end) in enumerate(result.intervals):
            item_id = f"{session_id}-{position:04d}"
            destination = output / "segments" / f"{item_id}.flac"
            write_clip_16k(audio, destination, start, end)
            session_rows.append(
                {
                    "item_id": item_id,
                    "session_id": session_id,
                    "source": session["source"],
                    "source_audio_path": session["audio_path"],
                    "audio_filepath": str(destination),
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "duration": round(end - start, 6),
                    "created_at": session["created_at"],
                    "target_app": session.get("target_app"),
                    "vad": vad_metadata,
                }
            )
        session_record = {
            "session_id": session_id,
            "sequence": sequence,
            "source_duration": float(session["duration"]),
            "segments": len(session_rows),
            "emitted_seconds": sum(float(row["duration"]) for row in session_rows),
        }
        return sequence, session_rows, session_record
    finally:
        canonical.unlink(missing_ok=True)


def prepare_training_derivative(  # noqa: C901, PLR0912, PLR0913, PLR0915
    root: Path,
    *,
    name: str,
    target_hours: float,
    dev_fraction: float,
    seed: int,
    workers: int,
    overwrite: bool = False,
) -> dict[str, object]:
    """Segment non-held-out owned sessions into a resumable, session-disjoint derivative."""
    if target_hours <= 0:
        raise ValueError("target hours must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    root = root.expanduser().resolve()
    output = root / "training" / name
    marker = output / ".english-dictation-training-derivative"
    config = {
        "version": TRAINING_DERIVATIVE_VERSION,
        "name": name,
        "target_hours": target_hours,
        "dev_fraction": dev_fraction,
        "seed": seed,
        "vad_profile": "silero-conservative-v1",
    }
    if output.exists() and overwrite:
        if not marker.is_file():
            raise ValueError(f"refusing to replace unmarked directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "segments").mkdir(exist_ok=True)
    scratch = output / "scratch"
    scratch.mkdir(exist_ok=True)
    if marker.is_file():
        existing_config = json.loads(marker.read_text(encoding="utf-8"))
        if existing_config != config:
            raise ValueError(f"existing derivative config does not match: {output}")
    else:
        marker.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    ledger_path = root / "ledgers/sessions.jsonl"
    ledger = _load_jsonl(ledger_path)
    heldout_path = root / "review/frozen-v1/heldout_sessions.jsonl"
    heldout = {str(row["session_id"]) for row in _load_jsonl(heldout_path)}
    eligible = [
        row
        for row in ledger
        if not row.get("duplicate")
        and row.get("privacy_state") != "exclude"
        and str(row["session_id"]) not in heldout
    ]
    rng = random.Random(seed)  # noqa: S311 - reproducible derivative order
    rng.shuffle(eligible)

    segments_path = output / "segments.jsonl"
    session_state_path = output / "session-state.jsonl"
    segments = _load_jsonl(segments_path)
    session_state = _load_jsonl(session_state_path)
    processed = {str(row["session_id"]) for row in session_state}
    emitted_seconds = sum(float(row["duration"]) for row in segments)
    target_seconds = target_hours * 3600
    policy = build_vad_policy(
        engine="silero",
        profile="conservative-v1",
        max_speech_s=30,
        threshold=0.5,
        silero_backend="onnx",
    )
    worker_state = local()
    try:
        remaining = [
            (sequence, session)
            for sequence, session in enumerate(eligible)
            if str(session["session_id"]) not in processed
        ]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for offset in range(0, len(remaining), workers):
                if emitted_seconds >= target_seconds:
                    break
                batch = remaining[offset : offset + workers]
                futures = [
                    executor.submit(
                        _segment_owned_session,
                        session,
                        sequence=sequence,
                        output=output,
                        scratch=scratch,
                        policy=policy,
                        worker_state=worker_state,
                    )
                    for sequence, session in batch
                ]
                completed = sorted((future.result() for future in futures), key=lambda row: row[0])
                previous_count = len(session_state)
                for _, session_rows, session_record in completed:
                    segments.extend(session_rows)
                    emitted_seconds += float(session_record["emitted_seconds"])
                    session_state.append(session_record)
                    _append_jsonl(segments_path, session_rows)
                    _append_jsonl(session_state_path, [session_record])
                if len(session_state) // 25 != previous_count // 25:
                    print(
                        f"segmented sessions={len(session_state)} rows={len(segments)} "
                        f"hours={emitted_seconds / 3600:.4f}/{target_hours:.4f}"
                    )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if emitted_seconds < target_seconds:
        raise RuntimeError(
            f"eligible owned data emitted only {emitted_seconds / 3600:.4f} hours, "
            f"below the {target_hours:.4f}-hour target"
        )
    dev_sessions = _dev_session_ids(segments, fraction=dev_fraction, seed=seed)
    split_rows = [
        {**row, "split": "dev" if str(row["session_id"]) in dev_sessions else "train"}
        for row in segments
    ]
    split_path = output / "segments-split.jsonl"
    _write_jsonl(split_path, split_rows)
    summary = {
        **config,
        "processing_workers": workers,
        "ledger": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "heldout_sessions": len(heldout),
        "eligible_sessions": len(eligible),
        "selected_sessions": len({str(row["session_id"]) for row in split_rows}),
        "dev_sessions": len(dev_sessions),
        "rows": len(split_rows),
        "hours": emitted_seconds / 3600,
        "train_rows": sum(row["split"] == "train" for row in split_rows),
        "train_hours": sum(float(row["duration"]) for row in split_rows if row["split"] == "train")
        / 3600,
        "dev_rows": sum(row["split"] == "dev" for row in split_rows),
        "dev_hours": sum(float(row["duration"]) for row in split_rows if row["split"] == "dev")
        / 3600,
        "segments": str(split_path),
        "segments_sha256": _sha256(split_path),
    }
    summary_path = output / "prepare-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**summary, "summary": str(summary_path)}


def label_training_derivative(  # noqa: C901, PLR0913, PLR0915
    root: Path,
    *,
    name: str,
    url: str,
    model: str,
    revision: str,
    timeout: float,
    concurrency: int,
) -> dict[str, object]:
    """Pseudo-label a prepared derivative through the approved OpenAI-compatible teacher."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    root = root.expanduser().resolve()
    derivative = root / "training" / name
    rows = _load_jsonl(derivative / "segments-split.jsonl")
    if not rows:
        raise FileNotFoundError(f"prepared derivative is missing or empty: {derivative}")
    raw_dir = derivative / "teacher/ark/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=url.rstrip("/"), timeout=timeout) as client:
        models_response = client.get("/v1/models")
        service_models = models_response.json() if models_response.is_success else None

        def label_one(row: dict[str, Any]) -> tuple[str, str, str | None]:
            item_id = str(row["item_id"])
            raw_path = raw_dir / f"{item_id}.json"
            if raw_path.is_file():
                previous = json.loads(raw_path.read_text(encoding="utf-8"))
                if previous.get("status") == "ok":
                    return item_id, "cached", None
            audio_path = Path(str(row["audio_filepath"]))
            try:
                with audio_path.open("rb") as audio:
                    response = client.post(
                        "/v1/audio/transcriptions",
                        data={"model": model},
                        files={"file": (audio_path.name, audio, "audio/flac")},
                    )
                response.raise_for_status()
                payload = response.json()
                transcript = _teacher_transcript(payload)
                record = {
                    "status": "ok",
                    "transcript": transcript,
                    "response": payload,
                    "labeled_at": datetime.now(UTC).isoformat(),
                }
                status = "ok"
                error = None
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                error = str(exc)
                record = {
                    "status": "failed",
                    "error": error,
                    "labeled_at": datetime.now(UTC).isoformat(),
                }
                status = "failed"
            temporary = raw_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(raw_path)
            return item_id, status, error

        status_counts: dict[str, int] = {}
        consecutive_failures = 0
        executor = ThreadPoolExecutor(max_workers=concurrency)
        try:
            futures = [executor.submit(label_one, row) for row in rows]
            for position, future in enumerate(as_completed(futures), start=1):
                _, status, error = future.result()
                status_counts[status] = status_counts.get(status, 0) + 1
                if status == "failed":
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                if position % 250 == 0 or position == len(futures):
                    print(f"teacher labels {position}/{len(futures)} {status_counts}")
                if consecutive_failures >= MAX_CONSECUTIVE_LABEL_FAILURES:
                    raise RuntimeError(
                        f"teacher labeling aborted after {MAX_CONSECUTIVE_LABEL_FAILURES} "
                        "consecutive failures; "
                        f"last error: {error}"
                    )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    labeled: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for row in rows:
        item_id = str(row["item_id"])
        raw_path = raw_dir / f"{item_id}.json"
        record = json.loads(raw_path.read_text(encoding="utf-8"))
        if record["status"] != "ok":
            failures.append({"item_id": item_id, "error": str(record.get("error", "failed"))})
            continue
        labeled[item_id] = record
    manifests = derivative / "manifests"
    manifests.mkdir(exist_ok=True)
    output_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for row in rows:
        item_id = str(row["item_id"])
        if item_id not in labeled:
            continue
        record = labeled[item_id]
        output_rows[str(row["split"])].append(
            {
                "audio_filepath": row["audio_filepath"],
                "text": record["transcript"],
                "duration": row["duration"],
                "item_id": item_id,
                "session_id": row["session_id"],
                "teacher": {
                    "model": model,
                    "revision": revision,
                    "url": url,
                    "service_models": service_models,
                    "labeled_at": record["labeled_at"],
                    "raw_response": str(raw_dir / f"{item_id}.json"),
                },
                "source": row["source"],
                "vad": row["vad"],
            }
        )
    train_path = manifests / "train.jsonl"
    dev_path = manifests / "dev.jsonl"
    failures_path = manifests / "failures.jsonl"
    _write_jsonl(train_path, output_rows["train"])
    _write_jsonl(dev_path, output_rows["dev"])
    _write_jsonl(failures_path, failures)
    summary = {
        "version": TRAINING_DERIVATIVE_VERSION,
        "name": name,
        "teacher_model": model,
        "teacher_revision": revision,
        "teacher_url": url,
        "service_models": service_models,
        "concurrency": concurrency,
        "input_rows": len(rows),
        "labeled_rows": len(labeled),
        "failures": len(failures),
        "train_rows": len(output_rows["train"]),
        "dev_rows": len(output_rows["dev"]),
        "train_manifest": str(train_path),
        "train_sha256": _sha256(train_path),
        "dev_manifest": str(dev_path),
        "dev_sha256": _sha256(dev_path),
        "failures_manifest": str(failures_path),
    }
    summary_path = derivative / "label-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**summary, "summary": str(summary_path)}


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="Owned English dictation data operations")
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--root", type=Path, default=DEFAULT_IMPORT)
    inventory.add_argument("--output", type=Path)
    freeze = sub.add_parser("freeze-review")
    freeze.add_argument("--root", type=Path, default=DEFAULT_IMPORT)
    freeze.add_argument("--sessions", type=int, default=60)
    freeze.add_argument("--segments", type=int, default=200)
    freeze.add_argument("--seed", type=int, default=20260716)
    freeze.add_argument("--overwrite", action="store_true")
    label = sub.add_parser("label-review")
    label.add_argument("--root", type=Path, default=DEFAULT_IMPORT)
    label.add_argument("--url", default="http://127.0.0.1:8014")
    label.add_argument("--model", default="AutoArk-AI/ARK-ASR-3B")
    label.add_argument("--revision", default="1e28271b79edc97635783bea65abc89195a09ed3")
    label.add_argument("--timeout", type=float, default=180)
    assemble = sub.add_parser("assemble-review")
    assemble.add_argument("--root", type=Path, default=DEFAULT_IMPORT)
    assemble.add_argument("--ctm-dir", type=Path, required=True)
    finalize = sub.add_parser("finalize-review")
    finalize.add_argument("--root", type=Path, default=DEFAULT_IMPORT)
    finalize.add_argument("--manual-review-backup", type=Path)
    prepare_training = sub.add_parser("prepare-training")
    prepare_training.add_argument("--root", type=Path, default=DEFAULT_IMPORT)
    prepare_training.add_argument("--name", default=DEFAULT_TRAINING_DERIVATIVE)
    prepare_training.add_argument("--hours", type=float, default=25)
    prepare_training.add_argument("--dev-fraction", type=float, default=0.05)
    prepare_training.add_argument("--seed", type=int, default=20260717)
    prepare_training.add_argument("--workers", type=int, default=4)
    prepare_training.add_argument("--overwrite", action="store_true")
    label_training = sub.add_parser("label-training")
    label_training.add_argument("--root", type=Path, default=DEFAULT_IMPORT)
    label_training.add_argument("--name", default=DEFAULT_TRAINING_DERIVATIVE)
    label_training.add_argument("--url", default="http://127.0.0.1:8014")
    label_training.add_argument("--model", default="AutoArk-AI/ARK-ASR-3B")
    label_training.add_argument("--revision", default="1e28271b79edc97635783bea65abc89195a09ed3")
    label_training.add_argument("--timeout", type=float, default=180)
    label_training.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Concurrent ARK requests; serialized inference avoids a vLLM variable-audio "
            "batching crash."
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "inventory":
        print(json.dumps(build_ledger(args.root, output=args.output), indent=2, sort_keys=True))
    elif args.command == "freeze-review":
        result = freeze_review(
            args.root,
            session_count=args.sessions,
            segment_count=args.segments,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "label-review":
        result = label_review(
            args.root,
            url=args.url,
            model=args.model,
            revision=args.revision,
            timeout=args.timeout,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "assemble-review":
        print(json.dumps(assemble_review(args.root, ctm_dir=args.ctm_dir), indent=2))
    elif args.command == "finalize-review":
        print(
            json.dumps(
                finalize_review(args.root, manual_review_backup=args.manual_review_backup),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "prepare-training":
        print(
            json.dumps(
                prepare_training_derivative(
                    args.root,
                    name=args.name,
                    target_hours=args.hours,
                    dev_fraction=args.dev_fraction,
                    seed=args.seed,
                    workers=args.workers,
                    overwrite=args.overwrite,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "label-training":
        print(
            json.dumps(
                label_training_derivative(
                    args.root,
                    name=args.name,
                    url=args.url,
                    model=args.model,
                    revision=args.revision,
                    timeout=args.timeout,
                    concurrency=args.concurrency,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
