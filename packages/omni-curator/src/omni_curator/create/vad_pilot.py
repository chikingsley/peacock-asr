"""Isolated, manifest-bounded comparison of curator VAD policies.

The pilot never opens the production queue and never writes beneath the production clips root.
Its JSONL manifest is the exact selector and its output directory is a self-contained run artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omni_curator.create.vad import (
    VadEngineName,
    build_vad_policy,
    effective_profile_id,
    load_vad_engine,
    segment_audio_with,
)
from omni_curator.process.audio import load_16k_mono, write_clip_16k

SCRIBE_FAILURE_LIMIT = 3
PILOT_MARKER = ".omni-vad-pilot"
PILOT_MARKER_CONTENT = '{"kind":"omni-vad-pilot","version":1}\n'
PILOT_CHILDREN = {
    PILOT_MARKER,
    "clips",
    "clips.jsonl",
    "intervals.jsonl",
    "run.json",
    "scribe.jsonl",
    "artifact-summary.json",
    "historical-scribe-anchor.json",
}
SAFE_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


@dataclass(frozen=True, slots=True)
class PilotSource:
    source_id: str
    path: Path
    tier: str
    channel: str


def read_pilot_manifest(path: Path) -> list[PilotSource]:  # noqa: C901
    """Read a bounded JSONL selector with id/path/tier/channel fields."""
    sources: list[PilotSource] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid pilot manifest JSON on line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"pilot manifest line {line_number} must be a JSON object")
        source_id = str(row.get("id") or "").strip()
        source_path = Path(str(row.get("path") or "")).expanduser().resolve()
        tier = str(row.get("tier") or "").strip()
        channel = str(row.get("channel") or "").strip()
        if not source_id or not tier or not channel or not str(row.get("path") or "").strip():
            raise ValueError(
                f"pilot manifest line {line_number} requires non-empty id/path/tier/channel"
            )
        if not SAFE_SOURCE_ID.fullmatch(source_id) or source_id in {".", ".."}:
            raise ValueError(
                f"pilot source id {source_id!r} is not a safe single path component"
            )
        if source_id in seen:
            raise ValueError(f"duplicate pilot source id: {source_id}")
        if tier not in {"clean", "noisy"}:
            raise ValueError(f"pilot source {source_id} has unsupported tier {tier!r}")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        seen.add(source_id)
        sources.append(PilotSource(source_id, source_path, tier, channel))
    if not sources:
        raise ValueError(f"pilot manifest has no sources: {path}")
    return sources


def run_vad_pilot(
    *,
    manifest: Path,
    output_dir: Path,
    engines: Sequence[VadEngineName],
    profile: str = "conservative-v1",
    max_duration: float = 30.0,
    threshold: float = 0.5,
    model_path: Path | None = None,
    silero_backend: str = "auto",
    device: str = "cpu",
    write_clips: bool = False,
    overwrite: bool = False,
    scribe_max_clips_per_engine: int = 0,
    scribe_model: str = "scribe-v2",
    scribe_language: str | None = None,
) -> dict[str, Any]:
    """Run all policies over the same sources and return/write aggregate metrics."""
    sources = read_pilot_manifest(manifest)
    output_dir = output_dir.expanduser().resolve()
    unique_engines = tuple(dict.fromkeys(engines))
    if not unique_engines:
        raise ValueError("at least one VAD engine is required")
    if scribe_max_clips_per_engine < 0:
        raise ValueError("scribe_max_clips_per_engine cannot be negative")
    _prepare_output_dir(output_dir, overwrite=overwrite)

    records_path = output_dir / "intervals.jsonl"
    clips_path = output_dir / "clips.jsonl"
    asr_path = output_dir / "scribe.jsonl"
    records: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []

    for engine_name in unique_engines:
        engine_started = time.perf_counter()
        policy = build_vad_policy(
            engine=engine_name,
            profile=profile,
            max_speech_s=max_duration,
            threshold=threshold,
            model_path=model_path,
            silero_backend=silero_backend,
        )
        engine = load_vad_engine(policy, device=device)
        effective_id = effective_profile_id(
            policy, engine.model_revision, runtime_metadata=engine.runtime_metadata
        )
        policy_record = {
            **policy.as_dict(),
            "policy_id": policy.profile_id,
            "profile_id": effective_id,
            "model_revision": engine.model_revision,
            "runtime": engine.runtime_metadata,
        }
        policies.append(policy_record)
        try:
            for source in sources:
                audio = load_16k_mono(source.path)
                started = time.perf_counter()
                result = segment_audio_with(engine, audio, policy=policy)
                elapsed = time.perf_counter() - started
                intervals = list(result.intervals)
                durations = [end - start for start, end in intervals]
                record: dict[str, Any] = {
                    "source_id": source.source_id,
                    "path": str(source.path),
                    "tier": source.tier,
                    "channel": source.channel,
                    "engine": engine_name,
                    "profile_id": effective_id,
                    "model_revision": engine.model_revision,
                    "max_duration": max_duration,
                    "audio_seconds": result.audio_seconds,
                    "runtime_seconds": elapsed,
                    "rtfx": result.audio_seconds / elapsed if elapsed else None,
                    "raw_intervals": [list(item) for item in result.raw_intervals],
                    "intervals": [list(item) for item in intervals],
                    "speech_seconds": sum(durations),
                    "coverage": (
                        sum(durations) / result.audio_seconds if result.audio_seconds else 0.0
                    ),
                    "empty": not intervals,
                }
                records.append(record)
                if write_clips or scribe_max_clips_per_engine:
                    clip_rows.extend(
                        _write_pilot_clips(
                            output_dir=output_dir,
                            source=source,
                            engine_name=engine_name,
                            profile_id=effective_id,
                            model_revision=engine.model_revision,
                            audio=audio,
                            intervals=intervals,
                        )
                    )
        finally:
            engine.close()
            policy_record["end_to_end_seconds"] = time.perf_counter() - engine_started

    _write_jsonl(records_path, records)
    if clip_rows:
        _write_jsonl(clips_path, clip_rows)

    asr_rows: list[dict[str, Any]] = []
    if scribe_max_clips_per_engine:
        asr_rows = _scribe_sample(
            clip_rows,
            engines=unique_engines,
            limit=scribe_max_clips_per_engine,
            model=scribe_model,
            language=scribe_language,
        )
        _write_jsonl(asr_path, asr_rows)

    summary: dict[str, Any] = {
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256_file(manifest.resolve()),
        "implementation_sha256": _sha256_files(
            [
                Path(__file__).resolve(),
                Path(__file__).with_name("vad.py").resolve(),
                Path(__file__).with_name("segment.py").resolve(),
            ]
        ),
        "requested_device": device,
        "output_dir": str(output_dir.resolve()),
        "source_count": len(sources),
        "source_ids": [source.source_id for source in sources],
        "tiers": {
            tier: sum(source.tier == tier for source in sources) for tier in ("clean", "noisy")
        },
        "policies": policies,
        "metrics": _aggregate_records(records),
        "artifacts": _aggregate_artifacts(clip_rows),
        "scribe": _aggregate_scribe(asr_rows),
        "production_queue_touched": False,
    }
    (output_dir / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"pilot output is not empty: {output_dir}; pass --overwrite")
        marker = output_dir / PILOT_MARKER
        if not marker.is_file() or marker.read_text(encoding="utf-8") != PILOT_MARKER_CONTENT:
            raise PermissionError(
                f"refusing to overwrite unmarked directory (missing valid {PILOT_MARKER}): "
                f"{output_dir}"
            )
        unexpected = sorted(
            child.name for child in output_dir.iterdir() if child.name not in PILOT_CHILDREN
        )
        if unexpected:
            raise PermissionError(
                f"refusing to overwrite pilot directory with unknown children: {unexpected}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PILOT_MARKER).write_text(PILOT_MARKER_CONTENT, encoding="utf-8")


def _write_pilot_clips(
    *,
    output_dir: Path,
    source: PilotSource,
    engine_name: str,
    profile_id: str,
    model_revision: str,
    audio: np.ndarray,
    intervals: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    clips_root = (output_dir / "clips").resolve()
    clip_dir = (clips_root / engine_name / source.tier / source.source_id).resolve()
    if not clip_dir.is_relative_to(clips_root):
        raise ValueError(f"pilot clip path escaped output root: {clip_dir}")
    clip_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(intervals):
        clip_path = clip_dir / f"seg_{index:04d}.flac"
        write_clip_16k(audio, clip_path, start, end)
        rows.append(
            {
                "clip_id": f"{engine_name}:{source.source_id}:{index:04d}",
                "source_id": source.source_id,
                "tier": source.tier,
                "channel": source.channel,
                "engine": engine_name,
                "profile_id": profile_id,
                "model_revision": model_revision,
                "path": str(clip_path.resolve()),
                "start": start,
                "end": end,
                "duration": end - start,
            }
        )
    return rows


def _scribe_sample(
    clip_rows: Sequence[dict[str, Any]],
    *,
    engines: Sequence[str],
    limit: int,
    model: str,
    language: str | None,
) -> list[dict[str, Any]]:
    from omni_curator.scribe.swservice import transcribe_file

    output: list[dict[str, Any]] = []
    consecutive_failures = 0
    for engine in engines:
        candidates = sorted(
            (row for row in clip_rows if row["engine"] == engine),
            key=lambda row: (str(row["tier"]), str(row["source_id"]), str(row["clip_id"])),
        )
        for row in _balanced_tier_sample(candidates, limit):
            started = time.perf_counter()
            error: str | None = None
            transcript = ""
            try:
                result = transcribe_file(str(row["path"]), asr_model=model, language=language)
                transcript = str(result.get("transcript") or "").strip()
                if result.get("error"):
                    error = str(result["error"])
            except Exception as exc:  # noqa: BLE001 — pilot records each service failure
                error = f"{type(exc).__name__}: {exc}"
                consecutive_failures += 1
            else:
                consecutive_failures = consecutive_failures + 1 if error else 0
            output.append(
                {
                    **row,
                    "scribe_model": model,
                    "scribe_language": language,
                    "transcript": transcript,
                    "transcript_characters": len(transcript),
                    "empty_transcript": not transcript,
                    "error": error,
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
            if consecutive_failures >= SCRIBE_FAILURE_LIMIT:
                return output
    return output


def _balanced_tier_sample(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    pools = {tier: [row for row in rows if row["tier"] == tier] for tier in ("clean", "noisy")}
    while len(chosen) < limit and any(pools.values()):
        for tier in ("clean", "noisy"):
            if pools[tier] and len(chosen) < limit:
                chosen.append(pools[tier].pop(0))
    return chosen


def _aggregate_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((str(record["engine"]), str(record["tier"])), []).append(record)
    output: list[dict[str, Any]] = []
    for (engine, tier), rows in sorted(groups.items()):
        intervals = [(float(end) - float(start)) for row in rows for start, end in row["intervals"]]
        audio_seconds = sum(float(row["audio_seconds"]) for row in rows)
        speech_seconds = sum(float(row["speech_seconds"]) for row in rows)
        runtime = sum(float(row["runtime_seconds"]) for row in rows)
        output.append(
            {
                "engine": engine,
                "tier": tier,
                "sources": len(rows),
                "empty_sources": sum(bool(row["empty"]) for row in rows),
                "audio_seconds": audio_seconds,
                "raw_intervals": sum(len(row["raw_intervals"]) for row in rows),
                "emitted_clips": len(intervals),
                "speech_seconds": speech_seconds,
                "coverage": speech_seconds / audio_seconds if audio_seconds else 0.0,
                "mean_clip_seconds": statistics.fmean(intervals) if intervals else None,
                "median_clip_seconds": statistics.median(intervals) if intervals else None,
                "p95_clip_seconds": _percentile(intervals, 0.95),
                "under_one_second": sum(value < 1.0 for value in intervals),
                "over_hard_cap": sum(
                    value > float(rows[0]["max_duration"]) + 1e-9 for value in intervals
                ),
                "runtime_seconds": runtime,
                "rtfx": audio_seconds / runtime if runtime else None,
            }
        )
    return output


def _aggregate_scribe(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["engine"]), str(row["tier"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (engine, tier), group in sorted(groups.items()):
        valid = [row for row in group if row["error"] is None]
        audio_seconds = sum(float(row["duration"]) for row in valid)
        characters = sum(int(row["transcript_characters"]) for row in valid)
        output.append(
            {
                "engine": engine,
                "tier": tier,
                "attempted": len(group),
                "errors": len(group) - len(valid),
                "empty_transcripts": sum(bool(row["empty_transcript"]) for row in valid),
                "nonempty_yield": (
                    sum(not bool(row["empty_transcript"]) for row in valid) / len(valid)
                    if valid
                    else None
                ),
                "transcript_characters_per_audio_minute": (
                    characters / (audio_seconds / 60) if audio_seconds else None
                ),
            }
        )
    return output


def _aggregate_artifacts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["engine"]), str(row["tier"])), []).append(row)
    return [
        {
            "engine": engine,
            "tier": tier,
            "clips": len(group),
            "bytes": sum(Path(str(row["path"])).stat().st_size for row in group),
        }
        for (engine, tier), group in sorted(groups.items())
    ]


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
