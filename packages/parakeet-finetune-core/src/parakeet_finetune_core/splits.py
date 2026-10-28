"""Validate manifest identities before allocating training resources."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from parakeet_finetune_core.manifests import manifest_paths

if TYPE_CHECKING:
    from collections.abc import Iterable

    from parakeet_finetune_core.project import ParakeetProject

_MAX_FAILURE_EXAMPLES = 10


def video_ids(record: dict[str, Any]) -> set[str]:
    """Read explicit identities or recover the historical labeled/channel/video layout."""
    ids = set(record.get("source_video_ids", []))
    for field in ("video_id", "source_video_id"):
        if record.get(field):
            ids.add(str(record[field]))
    parts = Path(record["audio_filepath"]).parts
    if "labeled" in parts:
        index = parts.index("labeled")
        if len(parts) > index + 2:
            ids.add(f"{parts[index + 1]}_{parts[index + 2]}")
    return ids


def _identities(record: dict[str, Any], manifest: Path) -> set[tuple[str, str]]:
    audio = Path(record["audio_filepath"])
    if not audio.is_absolute():
        audio = manifest.parent / audio
    result = {("path", str(audio.resolve()))}
    if record.get("audio_sha256"):
        result.add(("sha256", str(record["audio_sha256"])))
    result.update(("video", value) for value in video_ids(record))
    result.update(("sample", value) for value in record.get("source_sample_ids", []))
    if record.get("sample_id"):
        result.add(("sample", str(record["sample_id"])))
    return result


def _problem(
    record: dict[str, Any],
    identities: set[tuple[str, str]],
    reserved: set[tuple[str, str]],
    *,
    training: bool,
    require_video_ids: bool,
) -> str | None:
    if training and identities & reserved:
        return "training identity overlaps a reserved evaluation identity"
    audio = next(value for kind, value in identities if kind == "path")
    if not Path(audio).is_file():
        return "audio file is unavailable"
    if (
        training
        and require_video_ids
        and str(record.get("corpus", "")).startswith("youtube-")
        and not video_ids(record)
    ):
        return "YouTube training row requires source_video_ids"
    return None


def validate_splits(
    train: Iterable[Path],
    evaluation: Iterable[Path],
    *,
    heldout_videos: frozenset[str] = frozenset(),
    require_video_ids: bool = False,
) -> dict[str, int]:
    """Reject overlap, absent audio, empty manifests, and missing required provenance."""
    reserved = {("video", value) for value in heldout_videos}
    counts = {"train_rows": 0, "evaluation_rows": 0}
    failures: list[str] = []
    problems: Counter[str] = Counter()
    for phase, paths in (("evaluation", evaluation), ("train", train)):
        for path in (shard for item in paths for shard in manifest_paths(item)):
            rows = 0
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    record = json.loads(line)
                    identities = _identities(record, path)
                    problem = _problem(
                        record,
                        identities,
                        reserved,
                        training=phase == "train",
                        require_video_ids=require_video_ids,
                    )
                    if problem:
                        problems[problem] += 1
                    if problem and len(failures) < _MAX_FAILURE_EXAMPLES:
                        failures.append(f"{path}:{line_number}: {problem}")
                    if phase == "evaluation":
                        reserved.update(identities)
                    rows += 1
            if rows == 0:
                failures.append(f"{path}: manifest requires at least one row")
            counts[f"{phase}_rows"] += rows
    if failures:
        raise ValueError(
            f"Training split preflight failed: {dict(problems)}\n" + "\n".join(failures)
        )
    return counts


def preflight(project: ParakeetProject, args: Any) -> None:
    """Apply the project's split contract to each training entry point."""
    if args.dry_run or getattr(args, "prepare_only", False):
        return
    sources = getattr(args, "train_source", [])
    train = [Path(value.rsplit("=", 1)[0]) for value in sources] or [args.train_manifest]
    evaluation = [args.validation_manifest, *project.evaluation_manifests]
    heldout = frozenset()
    if project.heldout_manifest is not None:
        payload = json.loads(project.heldout_manifest.read_text(encoding="utf-8"))
        heldout = frozenset(payload["video_ids"])
    result = validate_splits(
        train,
        evaluation,
        heldout_videos=heldout,
        require_video_ids=project.heldout_manifest is not None,
    )
    print(f"Training split preflight passed: {result}", flush=True)
