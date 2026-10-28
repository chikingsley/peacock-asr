"""Resolve manifest shards consistently for NeMo loading, preflight, and provenance."""

from pathlib import Path


def manifest_paths(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]
    shards = sorted(path.glob("shard_*.jsonl"))
    if not shards or any(
        item.name != f"shard_{index:05d}.jsonl" for index, item in enumerate(shards)
    ):
        raise ValueError(f"Manifest directory requires contiguous shard_00000.jsonl files: {path}")
    return shards


def manifest_pattern(path: Path) -> str:
    shards = manifest_paths(path)
    if len(shards) == 1:
        return str(shards[0])
    return str(path / f"shard_{{00000..{len(shards) - 1:05d}}}.jsonl")
