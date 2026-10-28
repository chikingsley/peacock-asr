"""Recover portable v3 audio and its source identities for split-safe Parakeet training."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from omni_curator.process import normalize
from parakeet_finetune_core.materialize import build_parser, materialize
from parakeet_finetune_core.splits import validate_splits

from tajik_asr import DB, LANGUAGE, ROOT
from tajik_asr.parakeet import PROJECT

DATASET_REPO = "Peacockery/tajik-asr-corpus-v3"
DATASET_REVISION = "3b05a4bb89104c21643081250729595347d1188e"

if TYPE_CHECKING:
    from collections.abc import Callable


def verify_snapshot(root: Path, repository: str, revision: str) -> dict[str, object]:
    """Require every pinned Parquet shard and verify its published content hash."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repository, revision=revision, files_metadata=True)
    shards = [item for item in info.siblings if item.rfilename.endswith(".parquet")]
    if not shards:
        raise ValueError("Pinned dataset requires Parquet shards")
    total = 0
    for item in shards:
        path = root / item.rfilename
        if not path.is_file() or path.stat().st_size != item.size:
            raise ValueError(f"Dataset restore requires {item.rfilename}; run with --download")
        if item.lfs is None:
            raise ValueError(f"Dataset shard requires a published SHA-256: {item.rfilename}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != item.lfs.sha256:
            raise ValueError(f"Dataset content hash differs: {item.rfilename}")
        total += item.size
    return {"revision": info.sha, "verified_shards": len(shards), "bytes": total}


def source_index(database: Path) -> dict[tuple[str, str, int], list[str]]:
    """Recover candidate source IDs by corpus, normalized transcript, and sample count."""
    index: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT id,source,text,duration FROM samples WHERE language=?",
            (LANGUAGE,),
        )
        for sample_id, source, text, duration in rows:
            if source == "fleurs" or source.startswith("youtube-"):
                key = (source, normalize(text, LANGUAGE), round(duration * 16_000))
                index[key].append(sample_id)
    return dict(index)


def annotate(
    record: dict[str, object],
    index: dict[tuple[str, str, int], list[str]],
    heldout: frozenset[str],
    split: str,
    counts: Counter[str],
    reserved_samples: frozenset[str] = frozenset(),
    reserved_audio: frozenset[str] = frozenset(),
) -> dict[str, object] | None:
    """Retain every possible source identity and exclude held-out candidates from train."""
    corpus = str(record["corpus"])
    key = (corpus, str(record["text"]), round(float(record["duration"]) * 16_000))
    explicit_ids = record.get("source_sample_ids", [])
    candidates = explicit_ids or index.get(key, [])
    if not candidates:
        counts["unmatched"] += 1
        if split != "train":
            raise ValueError(f"Evaluation row requires a source identity: {key}")
        return None
    videos = record.get("source_video_ids") or (
        sorted({value.rsplit("_", 1)[0] for value in candidates})
        if corpus.startswith("youtube-")
        else []
    )
    if split == "train" and heldout.intersection(videos):
        counts["heldout_video"] += 1
        return None
    if split == "train" and (
        reserved_samples.intersection(candidates) or record.get("audio_sha256") in reserved_audio
    ):
        counts["evaluation_identity"] += 1
        return None
    counts["unique_match" if len(candidates) == 1 else "ambiguous_match"] += 1
    return {
        **record,
        "source_sample_ids": sorted(candidates),
        "source_video_ids": videos,
        "source_match": "exported_identity"
        if explicit_ids
        else "corpus_normalized_text_sample_count",
        "label_origin": "scribe_ensemble_llm_fusion" if videos else "fleurs_reference",
        "license": "unknown" if videos else "CC-BY-4.0",
    }


def _annotate_and_reserve(
    record: dict[str, object],
    annotation: Callable[[dict[str, object]], dict[str, object] | None],
    split: str,
    heldout: set[str],
    reserved_samples: set[str],
    reserved_audio: set[str],
) -> dict[str, object] | None:
    result = annotation(record)
    if result is not None and split != "train":
        reserved_samples.update(result["source_sample_ids"])
        reserved_audio.add(result["audio_sha256"])
        heldout.update(result["source_video_ids"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(PROJECT)
    parser.set_defaults(dataset_repo=DATASET_REPO, dataset_revision=DATASET_REVISION)
    parser.add_argument("--all-splits", action="store_true")
    parser.add_argument("--source-database", type=Path, default=DB)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            args.dataset_repo,
            repo_type="dataset",
            revision=args.dataset_revision,
            local_dir=args.dataset_root.parent,
            max_workers=2,
        )
    inventory = verify_snapshot(args.dataset_root.parent, args.dataset_repo, args.dataset_revision)
    print(f"Verified pinned dataset: {inventory}", flush=True)
    index = source_index(args.source_database)
    heldout_path = ROOT / "manifests" / "heldout_test_videos.json"
    heldout = set(json.loads(heldout_path.read_text())["video_ids"])
    reserved_samples: set[str] = set()
    reserved_audio: set[str] = set()
    report = {
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.dataset_revision,
        "dataset_inventory": inventory,
        "heldout_manifest_sha256": hashlib.sha256(heldout_path.read_bytes()).hexdigest(),
        "source_database": str(args.source_database.resolve()),
        "source_identity_method": "candidate match by corpus, normalized text, and sample count",
        "splits": {},
    }
    train_max_duration = args.max_duration
    train_min_duration = args.min_duration
    for split in ["dev", "test", "train"] if args.all_splits else [args.split]:
        args.split = split
        # Preserve the published v3 benchmark; training has its own duration cap.
        args.max_duration = train_max_duration if split == "train" else 40.0
        args.min_duration = train_min_duration if split == "train" else 0.0
        counts: Counter[str] = Counter()
        if split == "train" and not args.all_splits:
            for evaluation_split in ("dev", "test"):
                path = args.output_root / "manifests" / f"{evaluation_split}.jsonl"
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    reserved_samples.update(row["source_sample_ids"])
                    reserved_audio.add(row["audio_sha256"])
                    heldout.update(row["source_video_ids"])
        annotation = partial(
            annotate,
            index=index,
            heldout=frozenset(heldout),
            split=split,
            counts=counts,
            reserved_samples=frozenset(reserved_samples),
            reserved_audio=frozenset(reserved_audio),
        )

        transform = partial(
            _annotate_and_reserve,
            annotation=annotation,
            split=split,
            heldout=heldout,
            reserved_samples=reserved_samples,
            reserved_audio=reserved_audio,
        )
        stats, manifest = materialize(args, record_transform=transform)
        report["splits"][split] = {
            "rows": stats.rows,
            "hours": stats.hours,
            "source_matches": dict(counts),
            "excluded_duration": stats.excluded_duration,
            "excluded_provenance": stats.excluded_provenance,
            "manifest": str(manifest),
        }
        print(
            f"{split}: rows={stats.rows}, hours={stats.hours:.3f}, identities={dict(counts)}",
            flush=True,
        )
    if not args.dry_run:
        root = args.output_root / "manifests"
        report_path = args.output_root / "source-audit.json"
        report["status"] = "prepared"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if args.all_splits:
            try:
                report["preflight"] = validate_splits(
                    [root / "train.jsonl"],
                    [root / "dev.jsonl", root / "test.jsonl"],
                    heldout_videos=frozenset(heldout),
                    require_video_ids=True,
                )
            except ValueError as error:
                report.update(status="preflight_failed", error=str(error))
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                raise
        report["status"] = "complete"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0
