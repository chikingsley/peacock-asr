"""Persist the exact inputs and runtime used by each training invocation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parakeet_finetune_core.project import ParakeetProject


def fingerprint(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def record_run(project: ParakeetProject, args: Any) -> None:
    if args.dry_run or args.prepare_only:
        return
    paths = [args.validation_manifest, *project.evaluation_manifests]
    paths.extend(
        [Path(item.rsplit("=", 1)[0]) for item in args.train_source]
        if args.train_source
        else [args.train_manifest]
    )
    model = Path(args.model_name)
    if model.is_file():
        paths.append(model)
    if args.tokenizer_dir is not None:
        paths.append(args.tokenizer_dir / "tokenizer.model")
    if project.heldout_manifest is not None:
        paths.append(project.heldout_manifest)
    packages = {}
    for name in ("nemo-toolkit", "torch", "lightning", "lhotse", "parakeet-finetune-core"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    run_dir = args.exp_dir / args.name
    resume_checkpoint = run_dir / "checkpoints" / "last.ckpt"
    if getattr(args, "resume", False) and resume_checkpoint.is_file():
        paths.append(resume_checkpoint)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "configured_at": datetime.now(UTC).isoformat(),
        "arguments": vars(args),
        "packages": packages,
        "inputs": [fingerprint(path) for path in dict.fromkeys(paths)],
    }
    with (run_dir / "run_invocations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
