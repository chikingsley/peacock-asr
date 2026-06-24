"""Hugging Face Hub helpers: create the repo, upload shards, verify they landed.

The target dataset is a private Hub repo; uploads use the configured HF token. ``huggingface_hub``
is imported lazily so the download/convert stages don't pay for it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from huggingface_hub import HfApi

LOGGER = logging.getLogger("cv26.upload")


def hf_api(token: str) -> HfApi:
    """Build an ``HfApi`` client (imported lazily)."""
    import huggingface_hub

    return huggingface_hub.HfApi(token=token)


def ensure_repo(api: HfApi, repo_id: str, *, create: bool) -> None:
    """Confirm the dataset repo exists, optionally creating it private."""
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        if not create:
            raise
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)


def upload_shards(api: HfApi, repo_id: str, root: Path, paths: list[Path]) -> list[str]:
    """Upload each shard as its own commit; return the repo-relative paths."""
    uploaded: list[str] = []
    for path in paths:
        remote = str(path.relative_to(root))
        LOGGER.info("upload: %s -> %s:%s", path.name, repo_id, remote)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add Common Voice parquet shard {remote}",
        )
        uploaded.append(remote)
    return uploaded


def verify_remote(api: HfApi, repo_id: str, remote_paths: list[str]) -> None:
    """Raise if any uploaded path is not visible in the dataset repo."""
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=False)
    siblings = {sibling.rfilename for sibling in getattr(info, "siblings", [])}
    missing = sorted(set(remote_paths) - siblings)
    if missing:
        raise RuntimeError(f"uploaded paths not visible on HF: {', '.join(missing)}")


def upload_file(api: HfApi, repo_id: str, local: Path, remote: str, message: str) -> None:
    """Upload a single metadata file (README, LICENSE, manifest)."""
    LOGGER.info("upload: %s -> %s:%s", local.name, repo_id, remote)
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
    )
