"""Helpers for uploading run checkpoints to Hugging Face Hub."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def upload_checkpoint_folder(
    *,
    folder_path: Path,
    repo_id: str,
    path_in_repo: str,
    token: str | None,
    private_repo: bool = True,
) -> str:
    """Upload a local checkpoint folder into a Hugging Face model repo."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private_repo,
        exist_ok=True,
    )
    commit_info = api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=f"Add checkpoint folder {path_in_repo}",
    )
    return str(commit_info)
