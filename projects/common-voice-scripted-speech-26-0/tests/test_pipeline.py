"""Pipeline state/delete-guard tests against a hand-written HF API double (no mock library)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from cv26 import pipeline
from cv26.convert import shard_path
from cv26.pipeline import RunConfig, load_state, process_one, select_candidates

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from huggingface_hub import HfApi

    from cv26.manifest import ManifestRow


class FakeApi:
    """Records uploaded files and reflects them back as repo siblings, like the Hub would."""

    def __init__(self) -> None:
        """Start with an empty repo."""
        self.files: dict[str, bytes] = {}

    def upload_file(self, *, path_or_fileobj: str, path_in_repo: str, **_: object) -> None:
        """Record an uploaded file's bytes under its repo path."""
        from pathlib import Path

        self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()

    def repo_info(self, **_: object) -> SimpleNamespace:
        """Reflect the recorded uploads back as repo siblings."""
        siblings = [SimpleNamespace(rfilename=name) for name in self.files]
        return SimpleNamespace(siblings=siblings, private=False)


def _cfg(tmp_path: Path, *, upload: bool = False, delete_after_upload: bool = False) -> RunConfig:
    return RunConfig(
        archive_dir=tmp_path / "archives",
        parquet_root=tmp_path / "out",
        state_path=tmp_path / "state.jsonl",
        repo_id="Peacockery/test",
        upload=upload,
        delete_after_upload=delete_after_upload,
    )


def _last_status(state_path: Path) -> str:
    line = state_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    return str(json.loads(line)["status"])


def test_upload_verify_delete_marks_complete(
    tmp_path: Path,
    make_archive: Callable[..., None],
    cv_row: Callable[..., ManifestRow],
) -> None:
    """A normal archive uploads, verifies, deletes the raw file, and ends ``complete``."""
    cfg = _cfg(tmp_path, upload=True, delete_after_upload=True)
    cfg.archive_dir.mkdir()
    row = cv_row()
    archive = cfg.archive_dir / row.filename
    make_archive(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text("x  cv\n", encoding="utf-8")
    api = FakeApi()

    process_one(row, cfg, cast("HfApi", api))

    assert not archive.exists()  # deleted only after verified upload
    assert not archive.with_suffix(archive.suffix + ".sha256").exists()
    assert not shard_path(cfg.parquet_root, row, "train").exists()  # local shard freed too
    assert len(api.files) == 1  # but it IS on the Hub
    assert _last_status(cfg.state_path) == "complete"
    assert load_state(cfg.state_path)[row.dataset_id] == "complete"


def test_upload_without_delete_keeps_local_shards(
    tmp_path: Path,
    make_archive: Callable[..., None],
    cv_row: Callable[..., ManifestRow],
) -> None:
    """Uploading without --delete-after-upload leaves the local archive and shards in place."""
    cfg = _cfg(tmp_path, upload=True)
    cfg.archive_dir.mkdir()
    row = cv_row()
    archive = cfg.archive_dir / row.filename
    make_archive(archive)
    api = FakeApi()

    process_one(row, cfg, cast("HfApi", api))

    assert archive.exists()
    assert shard_path(cfg.parquet_root, row, "train").exists()
    assert _last_status(cfg.state_path) == "complete"


def test_zero_shard_archive_is_never_deleted(
    tmp_path: Path,
    make_archive: Callable[..., None],
    cv_row: Callable[..., ManifestRow],
) -> None:
    """An archive that yields no shards is kept on disk and nothing uploads (the Critical bug)."""
    cfg = _cfg(tmp_path, upload=True, delete_after_upload=True)
    cfg.archive_dir.mkdir()
    row = cv_row()
    archive = cfg.archive_dir / row.filename
    # train.tsv references a clip that is not in the archive -> zero valid records
    make_archive(archive, clips=["one.mp3"], tsv_paths=["ghost.mp3"])
    api = FakeApi()

    process_one(row, cfg, cast("HfApi", api))

    assert archive.exists()  # NOT deleted
    assert api.files == {}  # nothing uploaded
    assert _last_status(cfg.state_path) == "no_shards"
    # terminal: not reprocessed, but also not lost
    assert select_candidates([row], cfg) == []


def test_convert_only_run_stays_a_candidate(
    tmp_path: Path,
    make_archive: Callable[..., None],
    cv_row: Callable[..., ManifestRow],
) -> None:
    """Without --upload the row stops at ``converted`` so a later upload run still picks it up."""
    cfg = _cfg(tmp_path, upload=False)
    cfg.archive_dir.mkdir()
    row = cv_row()
    make_archive(cfg.archive_dir / row.filename)

    process_one(row, cfg, None)

    assert _last_status(cfg.state_path) == "converted"
    assert select_candidates([row], cfg) == [row]


def test_path_guard_rejects_escape(tmp_path: Path) -> None:
    """A path that resolves outside the archive root is refused before any delete."""
    cfg = _cfg(tmp_path)
    outside = tmp_path / "outside.tar.gz"
    with pytest.raises(pipeline.PipelineError):
        pipeline._assert_within(outside, cfg.archive_dir)  # noqa: SLF001
