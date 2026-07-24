"""Downloader completeness and resume tests."""

from __future__ import annotations

import io
from email.message import Message
from typing import TYPE_CHECKING, Self

import pytest

from cv26.mdc import download

if TYPE_CHECKING:
    from pathlib import Path


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = Message()
        for name, value in headers.items():
            self.headers[name] = value

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def getcode(self) -> int:
        return self.status


def test_download_once_refuses_early_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An EOF before Content-Length leaves only a resumable partial file."""
    response = _Response(b"short", status=200, headers={"Content-Length": "10"})
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *_args, **_kwargs: response)
    destination = tmp_path / "archive.tar.gz"

    with pytest.raises(download.IncompleteDownloadError, match="5 of 10"):
        download._download_once("https://example.invalid/archive", destination)  # noqa: SLF001

    assert not destination.exists()
    assert (tmp_path / ".archive.tar.gz.part").read_bytes() == b"short"


def test_download_once_resumes_to_declared_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ranged response appends to the partial and promotes only at the total size."""
    destination = tmp_path / "archive.tar.gz"
    partial = tmp_path / ".archive.tar.gz.part"
    partial.write_bytes(b"first")
    response = _Response(
        b"second",
        status=206,
        headers={"Content-Length": "6", "Content-Range": "bytes 5-10/11"},
    )
    seen: dict[str, object] = {}

    def urlopen(request: object, **_kwargs: object) -> _Response:
        seen["range"] = request.headers.get("Range")
        return response

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)

    size = download._download_once("https://example.invalid/archive", destination)  # noqa: SLF001

    assert size == 11
    assert seen["range"] == "bytes=5-"
    assert destination.read_bytes() == b"firstsecond"
    assert not partial.exists()
