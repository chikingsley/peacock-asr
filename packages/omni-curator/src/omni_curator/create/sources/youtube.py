"""YouTube as a create source: download a video's audio as 16 kHz mono FLAC.

create generates the labels itself, so we only need clean audio. yt-dlp needs a JS runtime for
YouTube's signature challenge — we use Deno (sandboxed). Needs the ``youtube`` extra (yt-dlp),
``deno`` installed, and ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class YoutubeAudio:
    video_id: str
    audio_path: Path
    title: str
    url: str


def _ytdlp_base() -> list[str]:
    deno = shutil.which("deno") or str(Path.home() / ".deno" / "bin" / "deno")
    return [
        sys.executable, "-m", "yt_dlp",
        "--remote-components", "ejs:github",
        "--js-runtimes", f"deno:{deno}",
    ]


def download_audio(url: str, *, out_dir: Path) -> YoutubeAudio:
    """Download a YouTube video's audio as 16 kHz mono FLAC at ``out_dir/<id>.flac``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(  # noqa: S603
        [*_ytdlp_base(), "--skip-download", "--dump-single-json", url],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(probe.stdout)
    video_id = str(info["id"])
    audio_path = out_dir / f"{video_id}.flac"
    subprocess.run(  # noqa: S603
        [
            *_ytdlp_base(), "--no-playlist", "-f", "ba/bestaudio",
            "--extract-audio", "--audio-format", "flac", "--audio-quality", "0",
            "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
            "-o", str(out_dir / "%(id)s.%(ext)s"), url,
        ],
        check=True,
    )
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    return YoutubeAudio(video_id, audio_path, str(info.get("title") or ""), url)
