from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("expected object JSON from yt-dlp")
    return payload


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ytdlp_base() -> list[str]:
    # YouTube extraction needs a JS runtime to solve the signature challenge. yt-dlp
    # recommends Deno (sandboxed: the untrusted JS gets no fs/network access); fall back
    # to Node >=22. Bun is deprecated by yt-dlp, so it's intentionally not used.
    command = [sys.executable, "-m", "yt_dlp", "--remote-components", "ejs:github"]
    deno = shutil.which("deno") or str(Path.home() / ".deno" / "bin" / "deno")
    node = shutil.which("node")
    if Path(deno).is_file():
        command.extend(["--js-runtimes", f"deno:{deno}"])
    elif node:
        command.extend(["--js-runtimes", f"node:{node}"])
    return command


def safe_video_url(value: str) -> str:
    if re.fullmatch(r"[-_A-Za-z0-9]{11}", value):
        return f"https://www.youtube.com/watch?v={value}"
    return value


def cut_audio(source_audio: Path, output_audio: Path, start: float, end: float) -> None:
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(source_audio),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "flac",
            str(output_audio),
        ]
    )
