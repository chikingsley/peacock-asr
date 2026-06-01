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


#: yt-dlp args that turn the best audio stream into a 16 kHz mono FLAC (shared by both downloaders).
_AUDIO_TO_16K_FLAC = [
    "-f", "ba/bestaudio",
    "--extract-audio", "--audio-format", "flac", "--audio-quality", "0",
    "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
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
            *_ytdlp_base(), "--no-playlist", *_AUDIO_TO_16K_FLAC,
            "-o", str(out_dir / "%(id)s.%(ext)s"), url,
        ],
        check=True,
    )
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    return YoutubeAudio(video_id, audio_path, str(info.get("title") or ""), url)


@dataclass
class ChannelDownload:
    """The result of pulling a channel's audio: where it landed + how much we got."""

    channel: str
    out_dir: Path
    flac_count: int
    total_seconds: float

    @property
    def hours(self) -> float:
        return self.total_seconds / 3600.0


def list_channel_videos(channel_url: str, *, limit: int | None = None) -> list[str]:
    """List a channel/playlist's video ids (flat, metadata-only — no audio downloaded).

    Fast way to size a channel before committing to the download. ``limit`` takes only the first N.
    """
    cmd = [*_ytdlp_base(), "--flat-playlist", "--print", "id"]
    if limit is not None:
        cmd += ["--playlist-end", str(limit)]
    cmd.append(channel_url)
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def download_channel(
    channel_url: str,
    *,
    out_dir: Path,
    limit: int | None = None,
    sleep: float = 1.0,
    cookies: Path | None = None,
) -> ChannelDownload:
    """Download a channel's videos as 16 kHz mono FLAC into ``out_dir`` (resumable, skip-existing).

    A yt-dlp ``--download-archive`` records what's been fetched, so re-running only pulls new
    videos; per-video failures (private/region-locked/removed) are skipped via ``--ignore-errors``
    rather than aborting the channel. ``limit`` caps to the first N videos. ``sleep`` seconds
    between requests keeps YouTube from rate-limiting / bot-blocking the session (too many fast
    parallel requests trigger "Sign in to confirm you're not a bot"). ``cookies`` is an optional
    Netscape cookies.txt (e.g. exported from a logged-in browser) — yt-dlp's official fix for the
    bot-check; passed via ``--cookies`` when supplied. Returns the file count and a header-only
    duration tally so a caller can see how many hours landed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *_ytdlp_base(),
        "--download-archive", str(out_dir / "downloaded.txt"),
        "--ignore-errors",
        # Throttle so YouTube doesn't rate-limit / bot-block us — the two official sleep knobs:
        # --sleep-requests delays metadata extraction, --sleep-interval/--max delays each download.
        "--sleep-requests", str(sleep),
        "--sleep-interval", str(sleep), "--max-sleep-interval", str(round(sleep * 3, 1)),
        "--retries", "10", "--extractor-retries", "5",
        *_AUDIO_TO_16K_FLAC,
        "-o", str(out_dir / "%(id)s.%(ext)s"),
    ]
    if cookies is not None:
        cmd += ["--cookies", str(cookies)]
    if limit is not None:
        cmd += ["--playlist-end", str(limit)]
    cmd.append(channel_url)
    # --ignore-errors makes yt-dlp continue past bad videos and can exit non-zero; don't treat
    # that as fatal — the per-file results on disk are the source of truth.
    subprocess.run(cmd, check=False)  # noqa: S603

    import soundfile as sf

    total = 0.0
    flacs = sorted(out_dir.glob("*.flac"))
    for flac in flacs:
        try:
            info = sf.info(str(flac))
        except (RuntimeError, OSError):
            continue
        total += info.frames / info.samplerate
    return ChannelDownload(channel_url, out_dir, len(flacs), total)
