"""YouTube as a create source: download a video's audio as 16 kHz mono FLAC.

create generates the labels itself, so we only need clean audio. yt-dlp needs a JS runtime for
YouTube's signature challenge — we use Deno (sandboxed). Needs the ``youtube`` extra (yt-dlp),
``deno`` installed, and ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Channel:
    """A vetted channel in a project's source registry: where to pull + how clean to expect it."""

    slug: str
    url: str
    tier: str  # "clean" = scripted/single-speaker | "noisy" = conversational
    note: str


def channel(slug: str, ident: str, tier: str, note: str) -> Channel:
    """Build a :class:`Channel`; ``ident`` is a full URL, an ``@handle``, or a ``UC...`` id."""
    if ident.startswith(("http://", "https://")):
        url = ident
    elif ident.startswith("@"):
        url = f"https://www.youtube.com/{ident}"
    else:
        url = f"https://www.youtube.com/channel/{ident}"
    return Channel(slug, url, tier, note)


@dataclass
class YoutubeAudio:
    video_id: str
    audio_path: Path
    title: str
    url: str


#: Containerized yt-dlp for VPN lanes: runs inside a gluetun container's network namespace so the
#: download egresses through the VPN's clean IP instead of the host (which YouTube may bot-block).
#: This image already ships ``deno`` (for the YouTube JS challenge), ``ffmpeg``, and ``python``, so
#: the exact same deno-bundled yt-dlp args run unchanged inside it. The image's entrypoint *is*
#: ``yt-dlp``, so the in-container argv is the bare ``--remote-components ...`` flags (no
#: ``python -m yt_dlp`` prefix). ``deno`` lives at ``/usr/bin/deno`` in the image.
YTDLP_LANE_IMAGE = "jauderho/yt-dlp:latest"
_LANE_DENO = "/usr/bin/deno"


def _ytdlp_yt_args() -> list[str]:
    """The deno-bundled yt-dlp flags (no program prefix), for the in-container invocation."""
    return [
        "--remote-components", "ejs:github",
        "--js-runtimes", f"deno:{_LANE_DENO}",
    ]


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


def _lane_docker_prefix(lane: str, out_dir: Path, cookies: Path | None) -> list[str]:
    """``docker run`` prefix that runs yt-dlp inside gluetun container ``lane``'s netns.

    The container borrows the VPN container's network namespace (``--network=container:<lane>``) so
    the download egresses through the VPN's clean IP. The output dir and cookies are bind-mounted
    at their **resolved real paths** (the project data dir is a symlink into ``/mnt/overflow``; only
    the real target is mounted), and the yt-dlp flags are built against those same paths — so
    ``--download-archive``, ``-o``, and ``--cookies`` resolve identically in and out of the
    container, and files / the archive / ``<id>.flac`` naming land in the same place as a host run.
    ``--user`` + a writable ``HOME`` (pointed at the mounted out_dir) make the output files
    ``simon:simon`` on ``/mnt/overflow`` and give deno a place for its cache. Cookies are mounted
    read-write because yt-dlp rewrites the jar on exit.
    """
    real_out = out_dir.resolve()
    prefix = [
        "docker", "run", "--rm",
        f"--network=container:{lane}",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"HOME={real_out}",
        "-v", f"{real_out}:{real_out}",
    ]
    if cookies is not None:
        real_cookies = cookies.resolve()
        prefix += ["-v", f"{real_cookies}:{real_cookies}"]
    prefix.append(YTDLP_LANE_IMAGE)
    return prefix


def download_channel(
    channel_url: str,
    *,
    out_dir: Path,
    limit: int | None = None,
    sleep: float = 1.0,
    cookies: Path | None = None,
    lane: str | None = None,
) -> ChannelDownload:
    """Download a channel's videos as 16 kHz mono FLAC into ``out_dir`` (resumable, skip-existing).

    A yt-dlp ``--download-archive`` records what's been fetched, so re-running only pulls new
    videos; per-video failures (private/region-locked/removed) are skipped via ``--ignore-errors``
    rather than aborting the channel. ``limit`` caps to the first N videos. ``sleep`` seconds
    between requests keeps YouTube from rate-limiting / bot-blocking the session (too many fast
    parallel requests trigger "Sign in to confirm you're not a bot"). ``cookies`` is an optional
    Netscape cookies.txt (e.g. exported from a logged-in browser) — yt-dlp's official fix for the
    bot-check; passed via ``--cookies`` when supplied. ``lane`` is an optional gluetun container
    name (e.g. ``gluetun-lane1``): when given, yt-dlp runs inside that container's network namespace
    so the download egresses through a clean VPN IP instead of the (possibly bot-blocked) host IP;
    when absent, the download runs on the host exactly as before. Returns the file count and a
    header-only duration tally so a caller can see how many hours landed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # In a lane run only the *resolved* real paths are bind-mounted into the container (the project
    # data dir is a symlink into /mnt/overflow), so build the file-path flags against those. A host
    # run uses the paths as given — behaviorally identical, since the symlink resolves locally.
    flag_dir = out_dir.resolve() if lane is not None else out_dir
    flag_cookies = (
        cookies.resolve() if (lane is not None and cookies is not None) else cookies
    )
    # yt-dlp flags, identical for host and lane runs; only the *program prefix* differs.
    yt_flags = [
        "--download-archive", str(flag_dir / "downloaded.txt"),
        "--ignore-errors",
        # Throttle so YouTube doesn't rate-limit / bot-block us — the two official sleep knobs:
        # --sleep-requests delays metadata extraction, --sleep-interval/--max delays each download.
        "--sleep-requests", str(sleep),
        "--sleep-interval", str(sleep), "--max-sleep-interval", str(round(sleep * 3, 1)),
        "--retries", "10", "--extractor-retries", "5",
        *_AUDIO_TO_16K_FLAC,
        "-o", str(flag_dir / "%(id)s.%(ext)s"),
    ]
    if flag_cookies is not None:
        yt_flags += ["--cookies", str(flag_cookies)]
    if limit is not None:
        yt_flags += ["--playlist-end", str(limit)]
    yt_flags.append(channel_url)

    if lane is None:  # host run — unchanged behavior
        cmd = [*_ytdlp_base(), *yt_flags]
    else:  # VPN lane — same deno-bundled yt-dlp args, but inside the container's netns
        cmd = [*_lane_docker_prefix(lane, out_dir, cookies), *_ytdlp_yt_args(), *yt_flags]
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


def refresh_cookies_from_browser(profile_dir: Path, out_path: Path) -> int:
    """Export fresh YouTube cookies from a local logged-in Chrome profile to ``out_path``.

    The durable fix for YouTube's bot-check: a real Chrome profile (e.g. the persistent home of
    a KasmVNC browser container, logged into YouTube once) lives on this machine, and yt-dlp
    extracts its live cookies on demand — no manual exports, no macOS TCC. Safe to run while
    that browser is up (yt-dlp copies the cookie DB). Returns the youtube.com cookie count;
    raises if extraction produced none (profile logged out / wrong profile dir).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--cookies-from-browser", f"chrome:{profile_dir}",
        "--cookies", str(out_path),
        "--skip-download", "--no-warnings", "--quiet",
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # any stable public video works
    ]
    subprocess.run(cmd, check=False)  # noqa: S603 — fixed argv, no shell
    if not out_path.exists():
        msg = f"yt-dlp wrote no cookie file at {out_path}"
        raise RuntimeError(msg)
    count = sum(
        1 for line in out_path.read_text(encoding="utf-8").splitlines()
        if "youtube.com" in line and not line.startswith("#")
    )
    if count == 0:
        msg = f"no youtube.com cookies extracted from {profile_dir} — is the profile logged in?"
        raise RuntimeError(msg)
    return count
