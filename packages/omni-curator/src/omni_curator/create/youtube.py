"""YouTube as a create source: download a video's audio as 16 kHz mono FLAC.

create generates the labels itself, so we only need clean audio. yt-dlp needs a JS runtime for
YouTube's signature challenge — we use Deno (sandboxed). Needs the ``youtube`` extra (yt-dlp),
``deno`` installed, and ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

#: yt-dlp's exit code when it stops because ``--max-downloads`` was reached (more of the channel
#: remains). Any other code means the playlist was exhausted or a fatal error ended the run.
_YTDLP_MAX_DOWNLOADS_RC = 101


@dataclass(frozen=True)
class Channel:
    """A vetted channel in a project's source registry: where to pull + how clean to expect it."""

    slug: str
    url: str
    tier: str  # "clean" = scripted/single-speaker | "noisy" = conversational
    note: str
    category: str = "uncategorized"


_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "language_learning",
        (
            "language learning",
            "language lessons",
            "phrase pairs",
            "grammar lessons",
            "learning tajik",
        ),
    ),
    (
        "audiobook",
        (
            "audiobook",
            "audiobooks",
            "audio book",
            "audio books",
            "narrated literature",
            "book readings",
        ),
    ),
    (
        "children",
        ("children", "kids", "kid", "alphabet", "riddles", "father/daughter"),
    ),
    (
        "religion",
        (
            "religious",
            "sermon",
            "sermons",
            "quran",
            "tafsir",
            "liturgy",
            "patriarchate",
            "hoji",
        ),
    ),
    (
        "education",
        (
            "education",
            "educational",
            "science",
            "school",
            "university",
            "lecture",
            "lectures",
            "lessons",
            "course",
            "academic",
            "mooc",
            "explainer",
        ),
    ),
    (
        "news",
        (
            "news",
            "bulletin",
            "bulletins",
            "reporting",
            "reports",
            "broadcaster",
            "journalism",
            "rfe/rl",
            "voa",
            "state channel",
        ),
    ),
    ("podcast", ("podcast",)),
    ("interview", ("interview", "interviews", "call-in")),
    (
        "talk",
        (
            "talk",
            "panel",
            "debate",
            "discussion",
            "conversation",
            "conversational",
            "commentary",
            "monologue",
            "analysis",
            "analytics",
            "show",
        ),
    ),
    ("documentary", ("documentary", "docs", "investigative")),
    ("comedy", ("comedy", "sketch", "skit", "sitcom", "humor", "ханда")),
    ("food", ("cooking", "cook", "food")),
    ("travel", ("travel", "around world")),
    ("vlog", ("vlog", "vlogs", "daily-life", "daily life", "lifestyle", "street")),
    ("entertainment", ("entertainment", "variety", "talent", "sports", "football", "film")),
    ("music", ("music", "song", "chant")),
)


def infer_channel_category(slug: str, note: str) -> str:
    """Infer the source-category taxonomy from a channel slug/note when not set explicitly."""
    text = f"{slug} {note}".lower().replace("-", " ").replace("_", " ")
    for category, needles in _CATEGORY_KEYWORDS:
        if any(needle in text for needle in needles):
            return category
    return "uncategorized"


def channel(
    slug: str, ident: str, tier: str, note: str, *, category: str = "uncategorized"
) -> Channel:
    """Build a :class:`Channel`; ``ident`` is a full URL, an ``@handle``, or a ``UC...`` id."""
    if ident.startswith(("http://", "https://")):
        url = ident
    elif ident.startswith("@"):
        url = f"https://www.youtube.com/{ident}"
    else:
        url = f"https://www.youtube.com/channel/{ident}"
    if category == "uncategorized":
        category = infer_channel_category(slug, note)
    return Channel(slug, url, tier, note, category)


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
        "--remote-components",
        "ejs:github",
        "--js-runtimes",
        f"deno:{_LANE_DENO}",
    ]


def _ytdlp_base() -> list[str]:
    deno = shutil.which("deno") or str(Path.home() / ".deno" / "bin" / "deno")
    return [
        sys.executable,
        "-m",
        "yt_dlp",
        "--remote-components",
        "ejs:github",
        "--js-runtimes",
        f"deno:{deno}",
    ]


#: yt-dlp args that turn the best audio stream into a 16 kHz mono FLAC (shared by both downloaders).
_AUDIO_TO_16K_FLAC = [
    "-f",
    "ba/bestaudio",
    "--extract-audio",
    "--audio-format",
    "flac",
    "--audio-quality",
    "0",
    "--postprocessor-args",
    "ffmpeg:-ar 16000 -ac 1",
]


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


@dataclass(frozen=True)
class PrescanResult:
    """One pre-download channel reachability check persisted to the project prescan DB."""

    slug: str
    url: str
    tier: str
    category: str
    lane: str | None
    status: str
    video_count: int
    last_error: str | None


_PRESCAN_SCHEMA = """
CREATE TABLE IF NOT EXISTS channel_prescan (
    slug        TEXT PRIMARY KEY,
    url         TEXT NOT NULL,
    tier        TEXT NOT NULL,
    category    TEXT NOT NULL,
    lane        TEXT,
    status      TEXT NOT NULL,
    video_count INTEGER NOT NULL,
    checked_at  REAL NOT NULL,
    last_error  TEXT
);
"""


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


def prescan_channels(
    channels: list[Channel],
    *,
    db_path: Path,
    limit: int | None = 1,
    lane: str | None = None,
    list_videos: Callable[..., list[str]] | None = None,
) -> list[PrescanResult]:
    """Record a reachability check for each channel before spending download bandwidth."""
    from omni_curator.data.store import connect_wal

    lister = list_videos or list_channel_videos
    conn = connect_wal(db_path, _PRESCAN_SCHEMA)
    checked_at = time.time()
    results: list[PrescanResult] = []
    with conn:
        for ch in channels:
            try:
                ids = lister(ch.url, limit=limit)
            except subprocess.CalledProcessError as exc:
                result = PrescanResult(
                    slug=ch.slug,
                    url=ch.url,
                    tier=ch.tier,
                    category=ch.category,
                    lane=lane,
                    status="error",
                    video_count=0,
                    last_error=(exc.stderr or str(exc))[:500],
                )
            else:
                result = PrescanResult(
                    slug=ch.slug,
                    url=ch.url,
                    tier=ch.tier,
                    category=ch.category,
                    lane=lane,
                    status="ok",
                    video_count=len(ids),
                    last_error=None,
                )
            conn.execute(
                "INSERT OR REPLACE INTO channel_prescan "
                "(slug, url, tier, category, lane, status, video_count, checked_at, last_error) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    result.slug,
                    result.url,
                    result.tier,
                    result.category,
                    result.lane,
                    result.status,
                    result.video_count,
                    checked_at,
                    result.last_error,
                ),
            )
            results.append(result)
    conn.close()
    return results


def _lane_docker_prefix(lane: str, out_dir: Path, cookies: Path | None) -> list[str]:
    """``docker run`` prefix that runs yt-dlp inside gluetun container ``lane``'s netns.

    The container borrows the VPN container's network namespace (``--network=container:<lane>``) so
    the download egresses through the VPN's clean IP. The output dir and cookies are bind-mounted
    at their **resolved real paths** (the project data dir is a symlink into ``/mnt/tiny-2t``; only
    the real target is mounted), and the yt-dlp flags are built against those same paths — so
    ``--download-archive``, ``-o``, and ``--cookies`` resolve identically in and out of the
    container, and files / the archive / ``<id>.flac`` naming land in the same place as a host run.
    ``--user`` + a writable ``HOME`` (pointed at the mounted out_dir) make the output files
    ``simon:simon`` on ``/mnt/tiny-2t`` and give deno a place for its cache. Cookies are mounted
    read-write because yt-dlp rewrites the jar on exit.
    """
    real_out = out_dir.resolve()
    prefix = [
        "docker",
        "run",
        "--rm",
        f"--network=container:{lane}",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        f"HOME={real_out}",
        "-v",
        f"{real_out}:{real_out}",
    ]
    if cookies is not None:
        real_cookies = cookies.resolve()
        prefix += ["-v", f"{real_cookies}:{real_cookies}"]
    prefix.append(YTDLP_LANE_IMAGE)
    return prefix


def _download_blocked(
    out_dir: Path, *, min_free_gb: float | None, abort: Callable[[], bool] | None
) -> str | None:
    """Why download must stop before the next batch (hard-halt / disk floor), else ``None``.

    The factory's backpressure floor and hard-halt flow through here: a per-batch re-check is what
    lets a download abort *mid-channel* instead of only between channels.
    """
    if abort is not None and abort():
        return "hard-halt signal"
    if min_free_gb is not None:
        free_gb = shutil.disk_usage(out_dir).free / 1e9
        if free_gb < min_free_gb:
            return f"disk floor: {free_gb:.0f} GB free < {min_free_gb:.0f} GB"
    return None


def download_channel(
    channel_url: str,
    *,
    out_dir: Path,
    limit: int | None = None,
    sleep: float = 1.0,
    cookies: Path | None = None,
    lane: str | None = None,
    min_free_gb: float | None = None,
    abort: Callable[[], bool] | None = None,
    batch: int = 25,
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

    ``min_free_gb`` and ``abort`` make the pull abortable *mid-channel* (the factory's backpressure
    floor / hard-halt): when either is set the channel is fetched in ``batch``-sized chunks
    (``--max-downloads``) with the guard re-checked between chunks, so the download stops within a
    chunk of crossing the floor instead of running the whole channel. With neither set the behavior
    is exactly the prior single yt-dlp pass.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # In a lane run only the *resolved* real paths are bind-mounted into the container (the project
    # data dir is a symlink into /mnt/tiny-2t), so build the file-path flags against those. A host
    # run uses the paths as given — behaviorally identical, since the symlink resolves locally.
    flag_dir = out_dir.resolve() if lane is not None else out_dir
    flag_cookies = cookies.resolve() if (lane is not None and cookies is not None) else cookies
    # yt-dlp flags, identical for host and lane runs; only the *program prefix* differs.
    yt_flags = [
        "--download-archive",
        str(flag_dir / "downloaded.txt"),
        "--lazy-playlist",  # download as the channel is listed, not after — files land immediately
        "--ignore-errors",
        "--write-info-json",
        # Throttle so YouTube doesn't rate-limit / bot-block us — the two official sleep knobs:
        # --sleep-requests delays metadata extraction, --sleep-interval/--max delays each download.
        "--sleep-requests",
        str(sleep),
        "--sleep-interval",
        str(sleep),
        "--max-sleep-interval",
        str(round(sleep * 3, 1)),
        "--retries",
        "10",
        "--extractor-retries",
        "5",
        *_AUDIO_TO_16K_FLAC,
        "-o",
        str(flag_dir / "%(id)s.%(ext)s"),
    ]
    if flag_cookies is not None:
        yt_flags += ["--cookies", str(flag_cookies)]
    if limit is not None:
        yt_flags += ["--playlist-end", str(limit)]

    guarded = min_free_gb is not None or abort is not None
    while True:
        if _download_blocked(out_dir, min_free_gb=min_free_gb, abort=abort) is not None:
            break
        batch_flags = [*yt_flags]
        if guarded:  # cap the chunk so the guard is re-checked; --download-archive resumes past it
            batch_flags += ["--max-downloads", str(batch)]
        batch_flags.append(channel_url)
        if lane is None:  # host run — unchanged behavior
            cmd = [*_ytdlp_base(), *batch_flags]
        else:  # VPN lane — same deno-bundled yt-dlp args, but inside the container's netns
            cmd = [*_lane_docker_prefix(lane, out_dir, cookies), *_ytdlp_yt_args(), *batch_flags]
        # --ignore-errors makes yt-dlp continue past bad videos and can exit non-zero; don't treat
        # that as fatal — the per-file results on disk are the source of truth.
        rc = subprocess.run(cmd, check=False).returncode  # noqa: S603
        # Unguarded: a single pass (legacy behavior). Guarded: keep going only while yt-dlp stopped
        # at the --max-downloads cap (101 -> more of the channel remains); any other exit means the
        # channel is drained (or failed), so stop.
        if not guarded or rc != _YTDLP_MAX_DOWNLOADS_RC:
            break

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
        sys.executable,
        "-m",
        "yt_dlp",
        "--cookies-from-browser",
        f"chrome:{profile_dir}",
        "--cookies",
        str(out_path),
        "--skip-download",
        "--no-warnings",
        "--quiet",
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # any stable public video works
    ]
    subprocess.run(cmd, check=False)  # noqa: S603 — fixed argv, no shell
    if not out_path.exists():
        msg = f"yt-dlp wrote no cookie file at {out_path}"
        raise RuntimeError(msg)
    count = sum(
        1
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if "youtube.com" in line and not line.startswith("#")
    )
    if count == 0:
        msg = f"no youtube.com cookies extracted from {profile_dir} — is the profile logged in?"
        raise RuntimeError(msg)
    return count
