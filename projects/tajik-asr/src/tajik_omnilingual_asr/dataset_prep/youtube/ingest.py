"""Stage 1 — download YouTube audio + captions into the sqlite DB.

`download_video` pulls 16 kHz mono FLAC + the info JSON + any requested caption tracks
via yt-dlp; `store_video` records the video and its captions. No transcription here —
Scribe is its own stage (`transcribe.py`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tajik_omnilingual_asr.dataset_prep.youtube.db import (
    connect,
    ensure_schema,
    has_archived_video,
)
from tajik_omnilingual_asr.dataset_prep.youtube.process import (
    now_iso,
    run,
    run_json,
    safe_video_url,
    ytdlp_base,
)

# --- caption parsing (yt-dlp writes json3/vtt/srt tracks alongside the audio) ---


def caption_language(path: Path, video_id: str) -> str:
    stem = path.name.removeprefix(f"{video_id}.")
    for suffix in (".json3", ".vtt", ".srt"):
        stem = stem.removesuffix(suffix)
    return stem


def official_caption_languages(info: dict[str, Any]) -> set[str]:
    subtitles = info.get("subtitles")
    if not isinstance(subtitles, dict):
        return set()
    return {str(language) for language, tracks in subtitles.items() if tracks}


def automatic_caption_languages(info: dict[str, Any]) -> set[str]:
    captions = info.get("automatic_captions")
    if not isinstance(captions, dict):
        return set()
    return {str(language) for language, tracks in captions.items() if tracks}


def caption_source_kind(path: Path, video_id: str, info: dict[str, Any]) -> str:
    language = caption_language(path, video_id)
    if language in official_caption_languages(info):
        return "manual"
    if language in automatic_caption_languages(info):
        return "auto"
    return "unknown"


def json3_cues(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cues: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        segs = event.get("segs")
        if not isinstance(segs, list):
            continue
        text = "".join(str(seg.get("utf8", "")) for seg in segs if isinstance(seg, dict)).strip()
        if not text:
            continue
        start_ms = int(event.get("tStartMs") or 0)
        dur_ms = int(event.get("dDurationMs") or 0)
        cues.append(
            {
                "start": start_ms / 1000,
                "end": (start_ms + dur_ms) / 1000 if dur_ms else None,
                "text": re.sub(r"\s+", " ", text),
            }
        )
    return cues


def parse_caption(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.suffix == ".json3":
        cues = json3_cues(path)
        return " ".join(cue["text"] for cue in cues), cues
    text_lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped == "WEBVTT" or "-->" in stripped or stripped.isdigit():
            continue
        text_lines.append(stripped)
    text = re.sub(r"\s+", " ", " ".join(text_lines)).strip()
    return text, []


@dataclass(frozen=True)
class DownloadedVideo:
    video_id: str
    video_dir: Path
    audio_path: Path
    info_path: Path
    caption_paths: list[Path]


def download_video(
    url: str,
    *,
    out_dir: Path,
    include_auto_captions: bool,
    caption_languages: str,
) -> DownloadedVideo:
    out_dir.mkdir(parents=True, exist_ok=True)
    info = run_json([*ytdlp_base(), "--skip-download", "--dump-single-json", url])
    video_id = str(info["id"])
    video_dir = out_dir / "videos" / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    command = [
        *ytdlp_base(),
        "--no-playlist",
        "--write-info-json",
        "--write-subs",
        "--sub-langs",
        caption_languages,
        "--sub-format",
        "json3/vtt/srt",
        "-f",
        "ba/bestaudio",
        "--extract-audio",
        "--audio-format",
        "flac",
        "--audio-quality",
        "0",
        "--postprocessor-args",
        "ffmpeg:-ar 16000 -ac 1",
        "-o",
        str(video_dir / "%(id)s.%(ext)s"),
        url,
    ]
    if include_auto_captions:
        command.insert(command.index("--sub-langs"), "--write-auto-subs")
    run(command)
    audio_path = video_dir / f"{video_id}.flac"
    info_path = video_dir / f"{video_id}.info.json"
    caption_paths = sorted(
        path
        for path in video_dir.iterdir()
        if path.name.startswith(f"{video_id}.") and path.suffix in {".json3", ".vtt", ".srt"}
    )
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    return DownloadedVideo(video_id, video_dir, audio_path, info_path, caption_paths)


def store_video(conn: Any, downloaded: DownloadedVideo, url: str) -> None:
    info = json.loads(downloaded.info_path.read_text(encoding="utf-8"))
    if official_caption_languages(info):
        caption_status = "manual"
    elif automatic_caption_languages(info):
        caption_status = "manual_none_auto_available"
    else:
        caption_status = "none"
    conn.execute(
        """
        insert into youtube_videos (
            video_id, url, title, channel, channel_id, duration, upload_date,
            webpage_url, audio_path, info_json_path, info_json, caption_status,
            created_at, updated_at
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(video_id) do update set
            url=excluded.url, title=excluded.title, channel=excluded.channel,
            channel_id=excluded.channel_id, duration=excluded.duration,
            upload_date=excluded.upload_date, webpage_url=excluded.webpage_url,
            audio_path=excluded.audio_path, info_json_path=excluded.info_json_path,
            info_json=excluded.info_json, caption_status=excluded.caption_status,
            updated_at=excluded.updated_at
        """,
        (
            downloaded.video_id,
            url,
            str(info.get("title") or ""),
            str(info.get("channel") or info.get("uploader") or ""),
            str(info.get("channel_id") or ""),
            info.get("duration"),
            str(info.get("upload_date") or ""),
            str(info.get("webpage_url") or url),
            str(downloaded.audio_path),
            str(downloaded.info_path),
            json.dumps(info, ensure_ascii=False, sort_keys=True),
            caption_status,
            now_iso(),
            now_iso(),
        ),
    )
    for path in downloaded.caption_paths:
        text, cues = parse_caption(path)
        conn.execute(
            """
            insert or replace into youtube_captions (
                video_id, language, source_kind, path, text, cues_json, created_at
            ) values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                downloaded.video_id,
                caption_language(path, downloaded.video_id),
                caption_source_kind(path, downloaded.video_id, info),
                str(path),
                text,
                json.dumps(cues, ensure_ascii=False),
                now_iso(),
            ),
        )
    conn.commit()


def channel_entries(channel_url: str) -> list[dict[str, Any]]:
    payload = run_json([*ytdlp_base(), "--flat-playlist", "--dump-single-json", channel_url])
    entries = payload.get("entries") or []
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict) and e.get("id")]


def _select(entries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    pattern = re.compile(args.exclude_title_regex) if args.exclude_title_regex else None
    selected = []
    for entry in entries:
        title = str(entry.get("title") or "")
        duration = entry.get("duration")
        if pattern and pattern.search(title):
            continue
        if args.max_duration_seconds and isinstance(duration, int | float):
            if float(duration) > args.max_duration_seconds:
                continue
        selected.append(entry)
    return selected[: args.limit] if args.limit else selected


def cmd_list_channel(args: argparse.Namespace) -> int:
    entries = channel_entries(args.channel_url)
    limit = args.limit or len(entries)
    for entry in entries[:limit]:
        print(f"{entry.get('id') or ''}\t{entry.get('duration') or ''}\t{entry.get('title') or ''}")
    print(f"total\t{len(entries)}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    db_path = args.db or args.artifact_dir / "youtube_learning_tajik.sqlite"
    if args.channel_url:
        targets = [
            (str(e["id"]), safe_video_url(str(e["id"])))
            for e in _select(channel_entries(args.channel_url), args)
        ]
    else:
        targets = [(url, safe_video_url(url)) for url in args.url]
    if not targets:
        raise ValueError("no videos to download (give URLs or --channel-url)")

    fail_path = args.fail_jsonl or args.artifact_dir / "download_failures.jsonl"
    completed = failed = skipped = 0
    with connect(db_path) as conn:
        ensure_schema(conn)
        for index, (video_id, url) in enumerate(targets, start=1):
            if args.skip_existing and has_archived_video(conn, video_id):
                skipped += 1
                print(f"skip\t{index}/{len(targets)}\t{video_id}")
                continue
            print(f"download\t{index}/{len(targets)}\t{video_id}")
            try:
                downloaded = download_video(
                    url,
                    out_dir=args.artifact_dir,
                    include_auto_captions=args.auto_captions,
                    caption_languages=args.caption_languages,
                )
                store_video(conn, downloaded, url)
                completed += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                fail_path.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "video_id": video_id,
                    "url": url,
                    "error": repr(exc),
                    "created_at": now_iso(),
                }
                with fail_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"failed\t{video_id}\t{exc}", file=sys.stderr)
    print(f"completed\t{completed}\tskipped\t{skipped}\tfailed\t{failed}\tdb\t{db_path}")
    return 1 if failed and args.fail_on_error else 0
