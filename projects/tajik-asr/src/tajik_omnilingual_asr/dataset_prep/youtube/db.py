from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma journal_mode=wal")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        create table if not exists youtube_videos (
            video_id text primary key,
            url text not null,
            title text not null,
            channel text not null default '',
            channel_id text not null default '',
            duration real,
            upload_date text not null default '',
            webpage_url text not null default '',
            audio_path text not null default '',
            info_json_path text not null default '',
            info_json text not null,
            caption_status text not null default '',
            created_at text not null,
            updated_at text not null
        );

        create table if not exists youtube_captions (
            video_id text not null references youtube_videos(video_id) on delete cascade,
            language text not null,
            source_kind text not null,
            path text not null,
            text text not null,
            cues_json text not null,
            created_at text not null,
            primary key (video_id, language, source_kind, path)
        );

        create table if not exists youtube_scribe_runs (
            id text primary key,
            video_id text not null references youtube_videos(video_id) on delete cascade,
            provider text not null,
            model_key text not null,
            model_id text not null,
            language text,
            transcript text not null,
            normalized_transcript text not null,
            raw_response_json text not null,
            words_json text not null,
            options_json text not null default '{}',
            diarize integer not null default 0,
            num_speakers integer,
            official_caption_language text not null default '',
            official_caption_text text not null default '',
            normalized_official_caption text not null default '',
            official_vs_scribe_wer real,
            official_vs_scribe_cer real,
            error text not null default '',
            created_at text not null
        );

        create table if not exists youtube_segments (
            segment_id text primary key,
            video_id text not null references youtube_videos(video_id) on delete cascade,
            run_id text not null references youtube_scribe_runs(id) on delete cascade,
            source_kind text not null,
            start real not null,
            end real not null,
            duration real not null,
            text text not null,
            normalized_text text not null,
            audio_path text not null,
            source_word_indexes_json text not null,
            created_at text not null
        );

        create table if not exists youtube_omni_transcripts (
            id text primary key,
            video_id text not null references youtube_videos(video_id) on delete cascade,
            run_id text,
            segment_id text,
            model_card text not null,
            audio_path text not null,
            transcript text not null,
            normalized_transcript text not null,
            error text not null default '',
            created_at text not null
        );
        """
    )
    _ensure_column(conn, "youtube_scribe_runs", "options_json", "text not null default '{}'")
    _ensure_column(conn, "youtube_scribe_runs", "diarize", "integer not null default 0")
    _ensure_column(conn, "youtube_scribe_runs", "num_speakers", "integer")
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {str(row["name"]) for row in conn.execute(f"pragma table_info({table})")}
    if column in columns:
        return
    conn.execute(f"alter table {table} add column {column} {definition}")


def latest_scribe_run(conn: sqlite3.Connection, video_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        select *
        from youtube_scribe_runs
        where video_id = ? and error = ''
        order by created_at desc
        limit 1
        """,
        (video_id,),
    ).fetchone()


def load_scribe_run(conn: sqlite3.Connection, run_id: str | None, video_id: str) -> sqlite3.Row:
    row = (
        conn.execute("select * from youtube_scribe_runs where id = ?", (run_id,)).fetchone()
        if run_id
        else latest_scribe_run(conn, video_id)
    )
    if row is None:
        raise ValueError(f"no successful Scribe run found for {video_id}")
    return row


def best_manual_caption(conn: sqlite3.Connection, video_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        select *
        from youtube_captions
        where video_id = ? and source_kind = 'manual'
        order by case language when 'tg' then 0 when 'tgk' then 1 when 'en-orig' then 2 else 3 end
        limit 1
        """,
        (video_id,),
    ).fetchone()


def has_successful_scribe(conn: sqlite3.Connection, video_id: str) -> bool:
    row = conn.execute(
        """
        select 1
        from youtube_scribe_runs
        where video_id = ? and error = ''
        limit 1
        """,
        (video_id,),
    ).fetchone()
    return row is not None


def has_archived_video(conn: sqlite3.Connection, video_id: str) -> bool:
    row = conn.execute(
        """
        select audio_path, info_json_path
        from youtube_videos
        where video_id = ?
        limit 1
        """,
        (video_id,),
    ).fetchone()
    if row is None:
        return False
    return Path(str(row["audio_path"])).exists() and Path(str(row["info_json_path"])).exists()
