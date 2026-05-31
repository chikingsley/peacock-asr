"""Stage 3 — NeMo VAD segmentation + automatic Scribe-word alignment.

For each video: run the NeMo frame-VAD to get speech windows, then assign each window the
Scribe words whose timestamps fall inside it (the latest successful Scribe run supplies the
word-level timing). Each window becomes a youtube_segments row with cut 16 kHz FLAC + the
aligned text. This replaces the old hand-authored cut plan — no human in the loop.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tajik_omnilingual_asr.dataset_prep.text_normalization import normalize_text
from tajik_omnilingual_asr.dataset_prep.youtube.db import connect, ensure_schema, load_scribe_run
from tajik_omnilingual_asr.dataset_prep.youtube.paths import ROOT
from tajik_omnilingual_asr.dataset_prep.youtube.process import cut_audio, now_iso

NEMO_VAD_MODEL_NAME = "vad_multilingual_frame_marblenet"
NEMO_VAD_MODEL_VERSION = "1.20.0"
NEMO_VAD_MODEL_PATH = (
    ROOT
    / "src/tajik_omnilingual_asr/dataset_prep/artifacts/models"
    / f"nemo_{NEMO_VAD_MODEL_NAME}_{NEMO_VAD_MODEL_VERSION}"
    / f"{NEMO_VAD_MODEL_NAME}.nemo"
)


@dataclass(frozen=True)
class SpeechWindow:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def merge_windows(
    windows: list[SpeechWindow], *, merge_gap_seconds: float, hard_max_seconds: float
) -> list[SpeechWindow]:
    merged: list[SpeechWindow] = []
    for window in windows:
        if merged and window.start - merged[-1].end <= merge_gap_seconds:
            combined = window.end - merged[-1].start
            if combined <= hard_max_seconds:
                merged[-1] = SpeechWindow(merged[-1].start, window.end)
                continue
        merged.append(window)
    return merged


def boolean_windows(
    flags: list[bool],
    *,
    frame_seconds: float,
    min_duration_seconds: float,
    merge_gap_seconds: float,
    hard_max_seconds: float,
) -> list[SpeechWindow]:
    raw: list[SpeechWindow] = []
    start_index: int | None = None
    for index, flag in enumerate(flags):
        if flag and start_index is None:
            start_index = index
        elif not flag and start_index is not None:
            raw.append(SpeechWindow(start_index * frame_seconds, index * frame_seconds))
            start_index = None
    if start_index is not None:
        raw.append(SpeechWindow(start_index * frame_seconds, len(flags) * frame_seconds))
    merged = merge_windows(
        raw, merge_gap_seconds=merge_gap_seconds, hard_max_seconds=hard_max_seconds
    )
    return [w for w in merged if w.duration >= min_duration_seconds]


def aligned_text(words: list[dict[str, Any]], start: float, end: float) -> tuple[str, list[int]]:
    """Join the Scribe words whose start time falls in [start, end); skip audio events."""
    parts: list[str] = []
    indexes: list[int] = []
    for i, word in enumerate(words):
        ws = word.get("start")
        if not isinstance(ws, int | float) or str(word.get("type") or "") == "audio_event":
            continue
        if start <= float(ws) < end:
            text = str(word.get("text") or "").strip()
            if text:
                parts.append(text)
                indexes.append(i)
    return " ".join(parts), indexes


def _video_rows(conn: Any, video_ids: list[str]) -> list[Any]:
    if video_ids:
        placeholders = ",".join("?" for _ in video_ids)
        query = (
            "select video_id, title, audio_path from youtube_videos "
            f"where video_id in ({placeholders}) order by upload_date, video_id"
        )
        return conn.execute(query, video_ids).fetchall()
    return conn.execute(
        "select video_id, title, audio_path from youtube_videos order by upload_date, video_id"
    ).fetchall()


def cmd_segment(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from nemo.collections.asr.parts.utils.vad_utils import EncDecFrameClassificationModel

    if not NEMO_VAD_MODEL_PATH.exists():
        raise FileNotFoundError(NEMO_VAD_MODEL_PATH)
    db_path = args.db or args.artifact_dir / "youtube_learning_tajik.sqlite"

    model = EncDecFrameClassificationModel.restore_from(
        str(NEMO_VAD_MODEL_PATH), map_location=args.device
    )
    model.eval()

    def windows_for(audio: Path) -> list[SpeechWindow]:
        with torch.no_grad():
            logits = model.transcribe([str(audio)], batch_size=1, logprobs=True)[0]
        probs = torch.softmax(torch.from_numpy(np.asarray(logits)), dim=-1).numpy()
        speech = probs[:, args.speech_class_index]
        return boolean_windows(
            [float(p) >= args.threshold for p in speech],
            frame_seconds=args.frame_seconds,
            min_duration_seconds=args.min_duration_seconds,
            merge_gap_seconds=args.merge_gap_seconds,
            hard_max_seconds=args.max_duration_seconds,
        )

    total = 0
    with connect(db_path) as conn:
        ensure_schema(conn)
        rows = _video_rows(conn, list(args.video_id or []))
        if not rows:
            raise ValueError("no videos selected")
        for row in rows:
            video_id = str(row["video_id"])
            source_audio = Path(str(row["audio_path"]))
            if not source_audio.exists():
                raise FileNotFoundError(source_audio)
            scribe_run = load_scribe_run(conn, None, video_id)
            words = json.loads(str(scribe_run["words_json"]) or "[]")
            out_dir = args.artifact_dir / "cuts" / video_id / str(scribe_run["id"])
            conn.execute(
                "delete from youtube_segments where video_id = ? and source_kind = 'nemo_vad'",
                (video_id,),
            )
            kept = 0
            for index, window in enumerate(windows_for(source_audio)):
                if not (args.min_duration_seconds <= window.duration <= args.max_duration_seconds):
                    continue
                text, word_indexes = aligned_text(words, window.start, window.end)
                if not text:
                    continue
                segment_id = f"{video_id}_nemo_vad_{index:04d}"
                audio_path = out_dir / f"{segment_id}.flac"
                cut_audio(source_audio, audio_path, window.start, window.end)
                conn.execute(
                    """
                    insert or replace into youtube_segments (
                        segment_id, video_id, run_id, source_kind, start, end,
                        duration, text, normalized_text, audio_path,
                        source_word_indexes_json, created_at
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        segment_id,
                        video_id,
                        str(scribe_run["id"]),
                        "nemo_vad",
                        round(window.start, 3),
                        round(window.end, 3),
                        round(window.duration, 3),
                        text,
                        normalize_text(text),
                        str(audio_path),
                        json.dumps(word_indexes),
                        now_iso(),
                    ),
                )
                kept += 1
            conn.commit()
            total += kept
            print(f"{video_id}\tsegments\t{kept}")
    print(f"total_segments\t{total}\tdb\t{db_path}")
    return 0
