"""Stages 2 & 4 — transcription.

`transcribe-scribe` runs ElevenLabs Scribe on full-video audio (its word-level timing
feeds segment alignment) and records WER/CER against any manual caption.
`transcribe-omni` runs the project's own omni CTC model on the cut segments. The two
transcripts (Scribe-aligned segment text vs omni per-segment) are later cross-checked for
agreement in `export.py`.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tajik_omnilingual_asr.dataset_prep.text_normalization import normalize_text
from tajik_omnilingual_asr.dataset_prep.youtube.db import (
    best_manual_caption,
    connect,
    ensure_schema,
    has_successful_scribe,
)
from tajik_omnilingual_asr.dataset_prep.youtube.process import now_iso


def extract_words(raw_response: dict[str, Any]) -> list[dict[str, Any]]:
    words = raw_response.get("words")
    if not isinstance(words, list):
        return []
    return [
        {
            "text": item.get("text") or item.get("word") or "",
            "start": item.get("start"),
            "end": item.get("end"),
            "type": item.get("type") or "",
            "speaker_id": item.get("speaker_id") or item.get("speaker") or "",
        }
        for item in words
        if isinstance(item, dict)
    ]


def cmd_transcribe_scribe(args: argparse.Namespace) -> int:
    from superwhisper_api.audio.models import audio_model
    from superwhisper_api.audio.transcribe import create_process_fn

    from tajik_omnilingual_asr.dataset_prep.curation.scribe import compute_cer, compute_wer

    db_path = args.db or args.artifact_dir / "youtube_learning_tajik.sqlite"
    with connect(db_path) as conn:
        ensure_schema(conn)
        if args.video_id:
            placeholders = ",".join("?" for _ in args.video_id)
            rows = conn.execute(
                "select video_id, audio_path from youtube_videos "
                f"where video_id in ({placeholders})",
                list(args.video_id),
            ).fetchall()
        else:
            rows = conn.execute("select video_id, audio_path from youtube_videos").fetchall()

        spec = audio_model(args.model)
        process = create_process_fn(spec, args.key, language=args.language, diarize=args.diarize)
        done = 0
        for row in rows:
            video_id = str(row["video_id"])
            if args.skip_existing and has_successful_scribe(conn, video_id):
                print(f"skip\t{video_id}")
                continue
            audio_path = Path(str(row["audio_path"]))
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
            payload: dict[str, Any] = process(audio_path).as_dict()
            error = str(payload.get("error") or "")
            transcript = str(payload.get("transcript") or "")
            raw_response = payload.get("raw_response")
            raw: dict[str, Any] = raw_response if isinstance(raw_response, dict) else payload
            normalized = normalize_text(transcript)
            manual = best_manual_caption(conn, video_id)
            official = str(manual["text"]) if manual else ""
            normalized_official = normalize_text(official) if official else ""
            ref = bool(normalized_official and not error)
            wer = compute_wer(normalized_official, normalized) if ref else None
            cer = compute_cer(normalized_official, normalized) if ref else None
            run_id = f"scribe-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{video_id}"
            conn.execute(
                """
                insert or replace into youtube_scribe_runs (
                    id, video_id, provider, model_key, model_id, language, transcript,
                    normalized_transcript, raw_response_json, words_json, options_json,
                    diarize, num_speakers, official_caption_language, official_caption_text,
                    normalized_official_caption, official_vs_scribe_wer, official_vs_scribe_cer,
                    error, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, video_id, spec.provider, spec.key, spec.model_id, args.language,
                    transcript, normalized,
                    json.dumps(raw, ensure_ascii=False, sort_keys=True),
                    json.dumps(extract_words(raw), ensure_ascii=False),
                    json.dumps({"diarize": args.diarize}, ensure_ascii=False, sort_keys=True),
                    1 if args.diarize else 0, None,
                    str(manual["language"]) if manual else "", official, normalized_official,
                    wer, cer, error, now_iso(),
                ),
            )
            conn.commit()
            done += 1
            print(f"scribe\t{video_id}\terror={bool(error)}\twords={len(extract_words(raw))}")
    print(f"transcribed\t{done}\tdb\t{db_path}")
    return 0


def cmd_transcribe_omni(args: argparse.Namespace) -> int:
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    # bf16 is slow/unsupported on CPU; the 300M CTC checkpoint runs fine in float32 on CPU.
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    db_path = args.db or args.artifact_dir / "youtube_learning_tajik.sqlite"
    with connect(db_path) as conn:
        ensure_schema(conn)
        if args.video_id:
            placeholders = ",".join("?" for _ in args.video_id)
            query = (
                "select segment_id, video_id, run_id, audio_path from youtube_segments "
                f"where source_kind='nemo_vad' and video_id in ({placeholders}) "
                "order by video_id, start"
            )
            rows = conn.execute(query, list(args.video_id)).fetchall()
        else:
            rows = conn.execute(
                "select segment_id, video_id, run_id, audio_path from youtube_segments "
                "where source_kind='nemo_vad' order by video_id, start"
            ).fetchall()
        if args.limit:
            rows = rows[: args.limit]
        audio_paths = [Path(str(r["audio_path"])) for r in rows]
        for path in audio_paths:
            if not path.exists():
                raise FileNotFoundError(path)
        if not rows:
            print("no segments to transcribe")
            return 0

        pipeline = ASRInferencePipeline(model_card=args.model_card, device=args.device, dtype=dtype)
        transcripts = pipeline.transcribe([str(p) for p in audio_paths], batch_size=args.batch_size)
        for row, transcript in zip(rows, transcripts, strict=True):
            segment_id = str(row["segment_id"])
            record_id = f"{args.model_card}:{segment_id}"
            conn.execute(
                """
                insert or replace into youtube_omni_transcripts (
                    id, video_id, run_id, segment_id, model_card, audio_path,
                    transcript, normalized_transcript, error, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, '', ?)
                """,
                (
                    record_id, str(row["video_id"]), str(row["run_id"]), segment_id,
                    args.model_card, str(row["audio_path"]), transcript,
                    normalize_text(transcript), now_iso(),
                ),
            )
        conn.commit()
    print(f"transcribed\t{len(rows)}\tdb\t{db_path}")
    return 0
