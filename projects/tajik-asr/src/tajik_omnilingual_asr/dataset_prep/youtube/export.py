"""Stage 5 — Scribe↔omni agreement gate → omni-parquet (fairseq2-usable).

Joins each segment's Scribe-aligned text with the omni model's transcript of the same
clip, keeps only segments where the two agree (per-segment WER ≤ ``--max-agreement-wer``),
and writes the survivors to the omni-parquet format
(``corpus=youtube_tajik/split=<split>/language=tgk_Cyrl``). Agreement is the quality gate:
we only train on labels both the teacher (Scribe) and our model concur on.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from tajik_omnilingual_asr.dataset_prep.curation.scribe import compute_wer
from tajik_omnilingual_asr.dataset_prep.youtube.db import connect, ensure_schema

LANGUAGE = "tgk_Cyrl"
SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("audio_bytes", pa.list_(pa.int8())),
        ("audio_size", pa.int64()),
    ]
)


def _write_shard(columns: dict[str, list[Any]], out_dir: Path, index: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [
            pa.array(columns["text"], type=pa.string()),
            pa.array(columns["audio_bytes"], type=pa.list_(pa.int8())),
            pa.array(columns["audio_size"], type=pa.int64()),
        ],
        schema=SCHEMA,
    )
    pq.write_table(table, out_dir / f"part-{index:05d}.parquet", row_group_size=100)
    for value in columns.values():
        value.clear()


def cmd_export(args: argparse.Namespace) -> int:
    db_path = args.db or args.artifact_dir / "youtube_learning_tajik.sqlite"
    out_dir = (
        args.output_root
        / "corpus=youtube_tajik"
        / f"split={args.split}"
        / f"language={LANGUAGE}"
    )
    with connect(db_path) as conn:
        ensure_schema(conn)
        rows = conn.execute(
            """
            select
                s.segment_id,
                s.audio_path,
                s.text as scribe_text,
                s.normalized_text as scribe_norm,
                o.transcript as omni_text,
                o.normalized_transcript as omni_norm
            from youtube_segments s
            left join youtube_omni_transcripts o on o.segment_id = s.segment_id
            where s.source_kind = 'nemo_vad'
            order by s.video_id, s.start
            """
        ).fetchall()

    columns: dict[str, list[Any]] = {"text": [], "audio_bytes": [], "audio_size": []}
    shard = kept = seen = empty = disagree = missing = ungated = 0
    for row in rows:
        seen += 1
        scribe_norm = str(row["scribe_norm"] or "")
        omni_norm = str(row["omni_norm"] or "")  # None until transcribe-omni runs
        if not scribe_norm:
            empty += 1
            continue
        if omni_norm:  # agreement gate only applies once omni has transcribed the clip
            if compute_wer(scribe_norm, omni_norm) > args.max_agreement_wer:
                disagree += 1
                continue
        elif args.label_source == "omni":
            missing += 1  # asked for omni labels but none exist for this segment
            continue
        else:
            ungated += 1  # Scribe-only: kept without an agreement check
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.exists():
            missing += 1
            continue
        label = str(row["omni_text"]) if args.label_source == "omni" else str(row["scribe_text"])
        columns["text"].append(label)
        columns["audio_bytes"].append(np.frombuffer(audio_path.read_bytes(), dtype=np.int8))
        columns["audio_size"].append(int(sf.info(audio_path).frames))
        kept += 1
        if len(columns["text"]) >= args.rows_per_file:
            _write_shard(columns, out_dir, shard)
            shard += 1
    if columns["text"]:
        _write_shard(columns, out_dir, shard)

    print(f"output\t{out_dir}")
    print(
        f"seen={seen} kept={kept} (ungated_scribe_only={ungated}) "
        f"dropped: empty={empty} disagree={disagree} missing={missing}"
    )
    print(f"label_source={args.label_source} max_agreement_wer={args.max_agreement_wer}")
    return 0
