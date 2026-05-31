from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from tajik_omnilingual_asr.dataset_prep.text_normalization import (
    normalize_text,
    strip_fleurs_token_trace,
)

RAW_ROOT = Path("/home/simon/github/peacock-asr/projects/tajik-asr/data/raw")
OUT_ROOT = Path(
    "/home/simon/github/peacock-asr/projects/tajik-asr/src/"
    "tajik_omnilingual_asr/dataset_prep/artifacts/tajik_asr_combined_v0"
)
OMNI_SAMPLE_RATE = 16_000
SPLITS = ("train", "dev", "test")
CV_SPLIT_FILES = {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"}
HF_SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test"}


@dataclass(frozen=True)
class Utterance:
    source: str
    source_split: str
    source_id: str
    raw_text: str
    text: str
    audio_path: Path
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class OutputRow:
    utt_id: str
    split: str
    source: str
    source_split: str
    source_id: str
    audio_filename: str
    raw_transcription: str
    transcription: str
    normalized_text: str
    characters: int
    audio_bytes: int
    duplicate_count: int
    source_metadata_json: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the combined deduplicated Tajik ASR dataset."
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--replace", action="store_true")
    return parser


def stable_id(source: str, split: str, source_id: str, text: str) -> str:
    payload = "\t".join([source, split, source_id, normalize_text(text)])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"tg_{digest}"


def source_priority(source: str) -> int:
    priorities = {
        "fleurs_tg_tj": 10,
        "common_voice_25_tg": 20,
        "shunyalabs_tajik_speech_dataset": 30,
        "wuenlp_sib_fleurs_tgk_cyrl": 40,
        "muhtasham_tajik_asr_augmented_test": 50,
    }
    return priorities.get(source, 100)


def parse_fleurs(root: Path) -> list[Utterance]:
    rows: list[Utterance] = []
    source = "fleurs_tg_tj"
    for split in SPLITS:
        tsv_path = root / source / split / "data.tsv"
        audio_dir = root / source / split / "audio"
        if not tsv_path.exists():
            continue
        with tsv_path.open(encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row_index, row in enumerate(reader):
                if len(row) < 4:
                    continue
                audio_file = row[1]
                raw_text = row[2]
                text = strip_fleurs_token_trace(row[3])
                audio_path = audio_dir / audio_file
                if not audio_path.exists():
                    continue
                rows.append(
                    Utterance(
                        source=source,
                        source_split=split,
                        source_id=row[0] or f"{split}-{row_index}",
                        raw_text=raw_text,
                        text=text,
                        audio_path=audio_path,
                        source_metadata={"fleurs_row": row},
                    )
                )
    return rows


def parse_common_voice(root: Path) -> list[Utterance]:
    rows: list[Utterance] = []
    source = "common_voice_25_tg"
    source_root = root / source
    clips = source_root / "clips"
    if not clips.exists():
        return rows
    for split, filename in CV_SPLIT_FILES.items():
        tsv_path = source_root / filename
        if not tsv_path.exists():
            continue
        with tsv_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row_index, row in enumerate(reader):
                audio_name = row.get("path", "")
                text = row.get("sentence", "")
                if not audio_name or not text:
                    continue
                audio_path = clips / audio_name
                if not audio_path.exists():
                    continue
                rows.append(
                    Utterance(
                        source=source,
                        source_split=split,
                        source_id=row.get("client_id", "") or f"{split}-{row_index}",
                        raw_text=text,
                        text=text,
                        audio_path=audio_path,
                        source_metadata={k: v for k, v in row.items() if k != "sentence"},
                    )
                )
    return rows


def text_from_row(row: dict[str, Any], preferred: tuple[str, ...]) -> str:
    for name in preferred:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def audio_from_row(row: dict[str, Any]) -> tuple[bytes | None, str]:
    value = row.get("audio")
    if isinstance(value, dict):
        audio_bytes = value.get("bytes")
        path = value.get("path") or "audio"
        if isinstance(audio_bytes, bytes):
            suffix = Path(str(path)).suffix or ".audio"
            return audio_bytes, suffix
        array = value.get("array")
        sampling_rate = value.get("sampling_rate")
        if array is not None and sampling_rate is not None:
            import io

            import numpy as np
            import soundfile as sf

            buffer = io.BytesIO()
            sf.write(buffer, np.asarray(array), int(sampling_rate), format="FLAC")
            return buffer.getvalue(), ".flac"
    return None, ""


def audio_items_from_row(row: dict[str, Any]) -> list[tuple[bytes, str, int]]:
    value = row.get("audio")
    if isinstance(value, list):
        items: list[tuple[bytes, str, int]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            item = cast(dict[str, Any], item)
            audio_bytes = item.get("bytes")
            if isinstance(audio_bytes, bytes):
                path = item.get("path") or f"audio_{index}"
                suffix = Path(str(path)).suffix or ".audio"
                items.append((audio_bytes, suffix, index))
        return items
    audio_bytes, suffix = audio_from_row(row)
    if audio_bytes is None:
        return []
    return [(audio_bytes, suffix, 0)]


def parquet_rows(
    source_root: Path,
    source: str,
    preferred_text: tuple[str, ...],
    temp_audio_root: Path,
) -> tuple[list[Utterance], dict[str, Any]]:
    rows: list[Utterance] = []
    report: dict[str, Any] = {"source": source, "files": [], "skipped_rows": 0}
    for parquet_path in sorted(source_root.glob("*/*.parquet")):
        source_split_name = parquet_path.parent.name
        split = HF_SPLIT_MAP.get(source_split_name)
        if split is None:
            report["files"].append({"path": str(parquet_path), "reason": "unsupported_split"})
            continue
        parquet = pq.ParquetFile(parquet_path)
        columns = [
            name for name in ("audio", *preferred_text) if name in parquet.schema_arrow.names
        ]
        file_rows = 0
        for batch_index, batch in enumerate(parquet.iter_batches(batch_size=128, columns=columns)):
            for row_index, row in enumerate(batch.to_pylist()):
                text = text_from_row(row, preferred_text)
                audio_items = audio_items_from_row(row)
                if not text or not audio_items:
                    report["skipped_rows"] += 1
                    continue
                for audio_bytes, suffix, audio_index in audio_items:
                    local_id = f"{source_split_name}-{batch_index}-{row_index}-{audio_index}"
                    digest = hashlib.sha1(
                        b"\t".join(
                            [
                                source.encode("utf-8"),
                                source_split_name.encode("utf-8"),
                                local_id.encode("utf-8"),
                                normalize_text(text).encode("utf-8"),
                            ]
                        )
                    ).hexdigest()[:20]
                    audio_path = temp_audio_root / source / source_split_name / f"{digest}{suffix}"
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    audio_path.write_bytes(audio_bytes)
                    rows.append(
                        Utterance(
                            source=source,
                            source_split=split,
                            source_id=local_id,
                            raw_text=text,
                            text=text,
                            audio_path=audio_path,
                            source_metadata={
                                "parquet": str(parquet_path),
                                "source_split": source_split_name,
                                "audio_index": audio_index,
                            },
                        )
                    )
                    file_rows += 1
        report["files"].append({"path": str(parquet_path), "rows": file_rows})
    return rows, report


def parse_hf_sources(
    root: Path, temp_audio_root: Path
) -> tuple[list[Utterance], list[dict[str, Any]]]:
    rows: list[Utterance] = []
    reports: list[dict[str, Any]] = []
    configs = [
        ("shunyalabs_tajik_speech_dataset", ("transcript",)),
        ("muhtasham_tajik_asr_augmented_test", ("sentence",)),
        ("wuenlp_sib_fleurs_tgk_cyrl", ("transcription", "sentence", "raw_transcription")),
    ]
    for source, text_columns in configs:
        source_root = root / source
        if not source_root.exists():
            continue
        parsed, report = parquet_rows(source_root, source, text_columns, temp_audio_root)
        rows.extend(parsed)
        reports.append(report)

    for source in ("wuenlp_belebele_fleurs_tgk_cyrl", "facebook_2m_belebele_tgk_cyrl"):
        source_root = root / source
        if source_root.exists():
            reports.append(
                {
                    "source": source,
                    "status": "kept_raw_skipped_from_asr_combined",
                    "reason": (
                        "nested evaluation or QA data needs a dedicated extractor "
                        "before ASR training use"
                    ),
                    "parquet_files": [
                        str(path) for path in sorted(source_root.glob("*/*.parquet"))
                    ],
                }
            )
    return rows, reports


def split_for_row(row: Utterance) -> str:
    if row.source_split in SPLITS:
        return row.source_split
    return HF_SPLIT_MAP.get(row.source_split, "train")


def copy_audio(row: Utterance, out_root: Path, split: str, utt_id: str) -> tuple[str, int]:
    # Resample every clip to 16 kHz mono. fairseq2's AudioDecoder decodes at the file's
    # native rate and never resamples, while the Omni parquet export records
    # ``audio_size = duration * 16000``; a non-16 kHz clip would then decode to more
    # samples than the length-batcher budgeted -> bucket overshoot -> OOM (and wrong-rate
    # audio fed to the model). Writing 16 kHz mono WAV here keeps length and audio honest.
    import librosa
    import soundfile as sf

    relative = Path(split) / "audio" / f"{utt_id}.wav"
    target = out_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    waveform, _ = librosa.load(str(row.audio_path), sr=OMNI_SAMPLE_RATE, mono=True)
    sf.write(str(target), waveform, OMNI_SAMPLE_RATE, subtype="PCM_16")
    return relative.as_posix(), target.stat().st_size


def choose_deduped(rows: list[Utterance]) -> tuple[list[Utterance], dict[str, int]]:
    by_text: dict[str, list[Utterance]] = {}
    empty_normalized = 0
    for row in rows:
        normalized = normalize_text(row.text)
        if not normalized:
            empty_normalized += 1
            continue
        by_text.setdefault(normalized, []).append(row)
    chosen = [
        sorted(group, key=lambda item: (source_priority(item.source), item.source_id))[0]
        for group in by_text.values()
    ]
    duplicate_rows = sum(len(group) - 1 for group in by_text.values())
    report = {
        "input_rows": len(rows),
        "empty_normalized_rows": empty_normalized,
        "unique_normalized_texts": len(by_text),
        "duplicate_rows": duplicate_rows,
        "output_rows": len(chosen),
    }
    return (
        sorted(chosen, key=lambda item: (split_for_row(item), item.source, item.source_id)),
        report,
    )


def write_tsv(rows: list[OutputRow], out_root: Path) -> None:
    header = [
        "id",
        "audio_filename",
        "raw_transcription",
        "transcription",
        "normalized_text",
        "characters",
        "audio_bytes",
        "source",
        "source_id",
        "duplicate_count",
    ]
    for split in SPLITS:
        split_rows = [row for row in rows if row.split == split]
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "data.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(header)
            for row in split_rows:
                writer.writerow(
                    [
                        row.utt_id,
                        row.audio_filename,
                        row.raw_transcription,
                        row.transcription,
                        row.normalized_text,
                        row.characters,
                        row.audio_bytes,
                        row.source,
                        row.source_id,
                        row.duplicate_count,
                    ]
                )


def write_sqlite(rows: list[OutputRow], out_root: Path) -> None:
    db_path = out_root / "tajik_asr_combined.sqlite"
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table utterances (
              id text primary key,
              split text not null,
              source text not null,
              source_split text not null,
              source_id text not null,
              audio_filename text not null,
              raw_transcription text not null,
              transcription text not null,
              normalized_text text not null,
              characters integer not null,
              audio_bytes integer not null,
              duplicate_count integer not null,
              source_metadata_json text not null
            )
            """
        )
        conn.executemany(
            """
            insert into utterances values (
              :utt_id, :split, :source, :source_split, :source_id, :audio_filename,
              :raw_transcription, :transcription, :normalized_text, :characters,
              :audio_bytes, :duplicate_count, :source_metadata_json
            )
            """,
            [asdict(row) for row in rows],
        )
        conn.execute("create index utterances_split_idx on utterances(split)")
        conn.execute("create index utterances_source_idx on utterances(source)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.raw_root.exists():
        raise SystemExit(
            f"raw input directory is missing: {args.raw_root}; restore raw inputs or use "
            "the existing combined artifact"
        )
    if args.out_root.exists() and args.replace:
        shutil.rmtree(args.out_root)
    if args.out_root.exists():
        raise SystemExit(f"output exists, rerun with --replace: {args.out_root}")

    temp_audio_root = args.out_root / "_extracted_audio"
    input_rows: list[Utterance] = []
    input_rows.extend(parse_fleurs(args.raw_root))
    input_rows.extend(parse_common_voice(args.raw_root))
    hf_rows, hf_reports = parse_hf_sources(args.raw_root, temp_audio_root)
    input_rows.extend(hf_rows)
    if not input_rows:
        raise SystemExit(
            f"no input rows found under {args.raw_root}; restore raw inputs or use "
            "the existing combined artifact"
        )

    chosen, dedupe_report = choose_deduped(input_rows)
    duplicate_counts: dict[str, int] = {}
    for row in input_rows:
        normalized = normalize_text(row.text)
        if normalized:
            duplicate_counts[normalized] = duplicate_counts.get(normalized, 0) + 1

    output_rows: list[OutputRow] = []
    for row in chosen:
        split = split_for_row(row)
        normalized = normalize_text(row.text)
        utt_id = stable_id(row.source, split, row.source_id, row.text)
        audio_filename, audio_bytes = copy_audio(row, args.out_root, split, utt_id)
        output_rows.append(
            OutputRow(
                utt_id=utt_id,
                split=split,
                source=row.source,
                source_split=row.source_split,
                source_id=row.source_id,
                audio_filename=audio_filename,
                raw_transcription=row.raw_text,
                transcription=normalized,
                normalized_text=normalized,
                characters=len(normalized),
                audio_bytes=audio_bytes,
                duplicate_count=duplicate_counts[normalized],
                source_metadata_json=json.dumps(
                    row.source_metadata, ensure_ascii=False, sort_keys=True
                ),
            )
        )

    if temp_audio_root.exists():
        shutil.rmtree(temp_audio_root)
    write_tsv(output_rows, args.out_root)
    write_sqlite(output_rows, args.out_root)

    summary = {
        "raw_root": str(args.raw_root),
        "out_root": str(args.out_root),
        "dedupe": dedupe_report,
        "rows_by_split": {
            split: sum(1 for row in output_rows if row.split == split) for split in SPLITS
        },
        "rows_by_source": {
            source: sum(1 for row in output_rows if row.source == source)
            for source in sorted({row.source for row in output_rows})
        },
        "hf_reports": hf_reports,
    }
    (args.out_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
