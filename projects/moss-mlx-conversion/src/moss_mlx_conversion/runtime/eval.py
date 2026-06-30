from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import jiwer
import librosa
import numpy as np
import soundfile as sf

from moss_mlx_conversion.dump import write_json

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class StreamingExample:
    row_idx: int
    example_id: str
    reference: str
    audio_url: str


def normalize_for_wer(text: str) -> str:
    lowered = text.lower()
    no_punctuation = NON_WORD_RE.sub(" ", lowered)
    return SPACE_RE.sub(" ", no_punctuation).strip()


def extract_audio_url(row: dict[str, Any], audio_column: str) -> str:
    audio_value = row[audio_column]
    if not isinstance(audio_value, list):
        raise TypeError(f"Expected audio cell to be a list, got {type(audio_value).__name__}")
    for entry in audio_value:
        if isinstance(entry, dict) and entry.get("src"):
            return str(entry["src"])
    raise ValueError(f"No audio src found in {audio_column}")


def iter_hf_rows(
    client: httpx.Client,
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    limit: int,
    page_size: int,
    text_column: str,
    audio_column: str,
    id_column: str,
) -> Iterator[StreamingExample]:
    emitted = 0
    current_offset = offset
    while emitted < limit:
        length = min(page_size, limit - emitted)
        response = client.get(
            ROWS_ENDPOINT,
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": current_offset,
                "length": length,
            },
        )
        response.raise_for_status()
        rows = response.json().get("rows", [])
        if not rows:
            break

        for item in rows:
            row = item["row"]
            yield StreamingExample(
                row_idx=int(item["row_idx"]),
                example_id=str(row.get(id_column, item["row_idx"])),
                reference=str(row[text_column]),
                audio_url=extract_audio_url(row, audio_column),
            )
            emitted += 1
            if emitted >= limit:
                break
        current_offset += len(rows)


def stream_audio_bytes(client: httpx.Client, url: str) -> bytes:
    chunks = bytearray()
    with client.stream("GET", url) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            chunks.extend(chunk)
    return bytes(chunks)


def decode_audio_bytes(content: bytes, *, sample_rate: int) -> tuple[np.ndarray, int]:
    waveform, source_sample_rate = sf.read(
        io.BytesIO(content),
        dtype="float32",
        always_2d=False,
    )
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if int(source_sample_rate) != sample_rate:
        waveform = librosa.resample(
            np.asarray(waveform, dtype=np.float32),
            orig_sr=int(source_sample_rate),
            target_sr=sample_rate,
        )
    return np.asarray(waveform, dtype=np.float32), int(source_sample_rate)


def sample_metrics(reference: str, hypothesis: str) -> dict[str, float | str]:
    normalized_reference = normalize_for_wer(reference)
    normalized_hypothesis = normalize_for_wer(hypothesis)
    return {
        "reference_normalized": normalized_reference,
        "hypothesis_normalized": normalized_hypothesis,
        "wer": float(jiwer.wer(normalized_reference, normalized_hypothesis)),
        "cer": float(jiwer.cer(normalized_reference, normalized_hypothesis)),
    }


def realtime_metrics(audio_duration_sec: float, elapsed_sec: float) -> dict[str, float | None]:
    rtf = elapsed_sec / audio_duration_sec if audio_duration_sec else None
    rtfx = audio_duration_sec / elapsed_sec if elapsed_sec else None
    return {"rtf": rtf, "rtfx": rtfx, "speed_x": rtfx}


def write_eval_summary(
    summary_path: Path,
    *,
    backend: str,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    limit: int,
    model_ref: str,
    jsonl_path: Path,
    sample_reports: list[dict[str, Any]],
    normalized_references: list[str],
    normalized_hypotheses: list[str],
    wall_elapsed_sec: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_audio_sec = sum(float(report["audio_duration_sec"]) for report in sample_reports)
    total_sample_elapsed_sec = sum(float(report["elapsed_sec"]) for report in sample_reports)
    timing_keys = sorted(
        {
            key
            for report in sample_reports
            for key, value in report.items()
            if key.endswith("_elapsed_sec") and isinstance(value, int | float)
        }
    )
    timing_totals = {
        key: float(sum(float(report.get(key, 0.0)) for report in sample_reports))
        for key in timing_keys
    }
    summary: dict[str, Any] = {
        "backend": backend,
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "limit": limit,
        "completed": len(sample_reports),
        "model_ref": model_ref,
        "jsonl_path": str(jsonl_path),
        "wer": float(jiwer.wer(normalized_references, normalized_hypotheses))
        if sample_reports
        else None,
        "cer": float(jiwer.cer(normalized_references, normalized_hypotheses))
        if sample_reports
        else None,
        "mean_sample_wer": float(np.mean([report["wer"] for report in sample_reports]))
        if sample_reports
        else None,
        "total_audio_sec": total_audio_sec,
        "total_sample_elapsed_sec": total_sample_elapsed_sec,
        "wall_elapsed_sec": wall_elapsed_sec,
        "timing_totals_sec": timing_totals,
        **realtime_metrics(total_audio_sec, total_sample_elapsed_sec),
    }
    if extra:
        summary.update(extra)
    write_json(summary_path, summary)
    return summary


def write_jsonl_row(path: Path, report: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as jsonl:
        jsonl.write(json.dumps(report, sort_keys=True) + "\n")
        jsonl.flush()
