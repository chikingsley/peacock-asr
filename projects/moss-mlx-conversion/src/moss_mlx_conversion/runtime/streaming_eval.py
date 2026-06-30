from __future__ import annotations

import argparse
import io
import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import jiwer
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import AutoTokenizer

from moss_mlx_conversion.dump import ensure_dir, write_json
from moss_mlx_conversion.mlx_compat import mx
from moss_mlx_conversion.paths import ARTIFACTS_DIR, MLX_DIR
from moss_mlx_conversion.processor import MossProcessor
from moss_mlx_conversion.runtime.transcribe import load_converted_model, transcribe_waveform

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class StreamingExample:
    row_idx: int
    example_id: str
    reference: str
    audio_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate converted MOSS MLX on HF streamed audio."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MLX_DIR / "MOSS-Transcribe-preview-2B-bf16",
    )
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "evals" / "librispeech-test-clean-streaming-20",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    jsonl_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    started = time.perf_counter()

    model_dir = args.model_dir.resolve()
    model, config = load_converted_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    processor = MossProcessor(
        tokenizer,
        template_path=model_dir / "chat_template_default.py",
        enable_time_marker=False,
    )

    normalized_references: list[str] = []
    normalized_hypotheses: list[str] = []
    sample_reports: list[dict[str, Any]] = []

    client = httpx.Client(timeout=httpx.Timeout(args.timeout_sec))
    try:
        examples = iter_hf_rows(
            client,
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            offset=args.offset,
            limit=args.limit,
            page_size=args.page_size,
            text_column=args.text_column,
            audio_column=args.audio_column,
            id_column=args.id_column,
        )
        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            for example in tqdm(examples, total=args.limit, desc="streaming eval"):
                report = evaluate_one(
                    client=client,
                    example=example,
                    model=model,
                    config=config,
                    processor=processor,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    prefill_step_size=args.prefill_step_size,
                )
                normalized_references.append(str(report["reference_normalized"]))
                normalized_hypotheses.append(str(report["hypothesis_normalized"]))
                sample_reports.append(report)
                jsonl.write(json.dumps(report, sort_keys=True) + "\n")
                jsonl.flush()
                print(
                    f"{example.row_idx} {example.example_id}: "
                    f"wer={float(report['wer']):.3f} "
                    f"rtf={float(report['rtf']):.2f} "
                    f"hyp={report['hypothesis']}"
                )
    finally:
        client.close()

    write_summary(
        summary_path,
        args=args,
        model_dir=model_dir,
        jsonl_path=jsonl_path,
        sample_reports=sample_reports,
        normalized_references=normalized_references,
        normalized_hypotheses=normalized_hypotheses,
        wall_elapsed_sec=time.perf_counter() - started,
    )


def evaluate_one(
    *,
    client: httpx.Client,
    example: StreamingExample,
    model: Any,
    config: Any,
    processor: MossProcessor,
    tokenizer: Any,
    max_new_tokens: int,
    prefill_step_size: int,
) -> dict[str, Any]:
    sample_started = time.perf_counter()
    audio_bytes = stream_audio_bytes(client, example.audio_url)
    waveform, source_sample_rate = decode_audio_bytes(audio_bytes, sample_rate=config.sample_rate)
    audio_duration_sec = float(len(waveform) / config.sample_rate)
    result = transcribe_waveform(
        model=model,
        config=config,
        processor=processor,
        tokenizer=tokenizer,
        waveform=waveform,
        max_new_tokens=max_new_tokens,
        prefill_step_size=prefill_step_size,
    )
    mx.clear_cache()

    metrics = sample_metrics(example.reference, result.transcript)
    elapsed_sec = time.perf_counter() - sample_started
    return {
        "row_idx": example.row_idx,
        "id": example.example_id,
        "reference": example.reference,
        "hypothesis": result.transcript,
        "reference_normalized": metrics["reference_normalized"],
        "hypothesis_normalized": metrics["hypothesis_normalized"],
        "wer": metrics["wer"],
        "cer": metrics["cer"],
        "audio_duration_sec": audio_duration_sec,
        "elapsed_sec": elapsed_sec,
        "rtf": elapsed_sec / audio_duration_sec if audio_duration_sec else None,
        "speed_x": audio_duration_sec / elapsed_sec if elapsed_sec else None,
        "source_sample_rate": source_sample_rate,
        "audio_bytes": len(audio_bytes),
        "prompt_length": result.prompt_length,
        "audio_placeholder_count": result.audio_placeholder_count,
        "generated_token_count": result.generated_token_count,
        "generation_elapsed_sec": result.generation_elapsed_sec,
        "generated_tokens_per_sec": result.generated_token_count / result.generation_elapsed_sec
        if result.generation_elapsed_sec
        else None,
        "first_5_new_ids": result.generated_ids[:5],
    }


def write_summary(
    summary_path: Path,
    *,
    args: argparse.Namespace,
    model_dir: Path,
    jsonl_path: Path,
    sample_reports: list[dict[str, Any]],
    normalized_references: list[str],
    normalized_hypotheses: list[str],
    wall_elapsed_sec: float,
) -> None:
    total_audio_sec = sum(float(report["audio_duration_sec"]) for report in sample_reports)
    total_sample_elapsed_sec = sum(float(report["elapsed_sec"]) for report in sample_reports)
    summary = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "completed": len(sample_reports),
        "model_dir": str(model_dir),
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
        "rtf": total_sample_elapsed_sec / total_audio_sec if total_audio_sec else None,
        "speed_x": total_audio_sec / total_sample_elapsed_sec if total_sample_elapsed_sec else None,
        "wall_elapsed_sec": wall_elapsed_sec,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"streaming eval summary: {summary_path}")


if __name__ == "__main__":
    main()
