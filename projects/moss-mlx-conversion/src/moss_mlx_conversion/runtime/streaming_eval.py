from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm
from transformers import AutoTokenizer

from moss_mlx_conversion.dump import ensure_dir
from moss_mlx_conversion.mlx_compat import mx
from moss_mlx_conversion.paths import ARTIFACTS_DIR, MLX_DIR
from moss_mlx_conversion.processor import MossProcessor
from moss_mlx_conversion.runtime.eval import (
    StreamingExample,
    decode_audio_bytes,
    iter_hf_rows,
    realtime_metrics,
    sample_metrics,
    stream_audio_bytes,
    write_eval_summary,
)
from moss_mlx_conversion.runtime.transcribe import load_converted_model, transcribe_waveform


def backend_name(model_dir: Path) -> str:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return "mlx"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config_data.get("quantization")
    if not isinstance(quantization, dict):
        return "mlx-bf16"
    bits = quantization.get("bits", "unknown")
    group_size = quantization.get("group_size", "unknown")
    scope = quantization.get("scope", "all")
    return f"mlx-{scope}-{bits}bit-g{group_size}"


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
    parser.add_argument(
        "--generation-mode",
        choices=["fast-greedy", "mlx-lm"],
        default="mlx-lm",
    )
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--summary-every",
        type=int,
        default=50,
        help="Write partial-summary.json after this many newly processed rows. Use 0 to disable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "evals" / "librispeech-test-clean-streaming-20",
    )
    return parser.parse_args()


def load_existing_reports(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    reports: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as jsonl:
        for line in jsonl:
            stripped = line.strip()
            if stripped:
                reports.append(json.loads(stripped))
    return reports


def normalized_pairs(
    sample_reports: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    return (
        [str(report["reference_normalized"]) for report in sample_reports],
        [str(report["hypothesis_normalized"]) for report in sample_reports],
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    jsonl_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    partial_summary_path = output_dir / "partial-summary.json"
    started = time.perf_counter()

    model_dir = args.model_dir.resolve()
    model, config = load_converted_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    processor = MossProcessor(
        tokenizer,
        template_path=model_dir / "chat_template_default.py",
        enable_time_marker=False,
    )

    sample_reports = load_existing_reports(jsonl_path) if args.resume else []
    seen_row_indices = {int(report["row_idx"]) for report in sample_reports}
    seen_ids = {str(report["id"]) for report in sample_reports}
    rows_written_this_run = 0

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
        jsonl_mode = "a" if args.resume else "w"
        with jsonl_path.open(jsonl_mode, encoding="utf-8") as jsonl:
            for example in tqdm(examples, total=args.limit, desc="streaming eval"):
                if example.row_idx in seen_row_indices or example.example_id in seen_ids:
                    continue
                report = evaluate_one(
                    client=client,
                    example=example,
                    model=model,
                    config=config,
                    processor=processor,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    prefill_step_size=args.prefill_step_size,
                    generation_mode=args.generation_mode,
                )
                sample_reports.append(report)
                jsonl.write(json.dumps(report, sort_keys=True) + "\n")
                jsonl.flush()
                rows_written_this_run += 1
                if not args.quiet:
                    print_sample(report)
                if args.summary_every and rows_written_this_run % args.summary_every == 0:
                    write_current_summary(
                        partial_summary_path,
                        args=args,
                        model_dir=model_dir,
                        jsonl_path=jsonl_path,
                        sample_reports=sample_reports,
                        started=started,
                    )
    finally:
        client.close()

    summary = write_current_summary(
        summary_path,
        args=args,
        model_dir=model_dir,
        jsonl_path=jsonl_path,
        sample_reports=sample_reports,
        started=started,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"streaming eval summary: {summary_path}")


def write_current_summary(
    summary_path: Path,
    *,
    args: argparse.Namespace,
    model_dir: Path,
    jsonl_path: Path,
    sample_reports: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    normalized_references, normalized_hypotheses = normalized_pairs(sample_reports)
    return write_eval_summary(
        summary_path,
        backend=backend_name(model_dir),
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
        model_ref=str(model_dir),
        jsonl_path=jsonl_path,
        sample_reports=sample_reports,
        normalized_references=normalized_references,
        normalized_hypotheses=normalized_hypotheses,
        wall_elapsed_sec=time.perf_counter() - started,
        extra={"resumable": True},
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
    generation_mode: str,
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
        generation_mode=generation_mode,
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
        **realtime_metrics(audio_duration_sec, elapsed_sec),
        "source_sample_rate": source_sample_rate,
        "audio_bytes": len(audio_bytes),
        "prompt_length": result.prompt_length,
        "audio_placeholder_count": result.audio_placeholder_count,
        "generated_token_count": result.generated_token_count,
        "generation_mode": result.generation_mode,
        **result.timings,
        "generated_tokens_per_sec": result.generated_token_count / result.generation_elapsed_sec
        if result.generation_elapsed_sec
        else None,
        "first_5_new_ids": result.generated_ids[:5],
    }


def print_sample(report: dict[str, Any]) -> None:
    print(
        f"{report['row_idx']} {report['id']}: "
        f"wer={float(report['wer']):.3f} "
        f"rtfx={float(report['rtfx']):.2f} "
        f"hyp={report['hypothesis']}"
    )


if __name__ == "__main__":
    main()
