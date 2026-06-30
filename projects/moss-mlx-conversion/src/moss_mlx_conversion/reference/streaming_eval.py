from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx
import torch
from tqdm import tqdm

from moss_mlx_conversion import DEFAULT_MODEL_ID
from moss_mlx_conversion.dump import ensure_dir
from moss_mlx_conversion.paths import ARTIFACTS_DIR
from moss_mlx_conversion.reference.reference import (
    build_processor,
    load_model,
    move_inputs_to_device,
    pick_device,
    pick_dtype,
)
from moss_mlx_conversion.runtime.eval import (
    StreamingExample,
    decode_audio_bytes,
    iter_hf_rows,
    realtime_metrics,
    sample_metrics,
    stream_audio_bytes,
    write_eval_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate upstream MOSS PyTorch on HF streamed audio."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
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
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "evals" / "librispeech-test-clean-pytorch-20",
    )
    return parser.parse_args()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    jsonl_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    jsonl_path.write_text("", encoding="utf-8")
    started = time.perf_counter()

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)
    tokenizer, processor, _template_path = build_processor(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    del tokenizer
    model = load_model(
        args.model_id,
        revision=args.revision,
        dtype=dtype,
        device=device,
        local_files_only=args.local_files_only,
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
        with jsonl_path.open("a", encoding="utf-8") as jsonl:
            for example in tqdm(examples, total=args.limit, desc="pytorch streaming eval"):
                report = evaluate_one(
                    client=client,
                    example=example,
                    model=model,
                    processor=processor,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
                normalized_references.append(str(report["reference_normalized"]))
                normalized_hypotheses.append(str(report["hypothesis_normalized"]))
                sample_reports.append(report)
                jsonl.write(json.dumps(report, sort_keys=True) + "\n")
                jsonl.flush()
                print_sample(report)
    finally:
        client.close()

    summary = write_eval_summary(
        summary_path,
        backend="pytorch-bf16" if args.dtype == "bf16" else f"pytorch-{args.dtype}",
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
        model_ref=f"{args.model_id}@{args.revision}",
        jsonl_path=jsonl_path,
        sample_reports=sample_reports,
        normalized_references=normalized_references,
        normalized_hypotheses=normalized_hypotheses,
        wall_elapsed_sec=time.perf_counter() - started,
        extra={"device": str(device), "dtype": args.dtype},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"PyTorch streaming eval summary: {summary_path}")


def evaluate_one(
    *,
    client: httpx.Client,
    example: StreamingExample,
    model: Any,
    processor: Any,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    sample_started = time.perf_counter()
    audio_started = time.perf_counter()
    audio_bytes = stream_audio_bytes(client, example.audio_url)
    waveform, source_sample_rate = decode_audio_bytes(audio_bytes, sample_rate=16_000)
    audio_decode_elapsed_sec = time.perf_counter() - audio_started
    audio_duration_sec = float(len(waveform) / 16_000)

    processor_started = time.perf_counter()
    inputs_cpu = processor(audio=waveform, return_tensors="pt")
    processor_elapsed_sec = time.perf_counter() - processor_started

    move_started = time.perf_counter()
    inputs = move_inputs_to_device(dict(inputs_cpu), device)
    inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    synchronize_device(device)
    input_move_elapsed_sec = time.perf_counter() - move_started

    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=[processor.end_token_id],
            return_dict_in_generate=True,
        )
    synchronize_device(device)
    generation_elapsed_sec = time.perf_counter() - generation_started

    sequences = generated.sequences
    new_ids_tensor = sequences[:, inputs["input_ids"].shape[1] :]
    generated_ids = new_ids_tensor.detach().cpu().reshape(-1).tolist()
    transcript = processor.batch_decode(new_ids_tensor, skip_special_tokens=True)[0].strip()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    metrics = sample_metrics(example.reference, transcript)
    elapsed_sec = time.perf_counter() - sample_started
    return {
        "row_idx": example.row_idx,
        "id": example.example_id,
        "reference": example.reference,
        "hypothesis": transcript,
        "reference_normalized": metrics["reference_normalized"],
        "hypothesis_normalized": metrics["hypothesis_normalized"],
        "wer": metrics["wer"],
        "cer": metrics["cer"],
        "audio_duration_sec": audio_duration_sec,
        "elapsed_sec": elapsed_sec,
        **realtime_metrics(audio_duration_sec, elapsed_sec),
        "source_sample_rate": source_sample_rate,
        "audio_bytes": len(audio_bytes),
        "prompt_length": int(inputs["input_ids"].shape[1]),
        "audio_placeholder_count": int(inputs["audio_input_mask"].sum().item()),
        "generated_token_count": len(generated_ids),
        "audio_decode_elapsed_sec": audio_decode_elapsed_sec,
        "processor_elapsed_sec": processor_elapsed_sec,
        "input_move_elapsed_sec": input_move_elapsed_sec,
        "generation_elapsed_sec": generation_elapsed_sec,
        "generated_tokens_per_sec": len(generated_ids) / generation_elapsed_sec
        if generation_elapsed_sec
        else None,
        "first_5_new_ids": generated_ids[:5],
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
