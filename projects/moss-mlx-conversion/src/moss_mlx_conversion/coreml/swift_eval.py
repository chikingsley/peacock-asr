from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf

from moss_mlx_conversion.dump import ensure_dir, write_json
from moss_mlx_conversion.paths import ARTIFACTS_DIR, PROJECT_ROOT
from moss_mlx_conversion.runtime.eval import (
    StreamingExample,
    decode_audio_bytes,
    iter_hf_rows,
    realtime_metrics,
    stream_audio_bytes,
    write_eval_summary,
)


@dataclass(frozen=True)
class SwiftEvalPaths:
    output_dir: Path
    audio_dir: Path
    reference_dir: Path
    swift_report_dir: Path
    predictions_path: Path
    summary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small Swift/CoreML MOSS eval over streamed HF audio rows."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-audio-sec", type=float, default=30.0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--audio-max-frames", type=int, default=3000)
    parser.add_argument("--compute-units", default="cpu-gpu")
    parser.add_argument(
        "--swift-package-path",
        type=Path,
        default=Path("swift/MossCoreMLFixture"),
    )
    parser.add_argument(
        "--packages-dir",
        type=Path,
        default=Path("coreml/build"),
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("artifacts/coreml/moss_swift_fixture_compact.json"),
    )
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        default=None,
        help="Fixture-free Swift runtime manifest for prompt/model constants.",
    )
    parser.add_argument(
        "--audio-package",
        default="compiled_audio_30s/moss_audio_encoder_adapter_30s_padded.mlmodelc",
    )
    parser.add_argument(
        "--decoder-package",
        default="compiled_stateful/moss_decoder_stateful_fused.mlmodelc",
    )
    parser.add_argument("--prefill-cache-package")
    parser.add_argument("--prefill-cache-seq-len", type=int)
    parser.add_argument("--step-package")
    parser.add_argument("--cache-len", type=int, default=768)
    parser.add_argument(
        "--swift-batch",
        action="store_true",
        help="Call the Swift runner once with a JSONL manifest and keep CoreML models loaded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "evals" / "librispeech-test-clean-swift-coreml-1",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolved_under(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def eval_paths(output_dir: Path) -> SwiftEvalPaths:
    return SwiftEvalPaths(
        output_dir=ensure_dir(output_dir),
        audio_dir=ensure_dir(output_dir / "audio"),
        reference_dir=ensure_dir(output_dir / "reference"),
        swift_report_dir=ensure_dir(output_dir / "swift-json"),
        predictions_path=output_dir / "predictions.jsonl",
        summary_path=output_dir / "summary.json",
    )


def safe_stem(example: StreamingExample) -> str:
    safe_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in example.example_id
    )
    return f"{example.row_idx:06d}-{safe_id}"


def write_row_inputs(
    *,
    client: httpx.Client,
    example: StreamingExample,
    paths: SwiftEvalPaths,
    sample_rate: int = 16_000,
) -> dict[str, Any]:
    stem = safe_stem(example)
    audio_path = paths.audio_dir / f"{stem}.wav"
    reference_path = paths.reference_dir / f"{stem}.txt"
    metadata_path = paths.reference_dir / f"{stem}.json"

    audio_bytes = stream_audio_bytes(client, example.audio_url)
    waveform, source_sample_rate = decode_audio_bytes(audio_bytes, sample_rate=sample_rate)
    sf.write(audio_path, waveform, sample_rate, subtype="PCM_16")
    reference_path.write_text(example.reference + "\n", encoding="utf-8")

    metadata = {
        "row_idx": example.row_idx,
        "id": example.example_id,
        "reference": example.reference,
        "audio_url": example.audio_url,
        "audio_path": str(audio_path),
        "reference_path": str(reference_path),
        "source_sample_rate": source_sample_rate,
        "sample_rate": sample_rate,
        "audio_samples": len(waveform),
        "audio_duration_sec": float(len(waveform) / sample_rate),
        "audio_bytes_streamed": len(audio_bytes),
    }
    write_json(metadata_path, metadata)
    return metadata


def swift_command(
    *,
    project_root: Path,
    args: argparse.Namespace,
    audio_path: Path,
    reference_path: Path,
    output_path: Path,
) -> list[str]:
    command = [
        "swift",
        "run",
        "--package-path",
        str(resolved_under(project_root, args.swift_package_path)),
        "-c",
        "release",
        "moss-coreml-fixture",
        "--packages-dir",
        str(resolved_under(project_root, args.packages_dir)),
        "--fixture",
        str(resolved_under(project_root, args.fixture)),
        "--audio",
        str(audio_path),
        "--audio-max-frames",
        str(args.audio_max_frames),
        "--audio-package",
        str(args.audio_package),
        "--decoder-package",
        str(args.decoder_package),
        "--compute-units",
        str(args.compute_units),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--reference-text-file",
        str(reference_path),
        "--output",
        str(output_path),
    ]
    if args.runtime_manifest is not None:
        command.extend(
            [
                "--runtime-manifest",
                str(resolved_under(project_root, args.runtime_manifest)),
            ]
        )
    if args.prefill_cache_package or args.step_package:
        if not args.prefill_cache_package or not args.step_package:
            raise ValueError("pass both --prefill-cache-package and --step-package")
        command.extend(
            [
                "--prefill-cache-package",
                str(args.prefill_cache_package),
                *(
                    [
                        "--prefill-cache-seq-len",
                        str(args.prefill_cache_seq_len),
                    ]
                    if args.prefill_cache_seq_len is not None
                    else []
                ),
                "--step-package",
                str(args.step_package),
                "--cache-len",
                str(args.cache_len),
            ]
        )
    return command


def swift_batch_command(
    *,
    project_root: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    batch_output_path: Path,
) -> list[str]:
    command = [
        "swift",
        "run",
        "--package-path",
        str(resolved_under(project_root, args.swift_package_path)),
        "-c",
        "release",
        "moss-coreml-fixture",
        "--packages-dir",
        str(resolved_under(project_root, args.packages_dir)),
        "--fixture",
        str(resolved_under(project_root, args.fixture)),
        "--audio-max-frames",
        str(args.audio_max_frames),
        "--audio-package",
        str(args.audio_package),
        "--decoder-package",
        str(args.decoder_package),
        "--compute-units",
        str(args.compute_units),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--batch-manifest",
        str(manifest_path),
        "--batch-output-jsonl",
        str(batch_output_path),
    ]
    if args.runtime_manifest is not None:
        command.extend(
            [
                "--runtime-manifest",
                str(resolved_under(project_root, args.runtime_manifest)),
            ]
        )
    if args.prefill_cache_package or args.step_package:
        if not args.prefill_cache_package or not args.step_package:
            raise ValueError("pass both --prefill-cache-package and --step-package")
        command.extend(
            [
                "--prefill-cache-package",
                str(args.prefill_cache_package),
                *(
                    [
                        "--prefill-cache-seq-len",
                        str(args.prefill_cache_seq_len),
                    ]
                    if args.prefill_cache_seq_len is not None
                    else []
                ),
                "--step-package",
                str(args.step_package),
                "--cache-len",
                str(args.cache_len),
            ]
        )
    return command


def run_swift_report(
    *,
    project_root: Path,
    args: argparse.Namespace,
    audio_path: Path,
    reference_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    command = swift_command(
        project_root=project_root,
        args=args,
        audio_path=audio_path,
        reference_path=reference_path,
        output_path=output_path,
    )
    started = time.perf_counter()
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    wall_elapsed_sec = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "swift runner failed with exit code "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    report["swift_stdout_tail"] = completed.stdout[-2000:]
    report["swift_stderr_tail"] = completed.stderr[-2000:]
    report["swift_process_wall_sec"] = wall_elapsed_sec
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_swift_batch(
    *,
    project_root: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    batch_output_path: Path,
) -> dict[str, Any]:
    command = swift_batch_command(
        project_root=project_root,
        args=args,
        manifest_path=manifest_path,
        batch_output_path=batch_output_path,
    )
    started = time.perf_counter()
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    wall_elapsed_sec = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "swift batch runner failed with exit code "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {
        "swift_stdout_tail": completed.stdout[-2000:],
        "swift_stderr_tail": completed.stderr[-2000:],
        "swift_batch_process_wall_sec": wall_elapsed_sec,
    }


def prediction_report(
    *,
    example: StreamingExample,
    input_metadata: dict[str, Any],
    swift_report: dict[str, Any],
    swift_report_path: Path,
) -> dict[str, Any]:
    elapsed_sec = float(swift_report["timing_seconds"]["total"])
    audio_duration_sec = float(input_metadata["audio_duration_sec"])
    return {
        "row_idx": example.row_idx,
        "id": example.example_id,
        "reference": example.reference,
        "hypothesis": swift_report["generated_text"],
        "reference_normalized": swift_report["normalized_expected_text"],
        "hypothesis_normalized": swift_report["normalized_generated_text"],
        "wer": float(swift_report["normalized_wer"]),
        "cer": float(swift_report["normalized_cer"]),
        "raw_wer": float(swift_report["raw_wer"]),
        "raw_cer": float(swift_report["raw_cer"]),
        "audio_duration_sec": audio_duration_sec,
        "elapsed_sec": elapsed_sec,
        **realtime_metrics(audio_duration_sec, elapsed_sec),
        "source_sample_rate": input_metadata["source_sample_rate"],
        "audio_path": input_metadata["audio_path"],
        "reference_path": input_metadata["reference_path"],
        "swift_report_path": str(swift_report_path),
        "prompt_length": int(swift_report["prompt_len"]),
        "audio_placeholder_count": int(swift_report["audio_token_count"]),
        "generated_token_count": len(swift_report["generated_ids"]),
        "stopped_on_eos": bool(swift_report["stopped_on_eos"]),
        "decoder_mode": swift_report.get("decoder_mode", "stateful"),
        "audio_frontend_elapsed_sec": float(swift_report["timing_seconds"]["audio_frontend"]),
        "audio_encoder_adapter_elapsed_sec": float(
            swift_report["timing_seconds"]["audio_encoder_adapter"]
        ),
        "decoder_prefill_elapsed_sec": float(
            swift_report["timing_seconds"]["stateful_decoder_prefill"]
        ),
        "decoder_decode_elapsed_sec": float(
            swift_report["timing_seconds"]["stateful_decoder_decode"]
        ),
        "token_embedding_prompt_elapsed_sec": float(
            swift_report["timing_seconds"]["token_embedding_prompt"]
        ),
        "token_embedding_decode_elapsed_sec": float(
            swift_report["timing_seconds"]["token_embedding_decode"]
        ),
        "swift_process_wall_sec": float(
            swift_report.get("swift_process_wall_sec", swift_report.get("row_wall_sec", 0.0))
        ),
        **(
            {
                "swift_batch_process_wall_sec": float(
                    swift_report["swift_batch_process_wall_sec"]
                )
            }
            if "swift_batch_process_wall_sec" in swift_report
            else {}
        ),
    }


def load_existing_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as jsonl:
        return [json.loads(line) for line in jsonl if line.strip()]


def append_prediction(path: Path, report: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as jsonl:
        jsonl.write(json.dumps(report, sort_keys=True) + "\n")
        jsonl.flush()


def print_sample(report: dict[str, Any]) -> None:
    print(
        f"{report['row_idx']} {report['id']}: "
        f"wer={float(report['wer']):.3f} "
        f"rtfx={float(report['rtfx']):.2f} "
        f"tokens={report['generated_token_count']} "
        f"hyp={report['hypothesis']}"
    )


def write_batch_manifest(
    manifest_path: Path,
    rows: list[tuple[StreamingExample, dict[str, Any], Path]],
) -> None:
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for example, input_metadata, swift_report_path in rows:
            record = {
                "row_idx": example.row_idx,
                "id": example.example_id,
                "audio": str(Path(str(input_metadata["audio_path"])).resolve()),
                "reference_text_file": str(
                    Path(str(input_metadata["reference_path"])).resolve()
                ),
                "output": str(swift_report_path.resolve()),
            }
            manifest.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    paths = eval_paths(args.output_dir)
    if not args.resume:
        paths.predictions_path.write_text("", encoding="utf-8")
    started = time.perf_counter()

    sample_reports = load_existing_predictions(paths.predictions_path) if args.resume else []
    seen_row_indices = {int(report["row_idx"]) for report in sample_reports}
    seen_ids = {str(report["id"]) for report in sample_reports}

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
        pending_batch_rows: list[tuple[StreamingExample, dict[str, Any], Path]] = []
        for example in examples:
            if example.row_idx in seen_row_indices or example.example_id in seen_ids:
                continue
            input_metadata = write_row_inputs(client=client, example=example, paths=paths)
            if float(input_metadata["audio_duration_sec"]) > float(args.max_audio_sec):
                raise ValueError(
                    f"row {example.row_idx} is {input_metadata['audio_duration_sec']:.2f}s, "
                    f"which exceeds --max-audio-sec {args.max_audio_sec}"
                )
            stem = safe_stem(example)
            swift_report_path = paths.swift_report_dir / f"{stem}.json"
            if args.swift_batch:
                pending_batch_rows.append((example, input_metadata, swift_report_path))
                continue
            swift_report = run_swift_report(
                project_root=project_root,
                args=args,
                audio_path=Path(str(input_metadata["audio_path"])).resolve(),
                reference_path=Path(str(input_metadata["reference_path"])).resolve(),
                output_path=swift_report_path.resolve(),
            )
            report = prediction_report(
                example=example,
                input_metadata=input_metadata,
                swift_report=swift_report,
                swift_report_path=swift_report_path,
            )
            sample_reports.append(report)
            append_prediction(paths.predictions_path, report)
            print_sample(report)
        if args.swift_batch and pending_batch_rows:
            manifest_path = paths.output_dir / "swift-batch-manifest.jsonl"
            batch_output_path = paths.output_dir / "swift-batch-results.jsonl"
            write_batch_manifest(manifest_path, pending_batch_rows)
            batch_metadata = run_swift_batch(
                project_root=project_root,
                args=args,
                manifest_path=manifest_path.resolve(),
                batch_output_path=batch_output_path.resolve(),
            )
            for example, input_metadata, swift_report_path in pending_batch_rows:
                swift_report = json.loads(swift_report_path.read_text(encoding="utf-8"))
                swift_report.update(batch_metadata)
                swift_report_path.write_text(
                    json.dumps(swift_report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                report = prediction_report(
                    example=example,
                    input_metadata=input_metadata,
                    swift_report=swift_report,
                    swift_report_path=swift_report_path,
                )
                sample_reports.append(report)
                append_prediction(paths.predictions_path, report)
                print_sample(report)
    finally:
        client.close()

    summary = write_eval_summary(
        paths.summary_path,
        backend="swift-coreml-moss-30s-padded",
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
        model_ref=str(resolved_under(project_root, args.packages_dir)),
        jsonl_path=paths.predictions_path,
        sample_reports=sample_reports,
        normalized_references=[str(report["reference_normalized"]) for report in sample_reports],
        normalized_hypotheses=[str(report["hypothesis_normalized"]) for report in sample_reports],
        wall_elapsed_sec=time.perf_counter() - started,
        extra={
            "max_audio_sec": args.max_audio_sec,
            "max_new_tokens": args.max_new_tokens,
            "compute_units": args.compute_units,
            "decoder_package": args.decoder_package,
            "runtime_manifest": str(args.runtime_manifest)
            if args.runtime_manifest is not None
            else None,
            "prefill_cache_package": args.prefill_cache_package,
            "prefill_cache_seq_len": args.prefill_cache_seq_len,
            "step_package": args.step_package,
            "cache_len": args.cache_len,
            "swift_batch": args.swift_batch,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Swift/CoreML eval summary: {paths.summary_path}")


if __name__ == "__main__":
    main()
