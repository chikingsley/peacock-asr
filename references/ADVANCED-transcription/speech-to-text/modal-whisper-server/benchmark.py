"""
Benchmark script for faster-whisper Modal servers.

Measures:
- Cold start time (first request after deployment)
- Warm inference time (average of subsequent requests)
- Real-Time Factor (RTF) = processing_time / audio_duration
- SSE streaming: time to first segment

Usage:
    uv run benchmark.py [--runs N] [--output FILE]
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from mutagen.mp3 import MP3
from openai import OpenAI


# Server endpoints - update these after deploying to your Modal workspace
# Format: https://<workspace>--faster-whisper-server-<classname>-serve.modal.run
ENDPOINTS = {
    "large-v3": "https://trelisresearch--faster-whisper-server-whisperlargev3-serve.modal.run",
    "large-v3-turbo": "https://trelisresearch--faster-whisper-server-whisperlargev3turbo-serve.modal.run",
}

# Sample files relative to repo root
REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLE_FILES = {
    "short": REPO_ROOT / "samples" / "llm_lingo_test.mp3",
    "long": REPO_ROOT / "samples" / "Run a Kokoro Text to Speech Server.mp3",
}


@dataclass
class BenchmarkResult:
    model: str
    file_name: str
    audio_duration: float
    run_number: int
    elapsed_seconds: float
    is_cold_start: bool
    sse_first_segment: float | None = None  # Time to first SSE segment (if streaming)

    @property
    def rtf(self) -> float:
        """Real-Time Factor: processing_time / audio_duration."""
        return self.elapsed_seconds / self.audio_duration


@dataclass
class SSEResult:
    """Result from SSE streaming benchmark."""
    model: str
    file_name: str
    audio_duration: float
    time_to_first_segment: float
    total_time: float
    num_segments: int


def get_audio_duration(file_path: Path) -> float:
    """Get audio duration in seconds using mutagen."""
    audio = MP3(file_path)
    return audio.info.length


def transcribe_file(client: OpenAI, model: str, file_path: Path) -> tuple[float, str]:
    """
    Transcribe a file and return (elapsed_time, transcript).
    """
    with open(file_path, "rb") as f:
        start_time = time.time()
        response = client.audio.transcriptions.create(
            model=model,
            file=f,
            language="en",
            response_format="text",
        )
        elapsed = time.time() - start_time

    return elapsed, response


def benchmark_sse_streaming(
    endpoint: str, model_key: str, file_key: str, file_path: Path
) -> SSEResult:
    """
    Benchmark SSE streaming transcription.
    Measures time to first segment and total completion time.
    """
    audio_duration = get_audio_duration(file_path)

    print(f"\n  SSE streaming test for {file_key}...")

    start_time = time.time()
    time_to_first_segment = None
    num_segments = 0

    with httpx.Client(timeout=300.0) as client:
        with open(file_path, "rb") as f:
            response = client.post(
                f"{endpoint}/v1/audio/transcriptions",
                files={"file": (file_path.name, f, "audio/mpeg")},
                data={"language": "en", "response_format": "json", "stream": "true"},
            )

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data.get("type") == "segment":
                        if time_to_first_segment is None:
                            time_to_first_segment = time.time() - start_time
                        num_segments += 1
                    elif data.get("type") == "done":
                        break

    total_time = time.time() - start_time

    result = SSEResult(
        model=model_key,
        file_name=file_key,
        audio_duration=audio_duration,
        time_to_first_segment=time_to_first_segment or total_time,
        total_time=total_time,
        num_segments=num_segments,
    )

    print(f"    First segment: {result.time_to_first_segment:.2f}s")
    print(f"    Total time: {result.total_time:.2f}s ({num_segments} segments)")

    return result


def benchmark_model(
    model_key: str,
    endpoint: str,
    file_key: str,
    file_path: Path,
    num_runs: int = 5,
) -> list[BenchmarkResult]:
    """
    Benchmark a single model with a single file.
    """
    client = OpenAI(api_key="EMPTY", base_url=f"{endpoint}/v1/")
    audio_duration = get_audio_duration(file_path)

    print(f"\nBenchmarking {model_key} with {file_key} ({audio_duration:.1f}s audio)")
    print("-" * 60)

    results = []
    for i in range(num_runs):
        is_cold = i == 0
        label = "cold start" if is_cold else f"run {i + 1}"

        elapsed, transcript = transcribe_file(client, model_key, file_path)
        rtf = elapsed / audio_duration

        result = BenchmarkResult(
            model=model_key,
            file_name=file_key,
            audio_duration=audio_duration,
            run_number=i + 1,
            elapsed_seconds=elapsed,
            is_cold_start=is_cold,
        )
        results.append(result)

        print(f"  {label}: {elapsed:.2f}s (RTF: {rtf:.3f})")

        # Show transcript snippet for first run only
        if i == 0:
            snippet = transcript[:100] + "..." if len(transcript) > 100 else transcript
            print(f"  Preview: {snippet}")

    return results


def format_results_markdown(
    results: list[BenchmarkResult], sse_results: list[SSEResult] | None = None
) -> str:
    """Format results as markdown."""
    from datetime import datetime

    lines = [
        "# Faster-Whisper Modal Benchmark Results",
        "",
        "## Test Environment",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "- GPU: H100 (80GB VRAM)",
        "- Platform: Modal",
        "- **GPU Memory Snapshots**: Enabled (fast cold starts)",
        "",
        "## Models Tested",
        "- **Whisper Large v3**: `Systran/faster-whisper-large-v3`",
        "- **Whisper Large v3 Turbo**: `deepdml/faster-whisper-large-v3-turbo-ct2`",
        "",
        "## Sample Files",
        "| File | Duration |",
        "|------|----------|",
    ]

    # Get unique files and durations
    files = {}
    for r in results:
        if r.file_name not in files:
            files[r.file_name] = r.audio_duration

    for name, duration in files.items():
        lines.append(f"| {name} | {duration:.1f}s |")

    lines.append("")

    # Group results by model
    models = {}
    for r in results:
        if r.model not in models:
            models[r.model] = {}
        if r.file_name not in models[r.model]:
            models[r.model][r.file_name] = []
        models[r.model][r.file_name].append(r)

    # Results for each model
    for model_key, file_results in models.items():
        lines.append(f"## {model_key.replace('-', ' ').title()}")
        lines.append("")
        lines.append("| Metric | " + " | ".join(files.keys()) + " |")
        lines.append("|--------|" + "|".join(["----------"] * len(files)) + "|")

        # Cold start row
        cold_values = []
        for file_key in files.keys():
            cold_result = next(
                (r for r in file_results.get(file_key, []) if r.is_cold_start), None
            )
            if cold_result:
                cold_values.append(f"{cold_result.elapsed_seconds:.2f}s")
            else:
                cold_values.append("N/A")
        lines.append("| Cold Start | " + " | ".join(cold_values) + " |")

        # Warm average row
        warm_values = []
        for file_key in files.keys():
            warm_results = [
                r for r in file_results.get(file_key, []) if not r.is_cold_start
            ]
            if warm_results:
                avg = sum(r.elapsed_seconds for r in warm_results) / len(warm_results)
                warm_values.append(f"{avg:.2f}s")
            else:
                warm_values.append("N/A")
        lines.append("| Warm Avg | " + " | ".join(warm_values) + " |")

        # Warm RTF row
        rtf_values = []
        for file_key in files.keys():
            warm_results = [
                r for r in file_results.get(file_key, []) if not r.is_cold_start
            ]
            if warm_results:
                avg_rtf = sum(r.rtf for r in warm_results) / len(warm_results)
                rtf_values.append(f"{avg_rtf:.3f}")
            else:
                rtf_values.append("N/A")
        lines.append("| Warm RTF | " + " | ".join(rtf_values) + " |")

        lines.append("")

    # Add SSE streaming results if available
    if sse_results:
        lines.extend([
            "## SSE Streaming Performance",
            "",
            "| Model | File | Time to First Segment | Total Time | Segments |",
            "|-------|------|----------------------|------------|----------|",
        ])
        for sse in sse_results:
            lines.append(
                f"| {sse.model} | {sse.file_name} | {sse.time_to_first_segment:.2f}s | {sse.total_time:.2f}s | {sse.num_segments} |"
            )
        lines.append("")

    # Add explanation
    lines.extend([
        "## Notes",
        "",
        "- **RTF (Real-Time Factor)**: `processing_time / audio_duration`",
        "  - RTF < 1.0 means faster than real-time",
        "  - RTF = 0.1 means 10x faster than real-time (10s of audio in 1s)",
        "- **Cold Start**: First request after container startup",
        "  - With GPU snapshots: ~1-5s (model pre-loaded in snapshot)",
        "  - Without GPU snapshots: ~30-60s (model loaded from disk)",
        "- **Warm Avg**: Average of subsequent requests (model already loaded)",
        "- **SSE Streaming**: Segments returned as they're generated via Server-Sent Events",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark faster-whisper Modal servers")
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of runs per model/file (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.md",
        help="Output markdown file (default: benchmark_results.md)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(ENDPOINTS.keys()),
        help="Test only a specific model",
    )
    parser.add_argument(
        "--file",
        type=str,
        choices=list(SAMPLE_FILES.keys()),
        help="Test only a specific file",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Include SSE streaming benchmarks",
    )
    args = parser.parse_args()

    # Validate sample files exist
    for key, path in SAMPLE_FILES.items():
        if not path.exists():
            print(f"Warning: Sample file not found: {path}")
            print("Make sure to place audio files in the samples/ folder")

    # Filter endpoints and files if specified
    endpoints = {args.model: ENDPOINTS[args.model]} if args.model else ENDPOINTS
    sample_files = {args.file: SAMPLE_FILES[args.file]} if args.file else SAMPLE_FILES

    all_results = []
    sse_results = []

    for model_key, endpoint in endpoints.items():
        print(f"\n{'=' * 60}")
        print(f"Testing: {model_key}")
        print(f"Endpoint: {endpoint}")
        print(f"{'=' * 60}")

        for file_key, file_path in sample_files.items():
            if not file_path.exists():
                print(f"Skipping {file_key}: file not found")
                continue

            try:
                results = benchmark_model(
                    model_key, endpoint, file_key, file_path, args.runs
                )
                all_results.extend(results)

                # Run SSE streaming benchmark if requested
                if args.sse:
                    try:
                        sse_result = benchmark_sse_streaming(
                            endpoint, model_key, file_key, file_path
                        )
                        sse_results.append(sse_result)
                    except Exception as e:
                        print(f"  SSE benchmark error: {e}")

            except Exception as e:
                print(f"Error benchmarking {model_key} with {file_key}: {e}")

    # Write results
    if all_results:
        output_path = Path(__file__).parent / args.output
        markdown = format_results_markdown(all_results, sse_results if sse_results else None)
        output_path.write_text(markdown)
        print(f"\nResults written to: {output_path}")
        print("\n" + markdown)
    else:
        print("\nNo results to report. Check that servers are deployed and sample files exist.")


if __name__ == "__main__":
    main()
