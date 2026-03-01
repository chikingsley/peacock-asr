#!/usr/bin/env python3
# /// script
# dependencies = [
#   "httpx>=0.25.0",
#   "datasets[audio]>=3.4.1",
#   "soundfile",
#   "numpy",
#   "torchcodec",
# ]
# ///
"""
vLLM Voxtral Throughput Test

Tests single-request and concurrent-request throughput for the Voxtral
vLLM serving endpoint. Reports tokens/second and latency.

Usage:
    # Test against deployed endpoint
    uv run test_throughput.py --url https://your-modal-url.modal.run

    # Test with concurrent requests
    uv run test_throughput.py --url https://your-modal-url.modal.run --concurrent 4
"""

import argparse
import asyncio
import time
import tempfile
import numpy as np


def extract_audio(audio_data):
    """Extract audio array and sample rate from various formats."""
    if hasattr(audio_data, "get_all_samples"):
        audio_samples = audio_data.get_all_samples()
        if hasattr(audio_samples, "data"):
            audio_tensor = audio_samples.data.squeeze()
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.mean(dim=0)
            audio_array = audio_tensor.numpy()
        else:
            audio_array = np.array(audio_samples).squeeze()
        sr = audio_data.metadata.sample_rate if hasattr(audio_data, "metadata") else 16000
        return audio_array, sr
    if isinstance(audio_data, dict) and "array" in audio_data:
        return audio_data["array"], audio_data.get("sampling_rate", 16000)
    if hasattr(audio_data, "array"):
        return audio_data.array, getattr(audio_data, "sampling_rate", 16000)
    return None, None


def prepare_test_files(dataset_name="Trelis/llm-lingo", max_samples=6):
    """Prepare temp WAV files from the dataset for throughput testing."""
    import soundfile as sf
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    if "validation" in ds:
        samples = ds["validation"]
    elif "test" in ds:
        samples = ds["test"]
    else:
        samples = ds["train"]

    if max_samples:
        samples = samples.select(range(min(max_samples, len(samples))))

    files = []
    for i in range(len(samples)):
        sample = samples[i]
        audio_data = sample.get("audio")
        text = sample.get("text", sample.get("transcription", ""))

        if audio_data is None:
            continue

        audio_array, sr = extract_audio(audio_data)
        if audio_array is None:
            continue

        if isinstance(audio_array, list):
            audio_array = np.array(audio_array, dtype=np.float32)

        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(
                audio_array.astype(np.float32), orig_sr=sr, target_sr=16000
            )
            sr = 16000

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio_array, sr)
        duration = len(audio_array) / sr
        files.append({
            "path": tmp.name,
            "text": text,
            "duration_s": duration,
        })

    print(f"Prepared {len(files)} test files")
    return files


async def transcribe_single(client, url, file_info):
    """Send a single transcription request and measure timing."""
    import httpx

    start = time.perf_counter()
    with open(file_info["path"], "rb") as f:
        resp = await client.post(
            f"{url}/v1/audio/transcriptions",
            files={"file": ("test.wav", f, "audio/wav")},
            data={"model": "voxtral", "language": "en", "response_format": "json"},
        )
    elapsed = time.perf_counter() - start

    if resp.status_code == 200:
        result = resp.json()
        text = result.get("text", "")
        word_count = len(text.split())
        return {
            "success": True,
            "elapsed_s": elapsed,
            "text": text,
            "word_count": word_count,
            "audio_duration_s": file_info["duration_s"],
            "rtf": elapsed / file_info["duration_s"] if file_info["duration_s"] > 0 else 0,
        }
    else:
        return {
            "success": False,
            "elapsed_s": elapsed,
            "error": resp.text,
        }


async def run_sequential(url, files):
    """Run requests one at a time."""
    import httpx

    print("\n--- Sequential (1 request at a time) ---")
    results = []
    async with httpx.AsyncClient(timeout=300) as client:
        for i, f in enumerate(files):
            r = await transcribe_single(client, url, f)
            results.append(r)
            if r["success"]:
                print(f"  [{i}] {r['elapsed_s']:.2f}s | RTF={r['rtf']:.2f} | "
                      f"audio={r['audio_duration_s']:.1f}s | words={r['word_count']}")
            else:
                print(f"  [{i}] FAILED: {r['error'][:100]}")

    successes = [r for r in results if r["success"]]
    if successes:
        total_audio = sum(r["audio_duration_s"] for r in successes)
        total_wall = sum(r["elapsed_s"] for r in successes)
        total_words = sum(r["word_count"] for r in successes)
        avg_rtf = total_wall / total_audio if total_audio > 0 else 0
        print(f"\n  Total: {len(successes)} requests | "
              f"audio={total_audio:.1f}s | wall={total_wall:.1f}s | "
              f"avg RTF={avg_rtf:.3f} | words={total_words}")
    return results


async def run_concurrent(url, files, concurrency):
    """Run requests concurrently."""
    import httpx

    print(f"\n--- Concurrent ({concurrency} requests at a time) ---")

    async with httpx.AsyncClient(timeout=300) as client:
        start = time.perf_counter()
        tasks = [transcribe_single(client, url, f) for f in files[:concurrency]]
        results = await asyncio.gather(*tasks)
        total_wall = time.perf_counter() - start

    for i, r in enumerate(results):
        if r["success"]:
            print(f"  [{i}] {r['elapsed_s']:.2f}s | RTF={r['rtf']:.2f} | "
                  f"audio={r['audio_duration_s']:.1f}s | words={r['word_count']}")
        else:
            print(f"  [{i}] FAILED: {r['error'][:100]}")

    successes = [r for r in results if r["success"]]
    if successes:
        total_audio = sum(r["audio_duration_s"] for r in successes)
        total_words = sum(r["word_count"] for r in successes)
        throughput = total_audio / total_wall if total_wall > 0 else 0
        print(f"\n  Total wall time: {total_wall:.1f}s for {len(successes)} requests")
        print(f"  Audio processed: {total_audio:.1f}s | Throughput: {throughput:.1f}x realtime")
        print(f"  Words generated: {total_words}")
    return results


async def async_main(args):
    files = prepare_test_files(max_samples=max(6, args.concurrent))

    print(f"\nTesting endpoint: {args.url}")
    print(f"Test files: {len(files)}")

    # Sequential test
    seq_results = await run_sequential(args.url, files)

    # Concurrent test
    if args.concurrent > 1:
        conc_results = await run_concurrent(args.url, files, args.concurrent)

        # Compare
        seq_successes = [r for r in seq_results if r["success"]]
        conc_successes = [r for r in conc_results if r["success"]]
        if seq_successes and conc_successes:
            seq_avg = sum(r["elapsed_s"] for r in seq_successes) / len(seq_successes)
            conc_avg = sum(r["elapsed_s"] for r in conc_successes) / len(conc_successes)
            print(f"\n--- Comparison ---")
            print(f"  Sequential avg latency: {seq_avg:.2f}s")
            print(f"  Concurrent avg latency: {conc_avg:.2f}s")
            print(f"  Latency increase with batching: {(conc_avg/seq_avg - 1)*100:.0f}%")

    # Cleanup temp files
    import os
    for f in files:
        os.unlink(f["path"])


def main():
    parser = argparse.ArgumentParser(description="Test vLLM Voxtral throughput")
    parser.add_argument("--url", required=True, help="Voxtral server base URL")
    parser.add_argument("--concurrent", type=int, default=4,
                        help="Number of concurrent requests (default: 4)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
