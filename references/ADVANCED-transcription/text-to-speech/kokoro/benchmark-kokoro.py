import asyncio
import time
import statistics
from openai import AsyncOpenAI
import argparse
from typing import List, Dict
import json

# Test text snippets of varying lengths
TEST_SNIPPETS = [
    "Hello, this is a short test.",
    "The quick brown fox jumps over the lazy dog. This is a medium length sentence for testing.",
    "In the heart of the bustling city, where skyscrapers touched the clouds and the streets hummed with endless activity, a small cafe nestled between two towering buildings offered a quiet refuge for those seeking a moment of peace.",
    "Technology has revolutionized the way we communicate, work, and live our daily lives. From smartphones to artificial intelligence, innovations continue to reshape our world in ways we never imagined possible. The future holds even more exciting possibilities as we push the boundaries of what machines can accomplish.",
    "The art of storytelling has been a fundamental part of human culture for thousands of years. From ancient cave paintings to modern digital media, we have always sought to share our experiences, dreams, and knowledge with others. Stories connect us, teach us, and help us understand the world around us in profound and meaningful ways that transcend language and culture.",
]

# Voice options to test
VOICES = ["af_heart", "af_bella", "am_michael", "bf_emma", "bm_george"]


class KokoroBenchmark:
    def __init__(self, base_url: str, concurrency: int = 5):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",
        )
        self.concurrency = concurrency
        self.results = []

    async def make_request(self, text: str, voice: str, request_id: int) -> Dict:
        """Make a single TTS request and measure performance"""
        start_time = time.time()

        try:
            response = await self.client.audio.speech.create(
                model="kokoro",
                voice=voice,
                input=text,
            )

            # Get the audio content
            audio_content = response.content
            audio_size = len(audio_content)

            end_time = time.time()
            duration = end_time - start_time

            # Estimate audio duration (rough estimate: ~150 words per minute for TTS)
            # or about 2.5 words per second
            word_count = len(text.split())
            estimated_audio_duration = word_count / 2.5  # seconds

            return {
                'success': True,
                'request_id': request_id,
                'text_length': len(text),
                'word_count': word_count,
                'audio_size': audio_size,
                'processing_time': duration,
                'estimated_audio_duration': estimated_audio_duration,
                'tokens_per_second': len(text.split()) / duration if duration > 0 else 0,
                'realtime_factor': estimated_audio_duration / duration if duration > 0 else 0,
            }
        except Exception as e:
            end_time = time.time()
            return {
                'success': False,
                'request_id': request_id,
                'error': str(e),
                'processing_time': end_time - start_time,
            }

    async def run_batch(self, num_requests: int) -> List[Dict]:
        """Run a batch of requests with controlled concurrency"""
        tasks = []
        results = []

        print(f"\n🚀 Starting batch of {num_requests} requests with concurrency={self.concurrency}")

        semaphore = asyncio.Semaphore(self.concurrency)

        async def bounded_request(request_id: int):
            async with semaphore:
                # Rotate through test snippets and voices
                text = TEST_SNIPPETS[request_id % len(TEST_SNIPPETS)]
                voice = VOICES[request_id % len(VOICES)]
                return await self.make_request(text, voice, request_id)

        batch_start = time.time()

        # Create all tasks
        tasks = [bounded_request(i) for i in range(num_requests)]

        # Run with progress updates
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            if (i + 1) % 10 == 0 or (i + 1) == num_requests:
                elapsed = time.time() - batch_start
                print(f"  Progress: {i + 1}/{num_requests} requests completed ({elapsed:.2f}s)")

        batch_end = time.time()
        batch_duration = batch_end - batch_start

        return results, batch_duration

    def analyze_results(self, results: List[Dict], batch_duration: float):
        """Analyze and print benchmark results"""
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', True)]

        if not successful:
            print("\n❌ No successful requests!")
            return

        # Calculate statistics
        processing_times = [r['processing_time'] for r in successful]
        audio_durations = [r['estimated_audio_duration'] for r in successful]
        tokens_per_sec = [r['tokens_per_second'] for r in successful]
        realtime_factors = [r['realtime_factor'] for r in successful]

        total_audio_duration = sum(audio_durations)

        print("\n" + "="*70)
        print("📊 BENCHMARK RESULTS")
        print("="*70)

        print(f"\n📈 Request Statistics:")
        print(f"  Total requests: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Success rate: {len(successful)/len(results)*100:.1f}%")

        print(f"\n⏱️  Timing Statistics:")
        print(f"  Total batch time: {batch_duration:.2f}s")
        print(f"  Avg processing time per request: {statistics.mean(processing_times):.3f}s")
        print(f"  Median processing time: {statistics.median(processing_times):.3f}s")
        print(f"  Min processing time: {min(processing_times):.3f}s")
        print(f"  Max processing time: {max(processing_times):.3f}s")
        if len(processing_times) > 1:
            print(f"  Std dev: {statistics.stdev(processing_times):.3f}s")

        print(f"\n🎤 Audio Generation:")
        print(f"  Total audio generated: {total_audio_duration:.2f}s ({total_audio_duration/60:.2f} minutes)")
        print(f"  Avg audio duration per request: {statistics.mean(audio_durations):.2f}s")

        print(f"\n⚡ Performance Metrics:")
        print(f"  Avg tokens/second: {statistics.mean(tokens_per_sec):.2f}")
        print(f"  Avg realtime factor: {statistics.mean(realtime_factors):.2f}x")
        print(f"    (1.0x = generates audio at same speed as playback)")
        print(f"    ({statistics.mean(realtime_factors):.2f}x = generates {statistics.mean(realtime_factors):.2f} seconds of audio per second)")

        print(f"\n🎯 Throughput Capacity:")
        audio_per_hour = (total_audio_duration / batch_duration) * 3600  # seconds of audio per hour
        print(f"  Audio processed per wall-clock hour: {audio_per_hour/3600:.2f} hours ({audio_per_hour/60:.2f} minutes)")
        print(f"  Effective speed-up: {audio_per_hour/3600:.2f}x realtime")

        # Calculate with current concurrency
        avg_realtime_factor = statistics.mean(realtime_factors)
        theoretical_max = avg_realtime_factor * self.concurrency
        print(f"\n💡 Theoretical Maximum (with concurrency={self.concurrency}):")
        print(f"  Could process ~{theoretical_max:.2f} hours of audio per wall-clock hour")
        print(f"  That's {theoretical_max:.2f}x realtime speed")

        print("\n" + "="*70)

        if failed:
            print(f"\n⚠️  Failed requests: {len(failed)}")
            for f in failed[:5]:  # Show first 5 failures
                print(f"  Request {f['request_id']}: {f.get('error', 'Unknown error')}")


async def main():
    parser = argparse.ArgumentParser(description='Benchmark Kokoro TTS endpoint')
    parser.add_argument('--url', default='http://213.181.105.225:10821/v1',
                      help='Base URL for the Kokoro endpoint')
    parser.add_argument('--requests', type=int, default=50,
                      help='Number of requests to make')
    parser.add_argument('--concurrency', type=int, default=5,
                      help='Number of concurrent requests')
    parser.add_argument('--test-levels', action='store_true',
                      help='Test multiple concurrency levels')

    args = parser.parse_args()

    if args.test_levels:
        # Test different concurrency levels
        concurrency_levels = [1, 3, 5, 10, 15, 20]
        print(f"🔬 Testing multiple concurrency levels: {concurrency_levels}")

        all_results = {}
        for concurrency in concurrency_levels:
            print(f"\n{'='*70}")
            print(f"Testing with concurrency={concurrency}")
            print(f"{'='*70}")

            benchmark = KokoroBenchmark(args.url, concurrency=concurrency)
            results, batch_duration = await benchmark.run_batch(args.requests)
            benchmark.analyze_results(results, batch_duration)

            successful = [r for r in results if r.get('success', False)]
            if successful:
                audio_durations = [r['estimated_audio_duration'] for r in successful]
                total_audio = sum(audio_durations)
                audio_per_hour = (total_audio / batch_duration) * 3600
                all_results[concurrency] = audio_per_hour / 3600

            # Brief pause between tests
            await asyncio.sleep(2)

        print(f"\n{'='*70}")
        print("📊 SUMMARY ACROSS CONCURRENCY LEVELS")
        print(f"{'='*70}")
        for concurrency, hours_per_hour in sorted(all_results.items()):
            print(f"  Concurrency {concurrency:2d}: {hours_per_hour:.2f} hours of audio per hour")

        best_concurrency = max(all_results.items(), key=lambda x: x[1])
        print(f"\n✨ Best performance: concurrency={best_concurrency[0]} with {best_concurrency[1]:.2f}x realtime")

    else:
        # Single test
        benchmark = KokoroBenchmark(args.url, concurrency=args.concurrency)
        results, batch_duration = await benchmark.run_batch(args.requests)
        benchmark.analyze_results(results, batch_duration)


if __name__ == "__main__":
    asyncio.run(main())
