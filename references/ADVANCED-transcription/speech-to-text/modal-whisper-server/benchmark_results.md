# Faster-Whisper Modal Benchmark Results

## Test Environment
- Date: 2026-01-06 23:38
- GPU: H100 (80GB VRAM)
- Platform: Modal
- **GPU Memory Snapshots**: Enabled (fast cold starts)

## Models Tested
- **Whisper Large v3**: `Systran/faster-whisper-large-v3`
- **Whisper Large v3 Turbo**: `deepdml/faster-whisper-large-v3-turbo-ct2`

## Sample Files
| File | Duration |
|------|----------|
| short | 23.6s |
| long | 772.0s |

## Large V3

| Metric | short | long |
|--------|----------|----------|
| Cold Start | 2.31s | 35.00s |
| Warm Avg | 1.10s | 33.88s |
| Warm RTF | 0.047 | 0.044 |

## Large V3 Turbo

| Metric | short | long |
|--------|----------|----------|
| Cold Start | 1.41s | 15.15s |
| Warm Avg | 0.60s | 13.60s |
| Warm RTF | 0.025 | 0.018 |

## SSE Streaming Performance

| Model | File | Time to First Segment | Total Time | Segments |
|-------|------|----------------------|------------|----------|
| large-v3 | short | 1.95s | 1.95s | 3 |
| large-v3 | long | 42.73s | 42.73s | 141 |
| large-v3-turbo | short | 1.53s | 1.53s | 3 |
| large-v3-turbo | long | 16.17s | 16.18s | 204 |

## Notes

- **RTF (Real-Time Factor)**: `processing_time / audio_duration`
  - RTF < 1.0 means faster than real-time
  - RTF = 0.1 means 10x faster than real-time (10s of audio in 1s)
- **Cold Start**: First request after container startup
  - With GPU snapshots: ~1-5s (model pre-loaded in snapshot)
  - Without GPU snapshots: ~30-60s (model loaded from disk)
- **Warm Avg**: Average of subsequent requests (model already loaded)
- **SSE Streaming**: Segments returned as they're generated via Server-Sent Events
