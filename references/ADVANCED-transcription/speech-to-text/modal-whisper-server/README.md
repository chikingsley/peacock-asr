# Modal Faster-Whisper Server

Deploy OpenAI-compatible transcription API endpoints on Modal using faster-whisper.

## Features

- **GPU Memory Snapshots**: Fast cold starts (~1-5s vs ~30-60s without)
- **SSE Streaming**: Transcription segments returned as they're generated
- **Word-level timestamps**: Confidence scores per word in `verbose_json`
- **OpenAI-compatible API**: Works with OpenAI SDK

## Models

Two Whisper variants are deployed:

| Model | HuggingFace ID | Performance (H100) |
|-------|----------------|-------------|
| Large v3 | `Systran/faster-whisper-large-v3` | RTF ~0.044 (22x real-time) |
| Large v3 Turbo | `deepdml/faster-whisper-large-v3-turbo-ct2` | RTF ~0.018 (55x real-time) |

## Prerequisites

1. Install Modal CLI and authenticate:
   ```bash
   pip install modal
   modal setup
   ```

2. Install dependencies:
   ```bash
   cd speech-to-text/modal-whisper-server
   uv sync
   ```

## Deployment

Deploy both servers to Modal:

```bash
uv run modal deploy server.py
```

This creates two endpoints:
- Large v3: `https://<workspace>--faster-whisper-server-whisperlargev3-serve.modal.run`
- Large v3 Turbo: `https://<workspace>--faster-whisper-server-whisperlargev3turbo-serve.modal.run`

## Usage

### OpenAI SDK (Python)

```python
from openai import OpenAI

# Choose endpoint
endpoint = "https://<workspace>--faster-whisper-server-whisperlargev3turbo-serve.modal.run"

client = OpenAI(api_key="EMPTY", base_url=f"{endpoint}/v1/")

with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="large-v3-turbo",
        file=f,
        language="en",
        response_format="text"  # or "json", "vtt", "srt", "verbose_json"
    )

print(transcript)
```

### cURL

```bash
curl -X POST https://<endpoint>/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=large-v3-turbo" \
  -F "language=en" \
  -F "response_format=text"
```

## API Endpoints

### `GET /health`
Health check endpoint.

### `GET /v1/models`
List available models.

### `POST /v1/audio/transcriptions`
OpenAI-compatible transcription endpoint.

**Parameters:**
- `file` (required): Audio file
- `model`: Model name (ignored, uses deployed model)
- `language`: Language code (default: "en")
- `response_format`: Output format - "json", "text", "vtt", "srt", "verbose_json" (default: "json")
- `temperature`: Sampling temperature (default: 0.0)
- `prompt`: Initial prompt for transcription
- `stream`: Set to `true` for SSE streaming (default: false)

### SSE Streaming

When `stream=true`, the endpoint returns Server-Sent Events:

```bash
curl -X POST https://<endpoint>/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "stream=true"
```

Response format:
```
data: {"type": "info", "language": "en", "duration": 23.6}
data: {"type": "segment", "id": 0, "start": 0.0, "end": 10.5, "text": "...", "words": [...]}
data: {"type": "segment", "id": 1, "start": 10.5, "end": 20.0, "text": "...", "words": [...]}
data: {"type": "done"}
```

## Benchmark Results

See [benchmark_results.md](benchmark_results.md) for detailed performance metrics.

**Summary (H100 GPU with GPU Snapshots):**

| Model | Cold Start | Warm RTF | Speed |
|-------|------------|----------|-------|
| Large v3 | ~2s | 0.044 | 22x real-time |
| Large v3 Turbo | ~1.5s | 0.018 | 55x real-time |

GPU memory snapshots reduce cold starts from ~30-60s to ~1-5s by pre-loading the model into the snapshot.

## Running Benchmarks

```bash
# Run full benchmark (3 runs per model/file)
uv run python benchmark.py --runs 3

# Include SSE streaming benchmarks
uv run python benchmark.py --runs 3 --sse

# Test specific model
uv run python benchmark.py --model large-v3-turbo

# Test specific file
uv run python benchmark.py --file short
```

## Configuration

The server is configured with:
- **GPU**: H100 (80GB VRAM)
- **Memory**: 32GB RAM
- **Timeout**: 10 minutes per request
- **Scaledown window**: 5 minutes (container stays warm)
- **GPU Memory Snapshots**: Enabled for fast cold starts

Models are cached in a Modal Volume. GPU memory snapshots capture the loaded model state, reducing cold starts from ~30-60s to ~1-5s.

## Files

- `server.py` - Modal deployment code (with GPU snapshots + SSE streaming)
- `server_legacy.py` - Previous implementation (without GPU snapshots)
- `benchmark.py` - Performance benchmarking script
- `benchmark_results.md` - Latest benchmark results
- `pyproject.toml` - Python dependencies
