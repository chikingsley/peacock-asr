# Voxtral Modal Serving

OpenAI-compatible transcription API using vLLM + Voxtral on Modal.

## Prerequisites

- Modal account and CLI (`uv pip install modal && modal setup`)
- `.env` file with `HF_TOKEN` (copy from `.env.example`)

## Usage

```bash
# Test locally
uv run modal run --env=dev voxtral_modal_serve.py

# Deploy
uv run modal deploy voxtral_modal_serve.py
```

## API Endpoints

### `GET /health`
Health check.

### `GET /v1/models`
List available models.

### `POST /v1/audio/transcriptions`
OpenAI-compatible transcription endpoint.

**Parameters:**
- `file` (required): Audio file (mp3/wav/m4a/flac/ogg/webm/mp4)
- `model`: Model name (ignored, uses Voxtral)
- `language`: Language code (default: `en`)
- `response_format`: `json` | `text` | `vtt` | `srt` | `verbose_json`
- `temperature`: Sampling temperature (default: `0.0`)

**Example:**
```bash
curl -X POST https://your-modal-url/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F response_format=json
```

## Architecture

- vLLM runs as a subprocess inside the Modal container
- FastAPI wraps and proxies requests to vLLM's native API
- Model cache volume persists downloads across cold starts
- 5-min scaledown keeps the GPU warm between requests

## Limitations

- No word-level timestamps (VTT/SRT use placeholder timestamps)
- Cold start: 30-60s without model cache, faster with volume
