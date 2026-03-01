# Word Confidence Comparison

Compare transcription quality across APIs with word-level confidence highlighting.

## What it does

Upload an audio file → get transcriptions from multiple providers side-by-side, with words color-coded by confidence:
- **Red**: <60% confidence
- **Yellow**: 60-80%
- **Normal**: ≥80%

Supported providers:
- AssemblyAI (requires API key)
- Soniox (requires API key)
- Whisper Large v3 (via Modal)
- Whisper Turbo (via Modal)

## Setup

```bash
cp .env.example .env  # Add your API keys
uv sync
```

## Run

```bash
cd speech-to-text/word-confidence-comparison
uv sync
uv run uvicorn server:app --reload --port 8011
# Open http://localhost:8011
```

## API Keys

Set in  `speech-to-text/word-confidence-comparison/.env`:
```
ASSEMBLYAI_API_KEY=your_key
SONIOX_API_KEY=your_key
```
