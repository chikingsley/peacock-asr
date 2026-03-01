# Audio Alignment with TorchAudio

Forced alignment maps text to audio, producing word-level timestamps. This tutorial demonstrates alignment using TorchAudio's MMS_FA (Massively Multilingual Speech - Forced Alignment) model.

## How It Works

The alignment process has 4 key steps:

### 1. Text Normalization
MMS_FA only understands lowercase letters and apostrophes. Text is normalized before alignment:
```
"Phi-2 is fine-tuning!" → ["phi", "is", "fine", "tuning"]
```

### 2. Character Tokenization
Each word is split into characters for the model:
```
"great" → ['g', 'r', 'e', 'a', 't']
```

### 3. CTC Alignment
- Audio is processed at 16kHz with 20ms frames (320 samples per frame)
- The model outputs emission probabilities for each character at each frame
- A CTC (Connectionist Temporal Classification) aligner finds the optimal path matching characters to frames:
  - CTC allows variable-length alignment by introducing a "blank" token for frames with no character output
  - The Viterbi algorithm finds the most likely path through the probability matrix (dynamic programming that walks frame-by-frame, keeping only the best path to each state rather than exploring all possible paths)
  - This handles speech rate variations (fast/slow speakers) without explicit duration modeling

**Mini Viterbi example:** Align "t" to 3 frames. States: `blank`, `t`. Constraint: must emit "t" in order (can start with blank, must pass through t).

```
Emission probs:     Frame1   Frame2   Frame3
blank                 0.8      0.2      0.9
t                     0.2      0.8      0.1
```

Frame 1 — "What's the prob of being in each state now?"
- blank: 0.8 | t: 0.2

Frame 2 — "Best path to each state?"
- blank: from blank → 0.8 × 0.2 = 0.16
- t: max(from blank: 0.8 × 0.8, from t: 0.2 × 0.8) = **0.64** (via blank)

Frame 3 — "Best path to each state?"
- blank: from t → 0.64 × 0.9 = **0.576** ✓ winner
- t: from t → 0.64 × 0.1 = 0.064

**Result:** blank → t → blank (t aligned to frame 2)

#### GPU/CPU Architecture

CTC alignment is a two-step process:

1. **GPU: Forward Pass** — Neural network outputs emission matrix (log probs per character per frame). For long audio, `ctc-forced-aligner` chunks into 30-second windows with 2-second context overlap, batches these chunks, then concatenates results.

2. **CPU: Viterbi Decoding** — Walks frame-by-frame through emissions to find the optimal alignment path. Inherently sequential (frame N depends on frame N-1).

**Batching support:**
| Type | GPU Forward | CPU Viterbi |
|------|-------------|-------------|
| Within-sequence (chunking) | Yes | No |
| Multiple inputs | No | No |

To align multiple files in parallel, run separate processes—the library processes one audio at a time.

**Parallelizing Viterbi:** Use VAD to split audio into independent speech segments. Each segment can be Viterbi-decoded in parallel since they don't depend on each other.

### 4. Timestamp Extraction
Frame indices are converted to seconds:
```python
start_time = start_frame / 50  # 50 frames per second at 16kHz
```

## Why Not Just Use Whisper Timestamps?

Models like Whisper and WhisperX can provide word-level timestamps during transcription. However, these timestamps become useless if you:

1. **Correct the transcript** - If you fix transcription errors, the original timestamps no longer match
2. **Generate synthetic speech** - Text-to-speech output has no pre-existing timestamps

In both cases, you need to **realign** the corrected/new text to the audio.

Sample whisper alignment:

```bash
cd audio-alignment
uv run --isolated --with transformers --with 'datasets<3.0' --with soundfile --with 'librosa>=0.10' --with 'numba>=0.58' python - <<'PY'
from transformers import pipeline

MODEL_ID = "openai/whisper-tiny"
audio_path = "../speech-to-text/data/llm-lingo-dataset/segment_0.mp3"

asr = pipeline("automatic-speech-recognition", model=MODEL_ID, return_timestamps="word")
result = asr(audio_path)

print(f"Result:\n---\n{result}\n")
PY
```

## Use Cases

### 1. Transcript Correction Workflow
```
Audio → Whisper → Initial Transcript (with timestamps)
                         ↓
              Human corrects errors
                         ↓
              Corrected Transcript (timestamps invalid!)
                         ↓
              Forced Alignment → New timestamps
```

### 2. Synthetic Speech Dataset Preparation
```
Text Dataset → TTS Model → Synthetic Audio
                               ↓
                   Forced Alignment → Word timestamps
                               ↓
                   Chunked training data for fine-tuning
```

## Setup

### Prerequisites

**FFmpeg** is required for processing M4A, MP3, AAC, and other compressed audio formats:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

### Install Dependencies

```bash
cd ADVANCED-transcription/audio-alignment
uv sync
```

echo 'FIREWORKS_API_KEY=fw_VWjkntpbo9MH6B9pny6rDb' > audio-alignment/.env

## Running the Demo

1. Start the server:
```bash
cd audio-alignment
uv sync
uv run python server.py
# Or with auto-reload for development:
uv run python server.py --reload
```

2. Open http://localhost:8200 in your browser

3. Upload an audio file and enter the transcript text (or use auto-transcription)

4. Click "Align" to see the word-level timestamps visualized

## Preserve Original Text

By default, the "Preserve original text" checkbox is enabled. This returns timestamps with your original words intact (capitalization, punctuation, hyphens).

**With preserve original (default):**
```
Input: "Hello World! This is fine-tuning."
Output: ["Hello", "World!", "This", "is", "fine-tuning."]
```

**Without preserve original:**
```
Input: "Hello World! This is fine-tuning."
Output: ["hello", "world", "this", "is", "fine", "tuning"]
```

The mapping approach tracks which normalized words correspond to which original words, then reverses the mapping after alignment to reconstruct the original text with accurate timestamps.

## Optional: Auto-Transcription with Fireworks

To auto-generate a transcript before alignment:

1. Get a Fireworks API key from https://fireworks.ai

2. Edit `.env` and add your key:
```
FIREWORKS_API_KEY=your-actual-api-key
```

3. Restart the server - the "Transcribe" button will appear

## Aligner Options

The demo provides two aligner backends:

### MMS-FA (TorchAudio) - Default
- Built-in TorchAudio forced aligner
- Fast and reliable for English
- License: CC-BY-NC-4.0 (non-commercial)

### CTC Aligner (Apache 2.0)
- Uses `facebook/wav2vec2-base-960h` model
- Commercial-friendly Apache 2.0 license
- Powered by [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)

## Adding Language Support (Experimental)

The CTC Aligner currently only supports English. Other Apache 2.0 licensed models exist but have compatibility issues with the ctc-forced-aligner library:

| Language | Model | License | Status |
|----------|-------|---------|--------|
| English | `facebook/wav2vec2-base-960h` | Apache 2.0 | Working |
| Hindi | `ai4bharat/indicwav2vec-hindi` | Apache 2.0 | Untested |
| Arabic | `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` | Apache 2.0 | Tokenization mismatch with star token preprocessing |

To add language support, modify `LANGUAGE_MODELS` in `server.py`:

```python
LANGUAGE_MODELS = {
    "en": "facebook/wav2vec2-base-960h",
    "hi": "ai4bharat/indicwav2vec-hindi",  # Untested
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",  # May have issues
}
```

Note: The ctc-forced-aligner library was designed for MMS-based forced alignment models. Some wav2vec2 models have different vocabulary structures that cause tokenization mismatches (e.g., `<star> != m` assertion errors). The default MMS model (`MahmoudAshraf/mms-300m-1130-forced-aligner`) works but is CC-BY-NC licensed
