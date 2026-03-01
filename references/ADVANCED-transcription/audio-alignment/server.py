"""
Audio alignment server with multiple aligner backends.
- MMS_FA: TorchAudio's built-in forced aligner
- CTC Aligner: Uses wav2vec2 models (Apache 2.0 licensed)

Run with: uv run python server.py
"""

import io
import os
import re
import subprocess
import tempfile
import time
import torch
import torchaudio
import soundfile as sf
import httpx
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Get directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env from script directory
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_ENDPOINT = "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions"

app = FastAPI(title="Audio Alignment Demo")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detect device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Formats that soundfile/libsndfile can read natively
NATIVE_FORMATS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}

# Formats that need FFmpeg conversion
FFMPEG_FORMATS = {".m4a", ".mp4", ".aac", ".mp3", ".wma", ".webm", ".opus"}


def convert_audio_with_ffmpeg(audio_bytes: bytes, original_ext: str) -> tuple[bytes, str]:
    """
    Convert audio to WAV format using FFmpeg if needed.

    Returns:
        Tuple of (audio_bytes, extension) - either original or converted to WAV
    """
    ext = original_ext.lower()

    # If it's a native format, return as-is
    if ext in NATIVE_FORMATS:
        return audio_bytes, ext

    # Convert using FFmpeg
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as input_file:
        input_file.write(audio_bytes)
        input_path = input_file.name

    output_path = input_path.rsplit(".", 1)[0] + ".wav"

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000",  # Resample to 16kHz
                "-ac", "1",      # Convert to mono
                "-f", "wav",     # Output format
                output_path
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

        with open(output_path, "rb") as f:
            wav_bytes = f.read()

        return wav_bytes, ".wav"

    finally:
        # Clean up temp files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


# Cache the MMS_FA model
_fa_model = None
_fa_tokenizer = None
_fa_aligner = None

# CTC Aligner: English model (Apache 2.0 licensed)
# Note: Other language models exist but have compatibility issues with ctc-forced-aligner:
#   - "hi": "ai4bharat/indicwav2vec-hindi" (Hindi, Apache 2.0) - untested
#   - "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic" (Arabic, Apache 2.0)
#           Has tokenization mismatch with ctc-forced-aligner's star token preprocessing
# See README.md for details on adding language support.
LANGUAGE_MODELS = {
    "en": "facebook/wav2vec2-base-960h",
}
_ctc_models = {}  # Cache: {language: (model, tokenizer)}


def _load_fa_model(device: str = "cpu"):
    """Load and cache the forced alignment model."""
    global _fa_model, _fa_tokenizer, _fa_aligner

    if _fa_model is not None:
        return _fa_model, _fa_tokenizer, _fa_aligner

    print("Loading MMS_FA model...")
    bundle = torchaudio.pipelines.MMS_FA
    _fa_model = bundle.get_model().to(device)
    _fa_tokenizer = bundle.get_tokenizer()
    _fa_aligner = bundle.get_aligner()
    print("Model loaded.")

    return _fa_model, _fa_tokenizer, _fa_aligner


def normalize_text(text: str) -> str:
    """Normalize text for MMS_FA: lowercase, letters and apostrophes only."""
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z' ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_word(word: str) -> list:
    """Normalize a single word, returning list of parts (hyphens split)."""
    word = word.lower()
    word = word.replace("-", " ")
    word = re.sub(r"[^a-z' ]", "", word)
    word = re.sub(r"\s+", " ", word).strip()
    return [p for p in word.split() if p]


def normalize_with_mapping(text: str):
    """
    Normalize text while tracking mapping from original to normalized words.

    Example:
        "Phi-2 is fine-tuning!" ->
        original_words = ["Phi-2", "is", "fine-tuning!"]
        normalized_words = ["phi", "is", "fine", "tuning"]
        mapping = [[0], [1], [2, 3]]
    """
    original_words = text.split()
    normalized_words = []
    mapping = []

    for word in original_words:
        norm_parts = _normalize_word(word)
        if norm_parts:
            start_idx = len(normalized_words)
            normalized_words.extend(norm_parts)
            mapping.append(list(range(start_idx, len(normalized_words))))
        else:
            mapping.append([])

    return original_words, normalized_words, mapping


def reverse_timestamps(timestamps: list, original_words: list, mapping: list) -> list:
    """Map normalized word timestamps back to original words."""
    result = []

    for orig_idx, word in enumerate(original_words):
        norm_indices = mapping[orig_idx]

        if not norm_indices:
            t = result[-1]["end"] if result else 0.0
            result.append({"word": word, "start": t, "end": t})
        elif norm_indices[0] >= len(timestamps):
            break
        else:
            first_idx = norm_indices[0]
            last_idx = min(norm_indices[-1], len(timestamps) - 1)
            result.append({
                "word": word,
                "start": timestamps[first_idx]["start"],
                "end": timestamps[last_idx]["end"]
            })

    return result


def align_audio(waveform: torch.Tensor, sample_rate: int, text: str, preserve_original: bool = False):
    """
    Align text to audio using MMS_FA and return word timestamps.

    Args:
        preserve_original: If True, return original words (with caps/punctuation).
                          If False, return normalized words.

    Returns:
        timestamps: List of {"word": str, "start": float, "end": float}
        duration: Audio duration in seconds
        align_time_ms: Alignment time in milliseconds
    """
    model, tokenizer, aligner = _load_fa_model()
    bundle = torchaudio.pipelines.MMS_FA

    # Start timing after model is loaded
    start_time = time.perf_counter()

    # Ensure 2D waveform
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, bundle.sample_rate)
        waveform = resampler(waveform)
        sample_rate = bundle.sample_rate

    # Get original words and mapping if preserving original
    original_words = None
    mapping = None
    if preserve_original:
        original_words, words, mapping = normalize_with_mapping(text)
    else:
        normalized = normalize_text(text)
        words = normalized.split()

    duration = waveform.shape[1] / sample_rate

    if not words:
        return [], duration, 0.0

    # Get emissions
    with torch.inference_mode():
        emission, _ = model(waveform)

    # Tokenize and align
    tokens = tokenizer(words)
    token_spans = aligner(emission[0], tokens)

    # Convert to timestamps (50 frames/sec at 16kHz)
    frames_per_second = 50
    timestamps = []

    for i, word in enumerate(words):
        if i >= len(token_spans):
            break

        char_spans = token_spans[i]
        if not char_spans:
            continue

        start_time_sec = char_spans[0].start / frames_per_second
        end_time_sec = char_spans[-1].end / frames_per_second

        timestamps.append({
            "word": word,
            "start": round(start_time_sec, 3),
            "end": round(end_time_sec, 3)
        })

    # Reverse mapping to get original words back
    if preserve_original and original_words and mapping:
        timestamps = reverse_timestamps(timestamps, original_words, mapping)

    align_time_ms = (time.perf_counter() - start_time) * 1000
    return timestamps, duration, align_time_ms


# ============================================================================
# CTC Aligner (Apache 2.0 licensed wav2vec2 models)
# ============================================================================

def _load_ctc_model(language: str = "en", device: str = DEVICE):
    """Load and cache wav2vec2 model for alignment."""
    global _ctc_models

    if language not in LANGUAGE_MODELS:
        print(f"Warning: Unsupported language '{language}', falling back to English")
        language = "en"

    if language in _ctc_models:
        return _ctc_models[language]

    from ctc_forced_aligner import load_alignment_model

    model_path = LANGUAGE_MODELS[language]
    print(f"Loading {model_path} alignment model...")

    # Use float16 on CUDA for speed, float32 otherwise
    dtype = torch.float16 if device == "cuda" else torch.float32
    model, tokenizer = load_alignment_model(device, model_path=model_path, dtype=dtype)

    _ctc_models[language] = (model, tokenizer)
    print(f"CTC model loaded on {device}")

    return model, tokenizer


def align_audio_ctc(
    waveform: torch.Tensor,
    sample_rate: int,
    text: str,
    preserve_original: bool = False,
    language: str = "en"
):
    """
    Align text to audio using ctc-forced-aligner with Apache 2.0 licensed models.

    Returns:
        timestamps: List of {"word": str, "start": float, "end": float}
        duration: Audio duration in seconds
        align_time_ms: Time taken for alignment in milliseconds
    """
    from ctc_forced_aligner import (
        generate_emissions,
        preprocess_text,
        get_alignments,
        get_spans,
        postprocess_results,
    )

    model, tokenizer = _load_ctc_model(language)

    # Start timing after model is loaded (excludes model download/load time)
    start_time = time.perf_counter()

    # Ensure 2D waveform [channels, samples] for resampling
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    duration = waveform.shape[1] / sample_rate

    # Get original words and mapping if preserving original
    original_words = None
    mapping = None
    if preserve_original:
        original_words, words, mapping = normalize_with_mapping(text)
        text_for_alignment = " ".join(words)
    else:
        text_for_alignment = normalize_text(text)
        words = text_for_alignment.split()

    if not words:
        return [], duration, 0.0

    # Convert to 1D and match model dtype for generate_emissions
    # ctc-forced-aligner expects shape (samples,) not (1, samples)
    audio_for_model = waveform.squeeze(0).to(device=DEVICE, dtype=model.dtype)

    # Generate emissions
    emissions, stride = generate_emissions(model, audio_for_model, batch_size=1)

    # Preprocess text for alignment
    tokens_starred, text_starred = preprocess_text(
        text_for_alignment,
        romanize=True,
        language=language,
    )

    # Get alignments
    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        tokenizer,
    )

    # Get word spans
    spans = get_spans(tokens_starred, segments, blank_token)

    # Use postprocess_results to get word timestamps
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    # Convert to our format
    timestamps = []
    for i, wt in enumerate(word_timestamps):
        if i >= len(words):
            break
        timestamps.append({
            "word": words[i],
            "start": round(wt["start"], 3),
            "end": round(wt["end"], 3),
        })

    # Reverse mapping to get original words back
    if preserve_original and original_words and mapping:
        timestamps = reverse_timestamps(timestamps, original_words, mapping)

    align_time_ms = (time.perf_counter() - start_time) * 1000

    return timestamps, duration, align_time_ms


@app.post("/align")
async def align(
    audio: UploadFile,
    text: str = Form(...),
    preserve_original: str = Form("false"),
    aligner: str = Form("mms_fa"),
    language: str = Form("en")
):
    """Align uploaded audio with provided text.

    Args:
        audio: Audio file to align
        text: Text transcript to align
        preserve_original: Preserve original capitalization/punctuation
        aligner: "mms_fa" (default) or "ctc_aligner"
        language: Language code for CTC Aligner (e.g., "en", "hi", "ar")
    """
    preserve_original_bool = preserve_original.lower() in ("true", "1", "yes")
    try:
        # Read audio file
        audio_bytes = await audio.read()

        # Get file extension and convert if needed
        original_ext = os.path.splitext(audio.filename or ".wav")[1].lower()
        audio_bytes, ext = convert_audio_with_ffmpeg(audio_bytes, original_ext)

        audio_buffer = io.BytesIO(audio_bytes)

        # Load with soundfile (torchaudio 2.9+ requires torchcodec which we avoid)
        audio_data, sample_rate = sf.read(audio_buffer)

        # Convert to torch tensor, shape [channels, samples]
        waveform = torch.from_numpy(audio_data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, samples]
        else:
            waveform = waveform.T  # soundfile returns [samples, channels], we need [channels, samples]

        # Perform alignment with selected aligner
        # Note: mono conversion and resampling handled inside each aligner function
        if aligner == "ctc_aligner":
            timestamps, duration, align_time_ms = align_audio_ctc(
                waveform, sample_rate, text, preserve_original_bool, language
            )
        else:
            # Default to MMS_FA
            timestamps, duration, align_time_ms = align_audio(
                waveform, sample_rate, text, preserve_original_bool
            )

        return JSONResponse({
            "success": True,
            "timestamps": timestamps,
            "duration": round(duration, 3),
            "preserve_original": preserve_original_bool,
            "aligner": aligner,
            "alignment_time_ms": round(align_time_ms, 2)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=400)


@app.get("/config")
async def config():
    """Return configuration info for the UI."""
    return JSONResponse({
        "transcription_available": bool(FIREWORKS_API_KEY and FIREWORKS_API_KEY != "your-api-key-here"),
        "aligners": [
            {
                "id": "mms_fa",
                "name": "MMS-FA (TorchAudio)",
                "description": "Fast, built-in aligner. English only.",
                "supports_language": False
            },
            {
                "id": "ctc_aligner",
                "name": "CTC Aligner (Apache 2.0)",
                "description": "wav2vec2 model. Commercial-friendly license.",
                "supports_language": False
            }
        ]
    })


@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    """Transcribe audio using Fireworks Whisper API."""
    if not FIREWORKS_API_KEY or FIREWORKS_API_KEY == "your-api-key-here":
        return JSONResponse({
            "success": False,
            "error": "FIREWORKS_API_KEY not configured in .env"
        }, status_code=400)

    try:
        audio_bytes = await audio.read()

        # Determine mime type
        ext = os.path.splitext(audio.filename or "audio.mp3")[1].lower()
        mime_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
        }
        mime_type = mime_types.get(ext, "audio/mpeg")

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                FIREWORKS_ENDPOINT,
                headers={"Authorization": FIREWORKS_API_KEY},
                files={"file": (audio.filename or "audio.mp3", audio_bytes, mime_type)},
                data={"model": "whisper-v3", "response_format": "text"},
            )

        response.raise_for_status()
        return JSONResponse({
            "success": True,
            "transcript": response.text.strip()
        })

    except httpx.HTTPStatusError as e:
        return JSONResponse({
            "success": False,
            "error": f"Fireworks API error: {e.response.status_code}"
        }, status_code=400)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=400)


@app.get("/")
async def root():
    """Serve the index.html file."""
    index_path = os.path.join(SCRIPT_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>Audio Alignment Demo</h1><p>index.html not found</p>")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Audio Alignment Server")
    parser.add_argument("--port", type=int, default=8200, help="Port to run server on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    print(f"\n  Open: http://localhost:{args.port}\n")

    if args.reload:
        uvicorn.run("server:app", host="127.0.0.1", port=args.port, reload=True)
    else:
        uvicorn.run(app, host="127.0.0.1", port=args.port)
