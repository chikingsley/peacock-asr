"""
Modal deployment for faster-whisper transcription servers.

Deploys OpenAI-compatible API endpoints for:
- Whisper Large v3
- Whisper Large v3 Turbo

Features:
- GPU memory snapshots for fast cold starts (~5-10s vs ~30-60s)
- SSE streaming transcription (segments sent as they're generated)
- Word-level confidence scores in verbose_json

Usage:
    modal deploy server.py
"""

import json
import tempfile
from pathlib import Path

import modal

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".mpeg", ".mpga"}
VALID_RESPONSE_FORMATS = {"json", "text", "vtt", "srt", "verbose_json"}

app = modal.App("faster-whisper-server")

# Volume for caching downloaded models
model_cache = modal.Volume.from_name("whisper-model-cache", create_if_missing=True)
CACHE_DIR = "/cache"

# Modal image with CUDA, cuDNN and faster-whisper
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper>=1.0.0",
        "fastapi>=0.100.0",
        "python-multipart",
        "uvicorn",
        "scipy",  # For warmup audio generation
    )
)

# Model configurations
MODELS = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}


def format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for VTT (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_transcription_app(model_key: str, model_instance):
    """Create a FastAPI app for transcription with SSE streaming support."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

    web_app = FastAPI(
        title=f"Faster-Whisper {model_key}",
        description=f"OpenAI-compatible transcription API for Whisper {model_key}",
    )

    @web_app.get("/health")
    def health():
        return {"status": "ok", "model": model_key}

    @web_app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_key,
                    "object": "model",
                    "owned_by": "openai",
                }
            ],
        }

    @web_app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form(None),
        language: str = Form("en"),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        prompt: str = Form(None),
        stream: bool = Form(False),
    ):
        """
        OpenAI-compatible transcription endpoint.

        Supported response_format values:
        - json: {"text": "..."}
        - text: plain text
        - vtt: WebVTT format
        - srt: SRT format
        - verbose_json: detailed JSON with segments and word probabilities

        Set stream=true for SSE streaming (segments sent as generated).
        """
        # Validate response_format before processing
        if response_format not in VALID_RESPONSE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported response_format: {response_format}. Valid options: {', '.join(VALID_RESPONSE_FORMATS)}",
            )

        # Read audio file
        audio_bytes = await file.read()

        # Validate and sanitize file extension
        suffix = Path(file.filename).suffix.lower() if file.filename else ".mp3"
        if suffix not in ALLOWED_EXTENSIONS:
            suffix = ".mp3"

        # SSE Streaming mode
        if stream:
            async def generate_sse():
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                    tmp.write(audio_bytes)
                    tmp.flush()

                    try:
                        segments, info = model_instance.transcribe(
                            tmp.name,
                            language=language if language else None,
                            temperature=temperature,
                            initial_prompt=prompt,
                            word_timestamps=True,
                        )

                        # Send info event first
                        info_data = {
                            "type": "info",
                            "language": info.language,
                            "language_probability": info.language_probability,
                            "duration": info.duration,
                        }
                        yield f"data: {json.dumps(info_data)}\n\n"

                        # Stream segments as they're generated
                        for i, seg in enumerate(segments):
                            segment_data = {
                                "type": "segment",
                                "id": i,
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text.strip(),
                                "words": (
                                    [
                                        {
                                            "word": w.word,
                                            "start": w.start,
                                            "end": w.end,
                                            "probability": w.probability,
                                        }
                                        for w in seg.words
                                    ]
                                    if seg.words
                                    else []
                                ),
                            }
                            yield f"data: {json.dumps(segment_data)}\n\n"

                        # Send done event
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"

                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming mode (original behavior)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            # Transcribe with error handling
            try:
                segments, info = model_instance.transcribe(
                    tmp.name,
                    language=language if language else None,
                    temperature=temperature,
                    initial_prompt=prompt,
                    word_timestamps=(response_format == "verbose_json"),
                )
                # Collect segments
                segments_list = list(segments)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Transcription failed: {str(e)}",
                )

        # Format response
        full_text = " ".join(seg.text.strip() for seg in segments_list)

        if response_format == "text":
            return PlainTextResponse(full_text)

        elif response_format == "json":
            return JSONResponse({"text": full_text})

        elif response_format == "verbose_json":
            return JSONResponse(
                {
                    "task": "transcribe",
                    "language": info.language,
                    "duration": info.duration,
                    "text": full_text,
                    "segments": [
                        {
                            "id": i,
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text.strip(),
                            "words": (
                                [
                                    {
                                        "word": w.word,
                                        "start": w.start,
                                        "end": w.end,
                                        "probability": w.probability,
                                    }
                                    for w in seg.words
                                ]
                                if seg.words
                                else []
                            ),
                        }
                        for i, seg in enumerate(segments_list)
                    ],
                }
            )

        elif response_format == "vtt":
            vtt_lines = ["WEBVTT", ""]
            for seg in segments_list:
                start = format_timestamp_vtt(seg.start)
                end = format_timestamp_vtt(seg.end)
                vtt_lines.append(f"{start} --> {end}")
                vtt_lines.append(seg.text.strip())
                vtt_lines.append("")
            return PlainTextResponse("\n".join(vtt_lines), media_type="text/vtt")

        elif response_format == "srt":
            srt_lines = []
            for i, seg in enumerate(segments_list, 1):
                start = format_timestamp_srt(seg.start)
                end = format_timestamp_srt(seg.end)
                srt_lines.append(str(i))
                srt_lines.append(f"{start} --> {end}")
                srt_lines.append(seg.text.strip())
                srt_lines.append("")
            return PlainTextResponse("\n".join(srt_lines), media_type="text/plain")

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported response_format: {response_format}",
            )

    return web_app


# Whisper Large v3 Server
@app.cls(
    image=image,
    gpu="H100",
    memory=32768,  # 32GB RAM
    scaledown_window=300,  # Keep warm for 5 minutes
    timeout=600,  # 10 minutes for long audio files
    volumes={CACHE_DIR: model_cache},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class WhisperLargeV3:
    model_id: str = MODELS["large-v3"]

    @modal.enter(snap=True)
    def load_model(self):
        """Load model at container startup - state will be GPU snapshotted."""
        import os
        import numpy as np
        from scipy.io import wavfile
        from faster_whisper import WhisperModel

        # Set HuggingFace cache to volume
        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"
        os.environ["HF_HUB_DISABLE_XET"] = "1"

        print(f"Loading model: {self.model_id}")
        self.model = WhisperModel(
            self.model_id,
            device="cuda",
            compute_type="float16",
            download_root=f"{CACHE_DIR}/models",
        )
        print(f"Model {self.model_id} loaded successfully")

        # Warmup: Run a short transcription to initialize CUDA kernels
        # This ensures the GPU snapshot includes compiled kernels
        print("Running warmup transcription...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sample_rate = 16000
            # 1 second of silence
            silent_audio = np.zeros(sample_rate, dtype=np.float32)
            wavfile.write(tmp.name, sample_rate, silent_audio)
            segments, _ = self.model.transcribe(tmp.name, language="en")
            list(segments)  # Consume generator
        print("Warmup complete - GPU state will be snapshotted")

        # Commit the volume to persist downloads
        model_cache.commit()

    @modal.asgi_app()
    def serve(self):
        return create_transcription_app("large-v3", self.model)


# Whisper Large v3 Turbo Server
@app.cls(
    image=image,
    gpu="H100",
    memory=32768,  # 32GB RAM
    scaledown_window=300,  # Keep warm for 5 minutes
    timeout=600,  # 10 minutes for long audio files
    volumes={CACHE_DIR: model_cache},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class WhisperLargeV3Turbo:
    model_id: str = MODELS["large-v3-turbo"]

    @modal.enter(snap=True)
    def load_model(self):
        """Load model at container startup - state will be GPU snapshotted."""
        import os
        import numpy as np
        from scipy.io import wavfile
        from faster_whisper import WhisperModel

        # Set HuggingFace cache to volume
        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"
        os.environ["HF_HUB_DISABLE_XET"] = "1"

        print(f"Loading model: {self.model_id}")
        self.model = WhisperModel(
            self.model_id,
            device="cuda",
            compute_type="float16",
            download_root=f"{CACHE_DIR}/models",
        )
        print(f"Model {self.model_id} loaded successfully")

        # Warmup: Run a short transcription to initialize CUDA kernels
        # This ensures the GPU snapshot includes compiled kernels
        print("Running warmup transcription...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sample_rate = 16000
            # 1 second of silence
            silent_audio = np.zeros(sample_rate, dtype=np.float32)
            wavfile.write(tmp.name, sample_rate, silent_audio)
            segments, _ = self.model.transcribe(tmp.name, language="en")
            list(segments)  # Consume generator
        print("Warmup complete - GPU state will be snapshotted")

        # Commit the volume to persist downloads
        model_cache.commit()

    @modal.asgi_app()
    def serve(self):
        return create_transcription_app("large-v3-turbo", self.model)
