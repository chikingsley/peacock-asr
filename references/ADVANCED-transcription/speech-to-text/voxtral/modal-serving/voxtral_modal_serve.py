#!/usr/bin/env python3
"""
Voxtral STT Serving via vLLM on Modal

Deploys an OpenAI-compatible transcription API using vLLM as the backend.

Usage:
    # Deploy to Modal
    cd modal-serving && uv run modal deploy voxtral_modal_serve.py

    # Test locally first
    uv run modal run --env=dev voxtral_modal_serve.py
"""

import modal
from pathlib import Path

app = modal.App("voxtral-stt-server")

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
CACHE_DIR = "/cache"
VLLM_PORT = 8000

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
VALID_RESPONSE_FORMATS = {"json", "text", "vtt", "srt", "verbose_json"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl")
    .pip_install(
        "vllm>=0.9",
        "fastapi>=0.100.0",
        "python-multipart",
        "uvicorn",
        "httpx",
        "huggingface_hub",
        "python-dotenv",
    )
)

model_cache = modal.Volume.from_name("voxtral-model-cache", create_if_missing=True)


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


@app.cls(
    image=image,
    gpu="H100",
    memory=32768,
    scaledown_window=300,  # 5 minutes
    timeout=600,  # 10 minutes per request
    volumes={CACHE_DIR: model_cache},
)
class VoxtralServer:
    @modal.enter()
    def start_vllm(self):
        """Start vLLM as a subprocess and wait for it to be ready."""
        import subprocess
        import time
        import os
        import httpx

        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"
        os.environ["HF_HUB_DISABLE_XET"] = "1"

        print(f"Starting vLLM server for {MODEL_ID}...")
        self.vllm_process = subprocess.Popen(
            [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", MODEL_ID,
                "--tokenizer-mode", "mistral",
                "--config-format", "mistral",
                "--load-format", "mistral",
                "--dtype", "bfloat16",
                "--port", str(VLLM_PORT),
                "--download-dir", CACHE_DIR,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for vLLM to be ready (up to 5 min)
        max_wait = 300
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                resp = httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=5)
                if resp.status_code == 200:
                    print(f"vLLM ready after {time.time() - start_time:.0f}s")
                    model_cache.commit()
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            time.sleep(5)

        raise TimeoutError(f"vLLM did not start within {max_wait}s")

    @modal.exit()
    def stop_vllm(self):
        if hasattr(self, "vllm_process") and self.vllm_process:
            self.vllm_process.terminate()
            self.vllm_process.wait(timeout=10)

    @modal.asgi_app()
    def serve(self):
        import tempfile
        import httpx
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import JSONResponse, PlainTextResponse

        web_app = FastAPI(
            title="Voxtral STT Server",
            description="OpenAI-compatible transcription API powered by vLLM + Voxtral",
        )

        @web_app.get("/health")
        def health():
            return {"status": "ok", "model": MODEL_ID}

        @web_app.get("/v1/models")
        def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": MODEL_ID,
                        "object": "model",
                        "owned_by": "mistralai",
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
        ):
            """OpenAI-compatible transcription endpoint.

            Accepts audio file upload and returns transcription in the
            requested format (json/text/vtt/srt/verbose_json).
            """
            if response_format not in VALID_RESPONSE_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported response_format: {response_format}. "
                    f"Valid: {', '.join(VALID_RESPONSE_FORMATS)}",
                )

            audio_bytes = await file.read()
            if len(audio_bytes) > 100 * 1024 * 1024:  # 100MB limit
                raise HTTPException(status_code=413, detail="File too large (max 100MB)")

            suffix = Path(file.filename).suffix.lower() if file.filename else ".mp3"
            if suffix not in ALLOWED_EXTENSIONS:
                suffix = ".mp3"

            # Write to temp file and proxy to vLLM
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()

                try:
                    async with httpx.AsyncClient(timeout=300) as client:
                        with open(tmp.name, "rb") as f:
                            vllm_resp = await client.post(
                                f"http://localhost:{VLLM_PORT}/v1/audio/transcriptions",
                                files={"file": (file.filename or "audio" + suffix, f, "audio/mpeg")},
                                data={
                                    "model": MODEL_ID,
                                    "language": language,
                                    "temperature": str(temperature),
                                },
                            )
                except Exception as e:
                    raise HTTPException(
                        status_code=502,
                        detail=f"vLLM backend error: {str(e)}",
                    )

                if vllm_resp.status_code != 200:
                    raise HTTPException(
                        status_code=vllm_resp.status_code,
                        detail=f"vLLM error: {vllm_resp.text}",
                    )

                result = vllm_resp.json()
                full_text = result.get("text", "")

            # Format response
            if response_format == "text":
                return PlainTextResponse(full_text)

            elif response_format == "json":
                return JSONResponse({"text": full_text})

            elif response_format == "verbose_json":
                return JSONResponse(
                    {
                        "task": "transcribe",
                        "language": language,
                        "duration": result.get("duration", 0),
                        "text": full_text,
                        "segments": [
                            {
                                "id": 0,
                                "start": 0.0,
                                "end": result.get("duration", 0),
                                "text": full_text,
                            }
                        ],
                    }
                )

            elif response_format == "vtt":
                duration = result.get("duration", 0)
                start = format_timestamp_vtt(0.0)
                end = format_timestamp_vtt(duration)
                vtt = f"WEBVTT\n\n{start} --> {end}\n{full_text}\n"
                return PlainTextResponse(vtt, media_type="text/vtt")

            elif response_format == "srt":
                duration = result.get("duration", 0)
                start = format_timestamp_srt(0.0)
                end = format_timestamp_srt(duration)
                srt = f"1\n{start} --> {end}\n{full_text}\n"
                return PlainTextResponse(srt, media_type="text/plain")

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported response_format: {response_format}",
                )

        return web_app
