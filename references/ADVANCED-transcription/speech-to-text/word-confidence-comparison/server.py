"""FastAPI server for Word Confidence Comparison."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from transcribers import transcribe_all

# Load environment variables
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY", "")

# Validation constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".mp4"}
ALLOWED_MIME_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/x-m4a", "audio/m4a",
    "audio/flac", "audio/x-flac",
    "audio/ogg", "audio/vorbis",
    "audio/webm",
    "audio/aac", "audio/x-aac",
    "video/mp4",  # Some audio in mp4 container
}

app = FastAPI(title="Word Confidence Comparison")


@app.get("/")
async def index():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    providers: Optional[str] = Form(None),
):
    """
    Transcribe audio using selected providers.

    Args:
        audio: Audio file to transcribe
        providers: JSON array of provider names to use

    Returns JSON with results from each provider.
    """
    # Parse selected providers
    selected = []
    if providers:
        try:
            selected = json.loads(providers)
            if not isinstance(selected, list) or not all(isinstance(p, str) for p in selected):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid providers format. Expected JSON array of strings."}
                )
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid providers format. Expected JSON array."}
            )

    # Build providers dict based on selection
    providers_dict = {}

    # API-based providers (require keys)
    if (not selected or "assemblyai" in selected) and ASSEMBLYAI_API_KEY:
        providers_dict["assemblyai"] = ASSEMBLYAI_API_KEY
    if (not selected or "soniox" in selected) and SONIOX_API_KEY:
        providers_dict["soniox"] = SONIOX_API_KEY

    # Whisper providers (via Modal, no API key needed)
    if selected and "whisper-large" in selected:
        providers_dict["whisper-large"] = "large-v3"
    if selected and "whisper-turbo" in selected:
        providers_dict["whisper-turbo"] = "turbo"

    if not providers_dict:
        return JSONResponse(
            status_code=400,
            content={"error": "No providers selected or configured"}
        )

    # Validate content type
    if audio.content_type and audio.content_type not in ALLOWED_MIME_TYPES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid content type: {audio.content_type}"}
        )

    # Validate file extension
    suffix = Path(audio.filename).suffix.lower() if audio.filename else ".wav"
    if suffix not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid audio format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}
        )

    # Read and validate file size
    content = await audio.read()
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=413,
            content={"error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"}
        )

    # Save uploaded file to temp location
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Run transcriptions in thread pool to avoid blocking event loop
        results = await asyncio.to_thread(transcribe_all, tmp_path, providers_dict)

        # Convert to JSON-serializable format
        response = []
        for result in results:
            response.append({
                "provider": result.provider,
                "text": result.text,
                "words": result.words,
                "duration_seconds": result.duration_seconds,
                "error": result.error,
            })

        return JSONResponse(content={"results": response})

    finally:
        # Clean up temp file
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.get("/api/providers")
async def get_providers():
    """Return list of configured providers."""
    providers = []
    if ASSEMBLYAI_API_KEY:
        providers.append("AssemblyAI")
    if SONIOX_API_KEY:
        providers.append("Soniox")
    # Whisper providers are always available (via Modal)
    providers.append("Whisper Large v3")
    providers.append("Whisper Turbo")
    return {"providers": providers}


# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    # Check for API keys
    if not ASSEMBLYAI_API_KEY and not SONIOX_API_KEY:
        print("Warning: No API keys found. Set ASSEMBLYAI_API_KEY and/or SONIOX_API_KEY in .env")
    else:
        enabled = []
        if ASSEMBLYAI_API_KEY:
            enabled.append("AssemblyAI")
        if SONIOX_API_KEY:
            enabled.append("Soniox")
        print(f"Enabled providers: {', '.join(enabled)}")

    uvicorn.run(app, host="127.0.0.1", port=8000)
