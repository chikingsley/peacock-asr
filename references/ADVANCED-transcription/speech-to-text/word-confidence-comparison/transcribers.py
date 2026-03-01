"""Transcription clients for different providers."""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import assemblyai as aai
import httpx

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from a transcription provider."""
    provider: str
    text: str
    words: List[Dict]  # [{text, start_ms, end_ms, confidence}]
    duration_seconds: float  # Time taken to transcribe
    error: Optional[str] = None


def transcribe_assemblyai(audio_path: str, api_key: str) -> TranscriptionResult:
    """
    Transcribe audio using AssemblyAI.

    Returns word-level data with confidence scores.
    """
    start_time = time.time()

    try:
        aai.settings.api_key = api_key

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path)

        if transcript.status == aai.TranscriptStatus.error:
            return TranscriptionResult(
                provider="AssemblyAI",
                text="",
                words=[],
                duration_seconds=time.time() - start_time,
                error=f"Transcription failed: {transcript.error}"
            )

        # Extract word-level data
        words = []
        if transcript.words:
            for word in transcript.words:
                words.append({
                    "text": word.text,
                    "start_ms": word.start,
                    "end_ms": word.end,
                    "confidence": word.confidence,
                })

        return TranscriptionResult(
            provider="AssemblyAI",
            text=transcript.text or "",
            words=words,
            duration_seconds=time.time() - start_time,
        )

    except Exception as e:
        logger.exception("AssemblyAI transcription failed")
        return TranscriptionResult(
            provider="AssemblyAI",
            text="",
            words=[],
            duration_seconds=time.time() - start_time,
            error="Transcription failed. Please try again."
        )


def transcribe_soniox(audio_path: str, api_key: str) -> TranscriptionResult:
    """
    Transcribe audio using Soniox Async API (batch processing).

    Returns word-level data with confidence scores.
    """
    start_time = time.time()
    base_url = "https://api.soniox.com/v1"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        # 1. Upload file to get file_id
        with open(audio_path, "rb") as f:
            files = {"file": (Path(audio_path).name, f)}
            resp = httpx.post(
                f"{base_url}/files",
                headers=headers,
                files=files,
                timeout=60.0,
            )

        if resp.status_code not in (200, 201):
            return TranscriptionResult(
                provider="Soniox",
                text="",
                words=[],
                duration_seconds=time.time() - start_time,
                error=f"File upload failed: {resp.status_code} - {resp.text}",
            )

        file_id = resp.json().get("id")
        if not file_id:
            return TranscriptionResult(
                provider="Soniox",
                text="",
                words=[],
                duration_seconds=time.time() - start_time,
                error=f"No file ID returned: {resp.text}",
            )

        # 2. Create transcription with file_id
        json_headers = {**headers, "Content-Type": "application/json"}
        resp = httpx.post(
            f"{base_url}/transcriptions",
            headers=json_headers,
            json={"model": "stt-async-v3", "file_id": file_id},
            timeout=30.0,
        )

        if resp.status_code not in (200, 201):
            return TranscriptionResult(
                provider="Soniox",
                text="",
                words=[],
                duration_seconds=time.time() - start_time,
                error=f"Create transcription failed: {resp.status_code} - {resp.text}",
            )

        transcription_id = resp.json().get("id")
        if not transcription_id:
            return TranscriptionResult(
                provider="Soniox",
                text="",
                words=[],
                duration_seconds=time.time() - start_time,
                error=f"No transcription ID returned: {resp.text}",
            )

        # 3. Poll until complete
        max_polls = 120  # 2 minutes max
        for _ in range(max_polls):
            resp = httpx.get(
                f"{base_url}/transcriptions/{transcription_id}",
                headers=headers,
                timeout=30.0,
            )
            if resp.status_code != 200:
                return TranscriptionResult(
                    provider="Soniox",
                    text="",
                    words=[],
                    duration_seconds=time.time() - start_time,
                    error=f"Status check failed: {resp.status_code} - {resp.text}",
                )
            status_data = resp.json()
            status = status_data.get("status")

            if status == "completed":
                break
            elif status == "failed":
                return TranscriptionResult(
                    provider="Soniox",
                    text="",
                    words=[],
                    duration_seconds=time.time() - start_time,
                    error=f"Transcription failed: {status_data.get('error', 'Unknown')}",
                )

            time.sleep(1)
        else:
            return TranscriptionResult(
                provider="Soniox",
                text="",
                words=[],
                duration_seconds=time.time() - start_time,
                error="Transcription timed out",
            )

        # 4. Get transcript with word-level data
        resp = httpx.get(
            f"{base_url}/transcriptions/{transcription_id}/transcript",
            headers=headers,
            timeout=30.0,
        )

        if resp.status_code != 200:
            return TranscriptionResult(
                provider="Soniox",
                text="",
                words=[],
                duration_seconds=time.time() - start_time,
                error=f"Failed to get transcript: {resp.status_code} - {resp.text}",
            )

        transcript_data = resp.json()

        # Parse tokens into words
        # Async API returns tokens similar to realtime
        all_tokens = transcript_data.get("tokens", [])
        full_text_parts = []
        words = []
        current_word = None

        for token in all_tokens:
            text = token.get("text", "")
            if not text.strip():
                continue

            full_text_parts.append(text)
            is_word_start = text.startswith(" ") or current_word is None

            if is_word_start:
                if current_word is not None:
                    words.append(current_word)
                current_word = {
                    "text": text.lstrip(),
                    "start_ms": token.get("start_ms", 0),
                    "end_ms": token.get("end_ms", token.get("start_ms", 0)),
                    "confidence": token.get("confidence", 1.0),
                    "_conf_sum": token.get("confidence", 1.0),
                    "_conf_count": 1,
                }
            else:
                current_word["text"] += text
                current_word["end_ms"] = token.get("end_ms", current_word["end_ms"])
                current_word["_conf_sum"] += token.get("confidence", 1.0)
                current_word["_conf_count"] += 1
                current_word["confidence"] = (
                    current_word["_conf_sum"] / current_word["_conf_count"]
                )

        if current_word is not None:
            words.append(current_word)

        # Clean up internal fields
        for word in words:
            word.pop("_conf_sum", None)
            word.pop("_conf_count", None)

        return TranscriptionResult(
            provider="Soniox",
            text="".join(full_text_parts).strip(),
            words=words,
            duration_seconds=time.time() - start_time,
        )

    except Exception as e:
        logger.exception("Soniox transcription failed")
        return TranscriptionResult(
            provider="Soniox",
            text="",
            words=[],
            duration_seconds=time.time() - start_time,
            error="Transcription failed. Please try again.",
        )


def transcribe_whisper(audio_path: str, model_size: str = "large-v3") -> TranscriptionResult:
    """
    Transcribe audio using Whisper via Modal (faster-whisper-server).

    The Modal server exposes an OpenAI-compatible API endpoint.

    Args:
        audio_path: Path to audio file
        model_size: "large-v3" or "large-v3-turbo"

    Returns word-level data with confidence scores.
    """
    start_time = time.time()
    provider_name = f"Whisper {model_size}"

    # Modal web endpoint URLs (deployed via modal deploy)
    # Format: https://<workspace>--<app>-<class>-serve.modal.run
    modal_workspace = os.getenv("MODAL_WORKSPACE", "trelisresearch")
    MODAL_ENDPOINTS = {
        "large-v3": f"https://{modal_workspace}--faster-whisper-server-whisperlargev3-serve.modal.run",
        "large-v3-turbo": f"https://{modal_workspace}--faster-whisper-server-whisperlargev3turbo-serve.modal.run",
    }

    base_url = MODAL_ENDPOINTS.get(model_size)
    if not base_url:
        return TranscriptionResult(
            provider=provider_name,
            text="",
            words=[],
            duration_seconds=time.time() - start_time,
            error=f"Unknown model size: {model_size}",
        )

    try:
        # Read audio file
        filename = Path(audio_path).name
        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, "audio/mpeg")}
            data = {
                "response_format": "verbose_json",  # Get word-level timestamps
                "language": "en",
            }

            # Call Modal web endpoint (longer timeout for cold starts)
            resp = httpx.post(
                f"{base_url}/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=180.0,  # 3 min for cold start + transcription
            )

        if resp.status_code != 200:
            logger.error(f"Whisper ({model_size}) failed: {resp.status_code} - {resp.text}")
            return TranscriptionResult(
                provider=provider_name,
                text="",
                words=[],
                duration_seconds=time.time() - start_time,
                error="Transcription failed. Please try again.",
            )

        result = resp.json()

        # Parse verbose_json response - words are nested in segments
        words = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                words.append({
                    "text": word.get("word", "").strip(),
                    "start_ms": int(word.get("start", 0) * 1000),
                    "end_ms": int(word.get("end", 0) * 1000),
                    "confidence": word.get("probability", 1.0),
                })

        return TranscriptionResult(
            provider=provider_name,
            text=result.get("text", ""),
            words=words,
            duration_seconds=time.time() - start_time,
        )

    except httpx.TimeoutException:
        return TranscriptionResult(
            provider=provider_name,
            text="",
            words=[],
            duration_seconds=time.time() - start_time,
            error="Request timed out (server may be cold starting)",
        )
    except Exception as e:
        logger.exception(f"Whisper ({model_size}) transcription failed")
        return TranscriptionResult(
            provider=provider_name,
            text="",
            words=[],
            duration_seconds=time.time() - start_time,
            error="Transcription failed. Please try again.",
        )


def transcribe_all(audio_path: str, providers: Dict[str, str]) -> List[TranscriptionResult]:
    """
    Transcribe audio using all specified providers.

    Args:
        audio_path: Path to audio file
        providers: Dict of provider name -> config (API key or model size)

    Returns:
        List of TranscriptionResult
    """
    results = []

    if "assemblyai" in providers:
        results.append(transcribe_assemblyai(audio_path, providers["assemblyai"]))

    if "soniox" in providers:
        results.append(transcribe_soniox(audio_path, providers["soniox"]))

    if "whisper-large" in providers:
        results.append(transcribe_whisper(audio_path, "large-v3"))

    if "whisper-turbo" in providers:
        results.append(transcribe_whisper(audio_path, "large-v3-turbo"))

    return results
