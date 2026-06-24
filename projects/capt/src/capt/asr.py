from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode

SAMPLE_RATE = 16_000
# ~100 ms of 16-bit mono PCM per binary frame, matching the reference replay client.
_CHUNK_MS = 100
_CHUNK_BYTES = SAMPLE_RATE * 2 * _CHUNK_MS // 1000
# Idle ceiling for the realtime session; the relay closes on its own once the vendor
# emits its committed transcript, so this is a safety bound.
_TIMEOUT_S = 30.0
# The relay defaults to ElevenLabs scribe_v2_realtime; this id is reported on results.
_ELEVENLABS_MODEL_ID = "scribe_v2_realtime"
_REALTIME_STREAM_PATH = "/v1/realtime/stream"
_DEFAULT_REALTIME_BASE = "https://api.peacockery.studio"


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str
    model_id: str


class AsrTranscriber(Protocol):
    """ASR lane interface: an audio path + optional language hint -> AsrResult."""

    model_id: str

    def transcribe(self, audio_path: str, language: str | None = None) -> AsrResult: ...


# ElevenLabs Scribe expects ISO 639-3 language hints; None lets it auto-detect.
_SCRIBE_LANGUAGES = {"ru": "rus", "en": "eng"}


def _scribe_language(language: str | None) -> str | None:
    if not language:
        return None
    return _SCRIBE_LANGUAGES.get(language.lower().split("_", 1)[0])


def _load_env_file(start: Path) -> dict[str, str]:
    """Best-effort parse of the nearest ``.env`` (``KEY=VALUE``) walking up from ``start``."""
    for directory in (start, *start.parents):
        env_path = directory / ".env"
        if not env_path.is_file():
            continue
        values: dict[str, str] = {}
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values
    return {}


def _resolve_realtime_target() -> tuple[str, str]:
    """Return ``(base_url, api_key)`` for the Cloudflare realtime relay.

    The relay lives at the Cloudflare Worker (``REALTIME_API_BASE``, default
    ``https://api.peacockery.studio``) and authenticates with a Cloudflare ``pa_live_…``
    bearer key (``REALTIME_API_KEY``). These are distinct from the gmkserver HTTP service
    (``SUPERWHISPER_API_BASE`` + ``sw_live_…``), so they read their own variables. Values
    come from the environment, falling back to the nearest ``.env`` walking up from here.
    """
    base = os.environ.get("REALTIME_API_BASE")
    key = os.environ.get("REALTIME_API_KEY")
    if not base or not key:
        fallback = _load_env_file(Path(__file__).resolve().parent)
        base = base or fallback.get("REALTIME_API_BASE")
        key = key or fallback.get("REALTIME_API_KEY")
    base = base or _DEFAULT_REALTIME_BASE
    if not key:
        msg = (
            "REALTIME_API_KEY must be set to a Cloudflare pa_live_ key "
            "(in the environment or a .env file) for the realtime ASR lane."
        )
        raise RuntimeError(msg)
    return base.rstrip("/"), key


def _ws_url(base: str, language: str | None) -> str:
    """Build the realtime WebSocket URL with config carried as query params.

    The relay route reads config from query params (``audio_format``, ``encoding``,
    ``sample_rate``, ``language``) and forwards them to the Durable Object, so query params
    are the authoritative carrier.
    """
    ws_base = "wss://" + base.removeprefix("https://").removeprefix("http://")
    params = {
        "provider": "elevenlabs",
        "sample_rate": str(SAMPLE_RATE),
        "audio_format": f"pcm_{SAMPLE_RATE}",
        "encoding": "linear16",
    }
    if language:
        params["language"] = language
    return f"{ws_base}{_REALTIME_STREAM_PATH}?{urlencode(params)}"


def _transcript_text(payload: dict[str, object]) -> str:
    """Pull transcript text from a relay message (house schema or raw vendor shape)."""
    text = payload.get("text")
    if isinstance(text, str):
        return text
    channel = payload.get("channel")
    if isinstance(channel, dict):
        alternatives = channel.get("alternatives")
        if isinstance(alternatives, list) and alternatives:
            first = alternatives[0]
            if isinstance(first, dict):
                transcript = first.get("transcript")
                if isinstance(transcript, str):
                    return transcript
    return ""


def _is_final(payload: dict[str, object]) -> bool:
    """A message is final on is_final / speech_final or a committed message_type."""
    committed = {"committed_transcript", "committed_transcript_with_timestamps"}
    return bool(
        payload.get("is_final")
        or payload.get("speech_final")
        or payload.get("message_type") in committed
    )


class ScribeAsrTranscriber:
    """ElevenLabs Scribe v2 realtime via the Cloudflare realtime relay (file mode).

    Streams the audio through the Cloudflare relay WebSocket and returns the final committed
    transcript. In manual-commit (file) mode the relay commits once, at the flush, so the
    last committed event carries the full cumulative transcript.
    """

    model_id = f"elevenlabs/{_ELEVENLABS_MODEL_ID}"

    def transcribe(self, audio_path: str, language: str | None = None) -> AsrResult:
        import numpy as np

        from capt.audio import load_audio_16k

        audio = load_audio_16k(audio_path)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        text = asyncio.run(_scribe_transcribe(pcm, _scribe_language(language)))
        return AsrResult(text=text, language=language or "", model_id=self.model_id)


def _file_chunks(pcm: bytes) -> list[bytes]:
    """Split raw PCM into fixed-size binary frames for the relay."""
    return [pcm[offset : offset + _CHUNK_BYTES] for offset in range(0, len(pcm), _CHUNK_BYTES)]


async def _scribe_transcribe(pcm: bytes, language: str | None) -> str:
    """Stream PCM through the Cloudflare realtime relay and return the committed transcript.

    Mirrors the reference client (``cloudflare-api/cli/commands/realtime.ts``): connect, send
    PCM as binary frames, signal end-of-audio with a ``{"type": "flush"}`` text frame, then
    collect transcript messages until the relay closes. The flush asks the vendor to commit,
    so the latest committed (final) text is the full transcript.
    """
    from websockets.asyncio.client import connect

    base, key = _resolve_realtime_target()
    url = _ws_url(base, language)
    headers = {"Authorization": f"Bearer {key}"}

    final = ""
    latest = ""
    async with connect(url, additional_headers=headers, open_timeout=_TIMEOUT_S) as ws:
        for chunk in _file_chunks(pcm):
            await ws.send(chunk)
        await ws.send(json.dumps({"type": "flush"}))

        try:
            async with asyncio.timeout(_TIMEOUT_S):
                async for message in ws:
                    if isinstance(message, bytes):
                        continue
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    text = _transcript_text(payload)
                    if not text:
                        continue
                    latest = text
                    if _is_final(payload):
                        final = text
        except TimeoutError:
            pass

    return (final or latest).strip()
