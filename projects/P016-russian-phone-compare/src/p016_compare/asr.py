from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


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


class ScribeAsrTranscriber:
    """ElevenLabs Scribe v2 via the superwhisper-api realtime stream (file mode).

    Streams the audio through the realtime websocket and returns the final committed
    transcript. In manual-commit (file) mode the server commits once, at the flush, so the
    last committed event carries the full cumulative transcript.
    """

    model_id = "elevenlabs/scribe_v2_realtime"

    def transcribe(self, audio_path: str, language: str | None = None) -> AsrResult:
        import numpy as np

        from p016_compare.audio import load_audio_16k

        audio = load_audio_16k(audio_path)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        text = asyncio.run(_scribe_transcribe(pcm, _scribe_language(language)))
        return AsrResult(text=text, language=language or "", model_id=self.model_id)


async def _scribe_transcribe(pcm: bytes, language: str | None) -> str:
    from superwhisper_api.audio.realtime import (
        ELEVENLABS_MODEL_ID,
        SAMPLE_RATE,
        file_chunks,
        stream_events,
    )

    final = ""
    async for event in stream_events(
        file_chunks(pcm),
        provider_name="elevenlabs",
        model_id=ELEVENLABS_MODEL_ID,
        sample_rate=SAMPLE_RATE,
        language=language,
    ):
        if event.kind == "committed":
            final = event.text
    return final.strip()
