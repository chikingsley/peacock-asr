"""Whisper-based transcription of Speechocean762 splits.

Yan et al. 2025, App. D: ``whisper-large-v3`` transcribes each learner
utterance; the transcribed words become the "perceived" reference for the
simulated free-speaking pipeline.

The cache format is JSON keyed by split name with a version/model-id sidecar,
so stale caches are detected when the model id changes. Transcriptions are
written once and re-used across all subsequent training/eval runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import torch
import torchaudio
from datasets import Audio, load_dataset
from torch import Tensor
from tqdm import tqdm

from p014.data import HF_DATASET_ID

TARGET_SAMPLE_RATE = 16_000
CACHE_FORMAT_VERSION = 1
WHISPER_DEFAULT_MODEL = "openai/whisper-large-v3"

# Whisper tends to emit punctuation attached to tokens (e.g. "hello,"). We
# split words on whitespace and strip everything that is not alnum or an
# interior apostrophe so the output matches SpeechOcean762's space-separated
# upper-case reference words.
_WORD_SPLIT = re.compile(r"\s+")
_NON_WORD_CHAR = re.compile(r"[^A-Za-z0-9']+")


@dataclass(frozen=True)
class TranscriptionResult:
    """The Whisper transcription of a single utterance."""

    utterance_id: str
    text: str
    words: tuple[str, ...]


def _normalise_whisper_text(text: str) -> tuple[str, tuple[str, ...]]:
    """Clean raw Whisper output into an uppercase string and a word tuple."""

    stripped = text.strip()
    words: list[str] = []
    for token in _WORD_SPLIT.split(stripped):
        if not token:
            continue
        cleaned = _NON_WORD_CHAR.sub("", token)
        if cleaned:
            words.append(cleaned.upper())
    return " ".join(words), tuple(words)


def _load_waveform(audio_payload: Any) -> Tensor:
    if hasattr(audio_payload, "get_all_samples"):
        samples = audio_payload.get_all_samples()
        waveform = cast(Tensor, samples.data).to(dtype=torch.float32)
        sampling_rate = int(samples.sample_rate)
    else:
        payload = cast(dict[str, Any], audio_payload)
        path = payload.get("path")
        if path is None:
            array = payload["array"]
            sampling_rate = int(payload["sampling_rate"])
            waveform = torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)
        else:
            waveform, sampling_rate = torchaudio.load(path)
            if waveform.dtype != torch.float32:
                waveform = waveform.to(dtype=torch.float32)

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sampling_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sampling_rate, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    return waveform.squeeze(0)


def _cache_paths(cache_dir: Path, split: str) -> tuple[Path, Path]:
    freespeak_dir = cache_dir / "freespeak"
    freespeak_dir.mkdir(parents=True, exist_ok=True)
    return (
        freespeak_dir / f"{split}_transcripts.json",
        freespeak_dir / f"{split}_transcripts.sidecar.json",
    )


def _cache_is_fresh(
    payload_path: Path,
    sidecar_path: Path,
    *,
    model_id: str,
    dataset_id: str,
    expected_count: int,
) -> bool:
    if not (payload_path.exists() and sidecar_path.exists()):
        return False
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        sidecar.get("version") == CACHE_FORMAT_VERSION
        and sidecar.get("model_id") == model_id
        and sidecar.get("dataset_id") == dataset_id
        and sidecar.get("num_examples") == expected_count
    )


def transcribe_split(
    split: str,
    cache_dir: Path,
    whisper_model: str = WHISPER_DEFAULT_MODEL,
    device: torch.device | None = None,
    batch_size: int = 8,
    dataset_id: str = HF_DATASET_ID,
    progress: bool = True,
) -> list[TranscriptionResult]:
    """Transcribe every utterance in ``split`` using Whisper.

    Args:
        split: Dataset split name (``train`` or ``test``).
        cache_dir: Root cache directory. Transcripts are written to
            ``{cache_dir}/freespeak/{split}_transcripts.json`` with a sidecar
            JSON recording version + model id for freshness detection.
        whisper_model: HF repo id of the speech seq2seq model.
        device: Optional torch device override (default: CUDA if available).
        batch_size: How many utterances to forward per Whisper batch.
        dataset_id: HF dataset id (default: SpeechOcean762).
        progress: Display tqdm progress.

    Returns:
        One :class:`TranscriptionResult` per utterance in the split, in dataset
        order.
    """

    payload_path, sidecar_path = _cache_paths(cache_dir, split)
    raw_any: Any = load_dataset(dataset_id, split=split)
    raw_any = raw_any.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    total = len(raw_any)

    if _cache_is_fresh(
        payload_path,
        sidecar_path,
        model_id=whisper_model,
        dataset_id=dataset_id,
        expected_count=total,
    ):
        records = cast(
            list[dict[str, Any]],
            json.loads(payload_path.read_text(encoding="utf-8")),
        )
        return [
            TranscriptionResult(
                utterance_id=str(record["utterance_id"]),
                text=str(record["text"]),
                words=tuple(str(word) for word in record["words"]),
            )
            for record in records
        ]

    resolved_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transformers_module = cast(Any, import_module("transformers"))
    processor_any: Any = transformers_module.AutoProcessor.from_pretrained(whisper_model)
    model_any: Any = transformers_module.AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_model
    )
    model_any = model_any.to(resolved_device)
    model_any.eval()

    results: list[TranscriptionResult] = []
    progress_iter = tqdm(
        range(0, total, batch_size),
        desc=f"whisper/{split}",
        disable=not progress,
    )
    with torch.no_grad():
        for start in progress_iter:
            end = min(start + batch_size, total)
            waveforms: list[Tensor] = []
            utterance_ids: list[str] = []
            for index in range(start, end):
                example = cast(dict[str, Any], raw_any[int(index)])
                utterance_id = str(
                    example.get("id") or example.get("speaker") or index
                )
                utterance_ids.append(utterance_id)
                waveforms.append(_load_waveform(cast(dict[str, Any], example["audio"])))

            inputs_any: Any = processor_any(
                [waveform.cpu().numpy() for waveform in waveforms],
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_features = cast(Tensor, inputs_any.input_features).to(resolved_device)
            generated = cast(Tensor, model_any.generate(input_features=input_features))
            decoded = cast(
                list[str],
                processor_any.batch_decode(generated, skip_special_tokens=True),
            )
            for utterance_id, raw_text in zip(utterance_ids, decoded, strict=True):
                text, words = _normalise_whisper_text(raw_text)
                results.append(
                    TranscriptionResult(
                        utterance_id=utterance_id, text=text, words=words
                    )
                )

    payload_path.write_text(
        json.dumps(
            [
                {
                    "utterance_id": record.utterance_id,
                    "text": record.text,
                    "words": list(record.words),
                }
                for record in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    sidecar_path.write_text(
        json.dumps(
            {
                "version": CACHE_FORMAT_VERSION,
                "model_id": whisper_model,
                "dataset_id": dataset_id,
                "split": split,
                "num_examples": len(results),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results
