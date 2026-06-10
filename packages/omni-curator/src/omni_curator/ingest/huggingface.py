"""Generic HuggingFace audio datasets -> ``Sample``."""

from __future__ import annotations

import io
from itertools import chain
from typing import TYPE_CHECKING, cast

from omni_curator.sample import Sample

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

_AUDIO_COLUMNS = ("audio",)
_MONO_DIMS = 1
_SAMPLE_RATE = 16_000
_TEXT_COLUMNS = ("transcript", "sentence", "text", "normalized_text")


def _column_names(dataset: object) -> set[str]:
    columns = getattr(dataset, "column_names", None)
    if columns:
        return set(columns)
    features = getattr(dataset, "features", None)
    if features:
        return set(features)
    return set()


def _audio_column(column: str | None, columns: set[str]) -> str:
    if column is not None:
        return column
    for candidate in _AUDIO_COLUMNS:
        if candidate in columns:
            return candidate
    if not columns:
        return _AUDIO_COLUMNS[0]
    raise ValueError("could not detect HF audio column; pass audio_column=")


def _text_column(column: str | None, columns: set[str]) -> str | None:
    if column is not None:
        return column
    for candidate in _TEXT_COLUMNS:
        if candidate in columns:
            return candidate
    if columns:
        raise ValueError("could not detect HF text column; pass text_column=")
    return None


def _text_column_from_example(example: dict[str, object]) -> str:
    for candidate in _TEXT_COLUMNS:
        if candidate in example:
            return candidate
    raise ValueError("could not detect HF text column; pass text_column=")


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _load_split(
    load_dataset: Any, repo: str, config: str | None, split: str, *, streaming: bool
) -> Any:
    """Load one split, tolerating absent ones: ``dev`` falls back to HF's usual
    ``validation`` name, and a split the dataset simply doesn't have returns ``None``
    (the caller skips it) — dataset split layouts vary too much to hard-require all three.
    """

    def _load(name: str) -> Any:
        if config is None:
            return load_dataset(repo, split=name, streaming=streaming)
        return load_dataset(repo, config, split=name, streaming=streaming)

    candidates = (split, "validation") if split == "dev" else (split,)
    for name in candidates:
        try:
            return _load(name)
        except ValueError as exc:  # datasets raises ValueError("Bad split: ...")
            if "Bad split" not in str(exc):
                raise
    return None


def _read_audio(audio: object, soundfile: Any) -> tuple[Any, int]:
    if not isinstance(audio, dict):
        raise TypeError("HF audio column must yield a dict after Audio(decode=False)")
    record = cast("dict[str, Any]", audio)
    raw = record.get("bytes")
    path = record.get("path")
    if raw is not None:
        return soundfile.read(io.BytesIO(raw))
    if path is not None:
        return soundfile.read(str(path))
    raise ValueError("HF audio record has neither bytes nor path")


def _to_16k_mono(data: object, sample_rate: int, *, librosa: Any, numpy: Any) -> Any:
    samples = numpy.asarray(data)
    if samples.ndim > _MONO_DIMS:
        samples = numpy.mean(samples, axis=1)
    if sample_rate != _SAMPLE_RATE:
        return librosa.resample(samples, orig_sr=sample_rate, target_sr=_SAMPLE_RATE)
    return samples


def load_hf_audio(
    repo: str,
    *,
    language: str,
    source: str,
    config: str | None = None,
    splits: tuple[str, ...] = ("train", "dev", "test"),
    audio_column: str | None = None,
    text_column: str | None = None,
    audio_dir: Path,
    streaming: bool = True,
) -> Iterator[Sample]:
    """Stream any HF audio dataset -> 16 kHz mono FLAC clips + ``Sample``s."""
    import librosa
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    audio_dir.mkdir(parents=True, exist_ok=True)
    source_slug = _slug(source)
    repo_slug = _slug(repo)
    config_slug = _slug(config) if config is not None else None

    for split in splits:
        loaded = _load_split(load_dataset, repo, config, split, streaming=streaming)
        if loaded is None:  # the dataset doesn't have this split
            continue
        audio_col = _audio_column(audio_column, _column_names(loaded))
        dataset = loaded.cast_column(audio_col, Audio(decode=False))
        detected_text_col = _text_column(text_column, _column_names(dataset))
        rows = iter(dataset)
        first = next(rows, None)
        if first is None:
            continue
        row_text_col = detected_text_col or _text_column_from_example(first)
        for index, example in enumerate(chain((first,), rows)):
            text = str(example.get(row_text_col) or "").strip()
            data, sample_rate = _read_audio(example[audio_col], sf)
            samples = _to_16k_mono(data, sample_rate, librosa=librosa, numpy=np)
            uid_parts = [source_slug, repo_slug]
            if config_slug is not None:
                uid_parts.append(config_slug)
            uid_parts.extend([_slug(split), str(index)])
            uid = "_".join(uid_parts)
            clip = audio_dir / f"{uid}.flac"
            sf.write(str(clip), samples, _SAMPLE_RATE, format="FLAC")
            yield Sample(
                id=uid,
                source=source,
                language=language,
                text=text,
                audio_path=str(clip),
                duration=len(samples) / _SAMPLE_RATE,
                sample_rate=_SAMPLE_RATE,
                split=split,
                citation=repo,
                meta={"config": config, "audio_column": audio_col, "text_column": row_text_col},
            )
