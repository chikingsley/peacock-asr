"""Generic HuggingFace audio datasets -> ``Sample``."""

from __future__ import annotations

import hashlib
import io
from itertools import chain
from typing import TYPE_CHECKING, cast

from omni_curator.data.sample import Sample
from omni_curator.ingest._util import slug as _slug

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

_AUDIO_COLUMNS = ("audio",)
_MONO_DIMS = 1
_SAMPLE_RATE = 16_000
_TEXT_COLUMNS = ("transcript", "sentence", "text", "normalized_text")
_ID_COLUMNS = ("id", "segment_id", "utterance_id", "audio_id")
_SPEAKER_COLUMNS = ("speaker_id", "speaker", "client_id")


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


def _optional_column(
    column: str | None, columns: set[str], candidates: tuple[str, ...]
) -> str | None:
    if column is not None:
        if columns and column not in columns:
            raise ValueError(f"HF column {column!r} does not exist")
        return column
    return next((candidate for candidate in candidates if candidate in columns), None)


def _derived_split(
    original: str, *, group: str, validation_fraction: float, split_seed: int
) -> str:
    is_training_split = original == "train" or original.startswith("train.")
    if not is_training_split:
        return original
    if validation_fraction <= 0:
        return "train"
    identity = f"{split_seed}\0{group}".encode()
    bucket = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") / 2**64
    return "dev" if bucket < validation_fraction else "train"


def _load_split(
    load_dataset: Any,
    repo: str,
    config: str | None,
    split: str,
    *,
    revision: str | None,
    streaming: bool,
) -> Any:
    """Load one split, tolerating absent ones: ``dev`` falls back to HF's usual
    ``validation`` name, and a split the dataset simply doesn't have returns ``None``
    (the caller skips it) — dataset split layouts vary too much to hard-require all three.
    """

    def _load(name: str) -> Any:
        if config is None:
            return load_dataset(repo, split=name, revision=revision, streaming=streaming)
        return load_dataset(repo, config, split=name, revision=revision, streaming=streaming)

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


def load_hf_audio(  # noqa: PLR0915
    repo: str,
    *,
    language: str,
    source: str,
    config: str | None = None,
    revision: str | None = None,
    splits: tuple[str, ...] = ("train", "dev", "test"),
    audio_column: str | None = None,
    text_column: str | None = None,
    id_column: str | None = None,
    speaker_column: str | None = None,
    audio_dir: Path,
    streaming: bool = False,
    max_hours_per_split: float | None = None,
    shuffle_seed: int | None = None,
    shuffle_buffer_size: int = 256,
    validation_fraction: float = 0.0,
    split_group_column: str | None = None,
    split_seed: int = 17,
) -> Iterator[Sample]:
    """Load a pinned, optionally bounded HF audio dataset as 16 kHz mono FLAC samples."""
    import librosa
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    audio_dir.mkdir(parents=True, exist_ok=True)
    source_slug = _slug(source)
    repo_slug = _slug(repo)
    config_slug = _slug(config) if config is not None else None

    for split in splits:
        loaded = _load_split(
            load_dataset, repo, config, split, revision=revision, streaming=streaming
        )
        if loaded is None:  # the dataset doesn't have this split
            continue
        if shuffle_seed is not None:
            shuffle_kwargs = {"seed": shuffle_seed}
            if streaming:
                shuffle_kwargs["buffer_size"] = shuffle_buffer_size
            loaded = loaded.shuffle(**shuffle_kwargs)
        audio_col = _audio_column(audio_column, _column_names(loaded))
        dataset = loaded.cast_column(audio_col, Audio(decode=False))
        columns = _column_names(dataset)
        detected_text_col = _text_column(text_column, columns)
        detected_id_col = _optional_column(id_column, columns, _ID_COLUMNS)
        detected_speaker_col = _optional_column(speaker_column, columns, _SPEAKER_COLUMNS)
        detected_split_group_col = _optional_column(split_group_column, columns, ())
        rows = iter(dataset)
        first = next(rows, None)
        if first is None:
            continue
        row_text_col = detected_text_col or _text_column_from_example(first)
        split_seconds = 0.0
        for index, example in enumerate(chain((first,), rows)):
            text = str(example.get(row_text_col) or "").strip()
            data, sample_rate = _read_audio(example[audio_col], sf)
            samples = _to_16k_mono(data, sample_rate, librosa=librosa, numpy=np)
            source_id = (
                str(example.get(detected_id_col) or index) if detected_id_col else str(index)
            )
            split_group = (
                str(example.get(detected_split_group_col) or source_id)
                if detected_split_group_col
                else source_id
            )
            output_split = _derived_split(
                split,
                group=split_group,
                validation_fraction=validation_fraction,
                split_seed=split_seed,
            )
            uid_parts = [source_slug, repo_slug]
            if config_slug is not None:
                uid_parts.append(config_slug)
            identity = "\0".join((repo, config or "", split, source_id))
            uid_parts.extend([_slug(split), hashlib.sha256(identity.encode()).hexdigest()[:20]])
            uid = "_".join(uid_parts)
            clip = audio_dir / f"{uid}.flac"
            sf.write(str(clip), samples, _SAMPLE_RATE, format="FLAC")
            duration = len(samples) / _SAMPLE_RATE
            metadata = {
                "config": config,
                "revision": revision,
                "audio_column": audio_col,
                "text_column": row_text_col,
                "source_id": source_id,
                "split_group": split_group,
            }
            if detected_speaker_col is not None:
                metadata["speaker_id"] = example.get(detected_speaker_col)
            yield Sample(
                id=uid,
                source=source,
                language=language,
                text=text,
                audio_path=str(clip),
                duration=duration,
                sample_rate=_SAMPLE_RATE,
                split=output_split,
                citation=repo,
                meta=metadata,
            )
            split_seconds += duration
            if max_hours_per_split is not None and split_seconds >= max_hours_per_split * 3600:
                break
