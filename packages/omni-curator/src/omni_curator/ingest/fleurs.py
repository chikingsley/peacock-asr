"""Google FLEURS -> ``Sample``.

Needs the ``datasets`` dependency (the ``ingest`` extra) and ``HF_TOKEN`` in the environment.
Audio arrives as raw bytes via ``Audio(decode=False)`` and is written as FLAC.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omni_curator.sample import Sample

if TYPE_CHECKING:
    from collections.abc import Iterator

# FLEURS ships train/validation/test; we store them as train/dev/test.
_FLEURS_SPLITS = {"train": "train", "validation": "dev", "test": "test"}


def load_fleurs(
    config: str,
    *,
    language: str,
    audio_dir: Path,
    source: str = "fleurs",
    splits: tuple[str, ...] | None = None,
    streaming: bool = True,
) -> Iterator[Sample]:
    """Stream ``google/fleurs`` ``config`` (e.g. ``"ka_ge"``) -> 16 kHz FLAC clips + ``Sample``s.

    ``language`` is the curator language code stored on each Sample (e.g. ``"kat_Geor"``);
    ``config`` is the HF FLEURS config id (e.g. ``"ka_ge"``).
    """
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    audio_dir.mkdir(parents=True, exist_ok=True)
    for hf_split in splits or tuple(_FLEURS_SPLITS):
        split = _FLEURS_SPLITS.get(hf_split, hf_split)
        # decode=False hands us the raw audio bytes; we decode with soundfile (no torchcodec).
        dataset = load_dataset(
            "google/fleurs", config, split=hf_split, streaming=streaming
        ).cast_column("audio", Audio(decode=False))
        for example in dataset:
            data, sample_rate = sf.read(io.BytesIO(example["audio"]["bytes"]))
            # FLEURS has many speakers per sentence (shared 'id'); the audio path is unique.
            uid = f"{source}_{config}_{hf_split}_{Path(example['audio']['path']).stem}"
            clip = audio_dir / f"{uid}.flac"
            sf.write(str(clip), data, sample_rate, format="FLAC")
            yield Sample(
                id=uid,
                source=source,
                language=language,
                text=str(example.get("transcription") or "").strip(),
                audio_path=str(clip),
                duration=len(data) / sample_rate,
                sample_rate=int(sample_rate),
                split=split,
                citation=f"google/fleurs:{config}",
                meta={"gender": example.get("gender")},
            )
