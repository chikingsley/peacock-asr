"""HuggingFace-Hub speech datasets -> ``Sample``. FLEURS is the first/canonical one.

Requires the ``datasets`` extra (``uv sync --extra ingest``) and an HF token in the environment
for gated/large pulls. FLEURS audio is already 16 kHz mono.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omni_curator.sample import Sample

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

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
    import soundfile as sf
    from datasets import load_dataset

    audio_dir.mkdir(parents=True, exist_ok=True)
    for hf_split in splits or tuple(_FLEURS_SPLITS):
        split = _FLEURS_SPLITS.get(hf_split, hf_split)
        dataset = load_dataset("google/fleurs", config, split=hf_split, streaming=streaming)
        for example in dataset:
            audio = example["audio"]
            sample_rate = int(audio["sampling_rate"])
            uid = f"{source}_{config}_{hf_split}_{example['id']}"
            clip = audio_dir / f"{uid}.flac"
            sf.write(str(clip), audio["array"], sample_rate, format="FLAC")
            yield Sample(
                id=uid,
                source=source,
                language=language,
                text=str(example.get("transcription") or "").strip(),
                audio_path=str(clip),
                duration=len(audio["array"]) / sample_rate,
                sample_rate=sample_rate,
                split=split,
                citation=f"google/fleurs:{config}",
                meta={"gender": example.get("gender")},
            )
