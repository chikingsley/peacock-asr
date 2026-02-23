from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import datasets

logger = logging.getLogger(__name__)

RE_STRESS = re.compile(r"^([A-Z]+)\d?$")


@dataclass
class Utterance:
    utterance_id: str
    audio: np.ndarray
    sample_rate: int
    phones: list[str]
    phone_scores: list[float]
    words: list[str]
    word_accuracies: list[float]
    sentence_accuracy: float
    sentence_fluency: float
    sentence_total: float


@dataclass
class SpeechOcean762:
    train: list[Utterance] = field(default_factory=list)
    test: list[Utterance] = field(default_factory=list)


def strip_stress(phone: str) -> str:
    m = RE_STRESS.match(phone)
    return m.group(1) if m else phone


SAMPLE_RATE = 16000


def _parse_split(ds_split: datasets.Dataset, limit: int = 0) -> list[Utterance]:
    import io  # noqa: PLC0415

    import soundfile as sf  # noqa: PLC0415

    utterances = []
    total = min(limit, len(ds_split)) if limit > 0 else len(ds_split)

    for i in range(total):
        sample = ds_split[i]
        phones: list[str] = []
        phone_scores: list[float] = []
        words: list[str] = []
        word_accuracies: list[float] = []

        for word_info in sample["words"]:
            words.append(word_info["text"])
            word_accuracies.append(float(word_info["accuracy"]))

            for phone, score in zip(
                word_info["phones"], word_info["phones-accuracy"], strict=True
            ):
                clean = strip_stress(phone)
                if clean != "<unk>":
                    phones.append(clean)
                    phone_scores.append(float(score))

        # Decode audio with soundfile (fast) instead of torchcodec (slow)
        audio_raw = sample["audio"]
        audio_array, sr = sf.read(io.BytesIO(audio_raw["bytes"]))
        if sr != SAMPLE_RATE:
            logger.warning("Unexpected sample rate %d for utt_%05d", sr, i)

        utterances.append(
            Utterance(
                utterance_id=f"utt_{i:05d}",
                audio=audio_array.astype(np.float32),
                sample_rate=sr,
                phones=phones,
                phone_scores=phone_scores,
                words=words,
                word_accuracies=word_accuracies,
                sentence_accuracy=float(sample["accuracy"]),
                sentence_fluency=float(sample["fluency"]),
                sentence_total=float(sample["total"]),
            )
        )

        if (i + 1) % 500 == 0:
            logger.info("  parsed %d/%d samples", i + 1, total)

    return utterances


DATASET_REVISION = "f95618ea1353303f34cf186b9c310fa2c1eb02c8"


def load_speechocean762(*, limit: int = 0) -> SpeechOcean762:
    import datasets as ds_lib  # noqa: PLC0415

    logger.info("Loading SpeechOcean762 from HuggingFace...")
    dataset = ds_lib.load_dataset("mispeech/speechocean762", revision=DATASET_REVISION)

    # Disable torchcodec auto-decoding — we use soundfile instead (much faster)
    for split in dataset:
        dataset[split] = dataset[split].cast_column("audio", ds_lib.Audio(decode=False))

    n_train = min(limit, len(dataset["train"])) if limit > 0 else len(dataset["train"])
    n_test = min(limit, len(dataset["test"])) if limit > 0 else len(dataset["test"])

    logger.info(
        "Parsing train split (%d/%d samples)...", n_train, len(dataset["train"])
    )
    train = _parse_split(dataset["train"], limit=limit)

    logger.info("Parsing test split (%d/%d samples)...", n_test, len(dataset["test"]))
    test = _parse_split(dataset["test"], limit=limit)

    return SpeechOcean762(train=train, test=test)
