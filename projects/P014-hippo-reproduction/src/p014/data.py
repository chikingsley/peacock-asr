from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Audio, load_dataset
from torch import Tensor
from torch.utils.data import Dataset

from p014.features import (
    extract_gop_for_split,
    extract_ssl_utterance_for_split,
)

HF_DATASET_ID = "mispeech/speechocean762"
UTTERANCE_SCORE_KEYS = ("accuracy", "completeness", "fluency", "prosodic", "total")
WORD_SCORE_KEYS = ("accuracy", "stress", "total")
SSL_UTT_DIM = 3072


@dataclass(frozen=True)
class ReadAloudAnnotation:
    text: str
    words: tuple[str, ...]
    phone_tokens: tuple[str, ...]
    phone_to_word: tuple[int, ...]
    phone_scores: tuple[float, ...]
    word_scores: tuple[tuple[float, float, float], ...]
    utterance_scores: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class ReadAloudSample:
    gop_features: Tensor
    ssl_utterance: Tensor
    phone_ids: Tensor
    phone_to_word: Tensor
    word_embeddings: Tensor
    phone_targets: Tensor
    word_targets: Tensor
    utterance_targets: Tensor


@dataclass(frozen=True)
class ReadAloudBatch:
    gop_features: Tensor
    ssl_utterance: Tensor
    phone_ids: Tensor
    phone_to_word: Tensor
    word_embeddings: Tensor
    phone_mask: Tensor
    word_mask: Tensor
    phone_targets: Tensor
    word_targets: Tensor
    utterance_targets: Tensor


def load_read_aloud_resources(
    cache_dir: Path,
    modernbert_model: str,
    embedding_device: torch.device,
    max_train_examples: int | None = None,
    max_test_examples: int | None = None,
    feature_device: torch.device | None = None,
    dataset_id: str = HF_DATASET_ID,
) -> tuple[ReadAloudFeatureDataset, ReadAloudFeatureDataset, int, int]:
    """Load / build all inputs required by the HiPPO read-aloud pipeline.

    Returns ``(train_dataset, test_dataset, num_phone_tokens, gop_dim)``.
    """

    train_annotations = load_annotations(
        "train", cache_dir=cache_dir, dataset_id=dataset_id
    )
    test_annotations = load_annotations(
        "test", cache_dir=cache_dir, dataset_id=dataset_id
    )
    if max_train_examples is not None:
        train_annotations = train_annotations[:max_train_examples]
    if max_test_examples is not None:
        test_annotations = test_annotations[:max_test_examples]

    gop_train_path = extract_gop_for_split(
        split="train",
        cache_dir=cache_dir,
        dataset_id=dataset_id,
        device=feature_device,
        max_examples=max_train_examples,
    )
    gop_test_path = extract_gop_for_split(
        split="test",
        cache_dir=cache_dir,
        dataset_id=dataset_id,
        device=feature_device,
        max_examples=max_test_examples,
    )
    ssl_train_path = extract_ssl_utterance_for_split(
        split="train",
        cache_dir=cache_dir,
        dataset_id=dataset_id,
        device=feature_device,
        max_examples=max_train_examples,
    )
    ssl_test_path = extract_ssl_utterance_for_split(
        split="test",
        cache_dir=cache_dir,
        dataset_id=dataset_id,
        device=feature_device,
        max_examples=max_test_examples,
    )

    gop_train = _load_gop_cache(gop_train_path)
    gop_test = _load_gop_cache(gop_test_path)
    ssl_train = _load_ssl_cache(ssl_train_path)
    ssl_test = _load_ssl_cache(ssl_test_path)

    gop_dim = gop_train.gop_dim
    if gop_test.gop_dim != gop_dim:
        raise ValueError(
            f"train/test GOP caches disagree on dim: {gop_dim} vs {gop_test.gop_dim}"
        )

    phone_vocab = build_phone_vocab(train_annotations, test_annotations)
    word_embedding_cache = load_or_build_word_embeddings(
        train_annotations=train_annotations,
        test_annotations=test_annotations,
        cache_dir=cache_dir,
        model_name=modernbert_model,
        device=embedding_device,
    )
    train_dataset = ReadAloudFeatureDataset(
        annotations=train_annotations,
        phone_vocab=phone_vocab,
        word_embeddings=word_embedding_cache["train"],
        gop_features=gop_train.features,
        ssl_utterance=ssl_train.features,
    )
    test_dataset = ReadAloudFeatureDataset(
        annotations=test_annotations,
        phone_vocab=phone_vocab,
        word_embeddings=word_embedding_cache["test"],
        gop_features=gop_test.features,
        ssl_utterance=ssl_test.features,
    )
    return train_dataset, test_dataset, len(phone_vocab), gop_dim


def load_freespeak_resources(
    cache_dir: Path,
    modernbert_model: str,
    embedding_device: torch.device,
    max_train_examples: int | None = None,
    max_test_examples: int | None = None,
    feature_device: torch.device | None = None,
    dataset_id: str = HF_DATASET_ID,
    whisper_model: str | None = None,
) -> tuple[ReadAloudFeatureDataset, ReadAloudFeatureDataset, int, int]:
    """Load / build all inputs for the HiPPO free-speaking pipeline.

    Pipeline (Yan et al. 2025, App. D):

    1. Load the reference read-aloud annotations (same as
       :func:`load_read_aloud_resources`) so we keep utterance-level targets.
    2. Run Whisper to obtain ASR transcripts for each utterance.
    3. Convert the ASR word list via g2pE, then align against the reference
       words/phones to produce free-speaking :class:`ReadAloudAnnotation`
       records.
    4. Re-extract CTC-GOP features using the ASR-derived canonical phone list
       (separate ``_freespeak`` cache).
    5. Utterance-level SSL features are phone-sequence independent, so we
       reuse the read-aloud cache unchanged.
    6. Rebuild ModernBERT word embeddings against the ASR word lists (separate
       ``_freespeak`` cache).

    Returns ``(train_dataset, test_dataset, num_phone_tokens, gop_dim)``.
    """

    from p014.config import FreeSpeakingAssignmentConfig
    from p014.freespeak import (
        TranscriptionResult,
        build_freespeak_annotations,
        transcribe_split,
    )
    from p014.freespeak.g2p import grapheme_to_phones

    ref_train = load_annotations("train", cache_dir=cache_dir, dataset_id=dataset_id)
    ref_test = load_annotations("test", cache_dir=cache_dir, dataset_id=dataset_id)
    if max_train_examples is not None:
        ref_train = ref_train[:max_train_examples]
    if max_test_examples is not None:
        ref_test = ref_test[:max_test_examples]

    whisper_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "device": feature_device,
        "dataset_id": dataset_id,
    }
    if whisper_model is not None:
        whisper_kwargs["whisper_model"] = whisper_model
    transcripts_train_full = transcribe_split(split="train", **whisper_kwargs)
    transcripts_test_full = transcribe_split(split="test", **whisper_kwargs)
    transcripts_train = (
        transcripts_train_full[:max_train_examples]
        if max_train_examples is not None
        else transcripts_train_full
    )
    transcripts_test = (
        transcripts_test_full[:max_test_examples]
        if max_test_examples is not None
        else transcripts_test_full
    )

    assignment = FreeSpeakingAssignmentConfig()
    fs_train = build_freespeak_annotations(ref_train, transcripts_train, assignment)
    fs_test = build_freespeak_annotations(ref_test, transcripts_test, assignment)

    def _per_word_phones(
        transcripts: list[TranscriptionResult],
    ) -> list[list[str]]:
        phone_lists: list[list[str]] = []
        for transcript in transcripts:
            per_word = grapheme_to_phones(list(transcript.words))
            flat: list[str] = []
            for word_phones in per_word:
                flat.extend(word_phones)
            phone_lists.append(flat)
        return phone_lists

    fs_phones_train = _per_word_phones(list(transcripts_train))
    fs_phones_test = _per_word_phones(list(transcripts_test))

    gop_train_path = extract_gop_for_split(
        split="train",
        cache_dir=cache_dir,
        dataset_id=dataset_id,
        device=feature_device,
        max_examples=max_train_examples,
        phones_per_utterance=fs_phones_train,
        suffix="_freespeak",
    )
    gop_test_path = extract_gop_for_split(
        split="test",
        cache_dir=cache_dir,
        dataset_id=dataset_id,
        device=feature_device,
        max_examples=max_test_examples,
        phones_per_utterance=fs_phones_test,
        suffix="_freespeak",
    )
    ssl_train_path = extract_ssl_utterance_for_split(
        split="train",
        cache_dir=cache_dir,
        dataset_id=dataset_id,
        device=feature_device,
        max_examples=max_train_examples,
    )
    ssl_test_path = extract_ssl_utterance_for_split(
        split="test",
        cache_dir=cache_dir,
        dataset_id=dataset_id,
        device=feature_device,
        max_examples=max_test_examples,
    )
    gop_train = _load_gop_cache(gop_train_path)
    gop_test = _load_gop_cache(gop_test_path)
    ssl_train = _load_ssl_cache(ssl_train_path)
    ssl_test = _load_ssl_cache(ssl_test_path)

    gop_dim = gop_train.gop_dim
    if gop_test.gop_dim != gop_dim:
        raise ValueError(
            f"train/test free-speaking GOP caches disagree on dim: "
            f"{gop_dim} vs {gop_test.gop_dim}"
        )

    # Build a phone vocab across read-aloud + free-speaking so a later
    # curriculum-learning run can reuse a single ``num_phone_tokens`` head.
    phone_vocab = build_phone_vocab(ref_train, ref_test, fs_train, fs_test)

    word_embedding_cache = load_or_build_word_embeddings(
        train_annotations=fs_train,
        test_annotations=fs_test,
        cache_dir=cache_dir,
        model_name=modernbert_model,
        device=embedding_device,
        suffix="_freespeak",
    )
    train_dataset = ReadAloudFeatureDataset(
        annotations=fs_train,
        phone_vocab=phone_vocab,
        word_embeddings=word_embedding_cache["train"],
        gop_features=gop_train.features,
        ssl_utterance=ssl_train.features,
    )
    test_dataset = ReadAloudFeatureDataset(
        annotations=fs_test,
        phone_vocab=phone_vocab,
        word_embeddings=word_embedding_cache["test"],
        gop_features=gop_test.features,
        ssl_utterance=ssl_test.features,
    )
    return train_dataset, test_dataset, len(phone_vocab), gop_dim


def load_annotations(
    split: str, cache_dir: Path, dataset_id: str = HF_DATASET_ID
) -> list[ReadAloudAnnotation]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{split}_read_aloud_annotations.json"
    if cache_file.exists():
        return [
            annotation_from_dict(item)
            for item in json.loads(cache_file.read_text(encoding="utf-8"))
        ]

    raw_any: Any = load_dataset(dataset_id, split=split)
    raw_any = raw_any.cast_column("audio", Audio(decode=False))

    annotations: list[ReadAloudAnnotation] = []
    for example in cast(list[dict[str, Any]], raw_any):
        annotations.append(example_to_annotation(example))
    cache_file.write_text(
        json.dumps(
            [annotation_to_dict(annotation) for annotation in annotations], indent=2
        ),
        encoding="utf-8",
    )
    return annotations


def annotation_to_dict(annotation: ReadAloudAnnotation) -> dict[str, Any]:
    return {
        "text": annotation.text,
        "words": list(annotation.words),
        "phone_tokens": list(annotation.phone_tokens),
        "phone_to_word": list(annotation.phone_to_word),
        "phone_scores": list(annotation.phone_scores),
        "word_scores": [list(scores) for scores in annotation.word_scores],
        "utterance_scores": list(annotation.utterance_scores),
    }


def annotation_from_dict(payload: dict[str, Any]) -> ReadAloudAnnotation:
    word_scores = tuple(
        tuple(float(value) for value in scores) for scores in payload["word_scores"]
    )
    utterance_scores = tuple(float(value) for value in payload["utterance_scores"])
    if len(utterance_scores) != len(UTTERANCE_SCORE_KEYS):
        raise ValueError("Unexpected utterance score count in cached annotations")
    return ReadAloudAnnotation(
        text=str(payload["text"]),
        words=tuple(str(word) for word in payload["words"]),
        phone_tokens=tuple(str(phone) for phone in payload["phone_tokens"]),
        phone_to_word=tuple(int(index) for index in payload["phone_to_word"]),
        phone_scores=tuple(float(score) for score in payload["phone_scores"]),
        word_scores=cast(tuple[tuple[float, float, float], ...], word_scores),
        utterance_scores=cast(
            tuple[float, float, float, float, float], utterance_scores
        ),
    )


def example_to_annotation(example: dict[str, Any]) -> ReadAloudAnnotation:
    words = cast(list[dict[str, Any]], example["words"])
    word_texts = tuple(str(word["text"]) for word in words)

    phone_tokens: list[str] = []
    phone_scores: list[float] = []
    phone_to_word: list[int] = []
    word_scores: list[tuple[float, float, float]] = []
    for word_index, word in enumerate(words):
        phones = cast(list[str], word["phones"])
        phone_accuracy = cast(list[float], word["phones-accuracy"])
        for phone, score in zip(phones, phone_accuracy, strict=True):
            phone_tokens.append(phone)
            phone_scores.append(float(score))
            phone_to_word.append(word_index)
        word_score = tuple(
            normalize_score(float(word[key])) for key in WORD_SCORE_KEYS
        )
        word_scores.append(cast(tuple[float, float, float], word_score))

    utterance_scores = tuple(
        normalize_score(float(example[key])) for key in UTTERANCE_SCORE_KEYS
    )
    return ReadAloudAnnotation(
        text=str(example["text"]),
        words=word_texts,
        phone_tokens=tuple(phone_tokens),
        phone_to_word=tuple(phone_to_word),
        phone_scores=tuple(phone_scores),
        word_scores=tuple(word_scores),
        utterance_scores=cast(
            tuple[float, float, float, float, float], utterance_scores
        ),
    )


def build_phone_vocab(*splits: list[ReadAloudAnnotation]) -> dict[str, int]:
    phones = sorted(
        {
            phone
            for split in splits
            for annotation in split
            for phone in annotation.phone_tokens
        }
    )
    return {phone: index for index, phone in enumerate(phones, start=1)}


_WORD_EMBEDDING_CACHE_VERSION = 1


def load_or_build_word_embeddings(
    train_annotations: list[ReadAloudAnnotation],
    test_annotations: list[ReadAloudAnnotation],
    cache_dir: Path,
    model_name: str,
    device: torch.device,
    suffix: str = "",
) -> dict[str, list[Tensor]]:
    sanitized_name = model_name.replace("/", "__")
    train_cache = cache_dir / f"train{suffix}_{sanitized_name}_word_embeddings.pt"
    test_cache = cache_dir / f"test{suffix}_{sanitized_name}_word_embeddings.pt"
    sidecar = cache_dir / f"{sanitized_name}{suffix}_word_embeddings.json"
    phone_source = "freespeak" if suffix == "_freespeak" else "reference"
    if train_cache.exists() and test_cache.exists() and sidecar.exists():
        try:
            sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            sidecar_payload = {}
        if (
            sidecar_payload.get("version") == _WORD_EMBEDDING_CACHE_VERSION
            and sidecar_payload.get("model_id") == model_name
            and sidecar_payload.get("phone_source") == phone_source
            and sidecar_payload.get("num_train") == len(train_annotations)
            and sidecar_payload.get("num_test") == len(test_annotations)
        ):
            train_embeddings = cast(
                list[Tensor],
                torch.load(train_cache, map_location="cpu", weights_only=False),
            )
            test_embeddings = cast(
                list[Tensor],
                torch.load(test_cache, map_location="cpu", weights_only=False),
            )
            return {"train": train_embeddings, "test": test_embeddings}

    transformers_module = cast(Any, import_module("transformers"))
    tokenizer = cast(
        object, transformers_module.AutoTokenizer.from_pretrained(model_name)
    )
    model = transformers_module.AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    train_embeddings = encode_word_embeddings(train_annotations, tokenizer, model, device)
    test_embeddings = encode_word_embeddings(test_annotations, tokenizer, model, device)
    torch.save(train_embeddings, train_cache)
    torch.save(test_embeddings, test_cache)
    sidecar.write_text(
        json.dumps(
            {
                "version": _WORD_EMBEDDING_CACHE_VERSION,
                "model_id": model_name,
                "phone_source": phone_source,
                "num_train": len(train_annotations),
                "num_test": len(test_annotations),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"train": train_embeddings, "test": test_embeddings}


def encode_word_embeddings(
    annotations: list[ReadAloudAnnotation],
    tokenizer: object,
    model: object,
    device: torch.device,
    batch_size: int = 32,
) -> list[Tensor]:
    encoded_examples: list[Tensor] = []
    typed_tokenizer = cast(Any, tokenizer)
    typed_model = cast(Any, model)
    for start in range(0, len(annotations), batch_size):
        batch_words = [
            list(annotation.words)
            for annotation in annotations[start : start + batch_size]
        ]
        tokenized: Any = typed_tokenizer(
            batch_words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = tokenized.to(device)
        with torch.no_grad():
            outputs = typed_model(**tokenized)
        hidden_states = cast(Tensor, outputs.last_hidden_state).detach().cpu()
        hidden_size = hidden_states.size(-1)

        for batch_index, words in enumerate(batch_words):
            token_word_ids = tokenized.word_ids(batch_index=batch_index)
            word_vectors: list[list[Tensor]] = [[] for _ in words]
            for token_index, word_id in enumerate(token_word_ids):
                if word_id is None:
                    continue
                word_vectors[word_id].append(hidden_states[batch_index, token_index])
            pooled = [
                torch.stack(tokens, dim=0).mean(dim=0)
                if tokens
                else torch.zeros(hidden_size, dtype=torch.float32)
                for tokens in word_vectors
            ]
            encoded_examples.append(torch.stack(pooled, dim=0))
    return encoded_examples


@dataclass(frozen=True)
class _GopCachePayload:
    features: list[Tensor]
    gop_dim: int


@dataclass(frozen=True)
class _SslCachePayload:
    features: Tensor


def _load_gop_cache(path: Path) -> _GopCachePayload:
    payload = cast(
        dict[str, Any], torch.load(path, map_location="cpu", weights_only=False)
    )
    feature_list = cast(list[Tensor], payload["features"])
    gop_dim = int(payload["gop_dim"])
    return _GopCachePayload(features=feature_list, gop_dim=gop_dim)


def _load_ssl_cache(path: Path) -> _SslCachePayload:
    payload = cast(
        dict[str, Any], torch.load(path, map_location="cpu", weights_only=False)
    )
    features = cast(Tensor, payload["features"]).to(dtype=torch.float32)
    return _SslCachePayload(features=features)


class ReadAloudFeatureDataset(Dataset[ReadAloudSample]):
    def __init__(
        self,
        annotations: list[ReadAloudAnnotation],
        phone_vocab: dict[str, int],
        word_embeddings: list[Tensor],
        gop_features: list[Tensor],
        ssl_utterance: Tensor,
    ) -> None:
        if len(annotations) != len(word_embeddings):
            raise ValueError("annotation count must match word embedding count")
        if len(annotations) != len(gop_features):
            raise ValueError("annotation count must match GOP feature count")
        if ssl_utterance.shape[0] != len(annotations):
            raise ValueError("annotation count must match SSL utterance count")
        self.annotations = annotations
        self.phone_vocab = phone_vocab
        self.word_embeddings = word_embeddings
        self.gop_features = gop_features
        self.ssl_utterance = ssl_utterance

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> ReadAloudSample:
        annotation = self.annotations[index]
        gop = self.gop_features[index].to(dtype=torch.float32)
        if gop.shape[0] != len(annotation.phone_tokens):
            # Trim to the shorter length so the pipeline stays consistent when
            # cache was built with a different phone count (e.g. stale cache).
            n_phones = min(gop.shape[0], len(annotation.phone_tokens))
            gop = gop[:n_phones]
            phone_tokens = annotation.phone_tokens[:n_phones]
            phone_scores = annotation.phone_scores[:n_phones]
            phone_to_word = annotation.phone_to_word[:n_phones]
        else:
            phone_tokens = annotation.phone_tokens
            phone_scores = annotation.phone_scores
            phone_to_word = annotation.phone_to_word

        phone_ids = torch.tensor(
            [self.phone_vocab[phone] for phone in phone_tokens],
            dtype=torch.long,
        )
        word_embeddings = self.word_embeddings[index].to(dtype=torch.float32)
        ssl_vec = self.ssl_utterance[index].to(dtype=torch.float32)
        return ReadAloudSample(
            gop_features=gop,
            ssl_utterance=ssl_vec,
            phone_ids=phone_ids,
            phone_to_word=torch.tensor(phone_to_word, dtype=torch.long),
            word_embeddings=word_embeddings,
            phone_targets=torch.tensor(phone_scores, dtype=torch.float32),
            word_targets=torch.tensor(annotation.word_scores, dtype=torch.float32),
            utterance_targets=torch.tensor(
                annotation.utterance_scores, dtype=torch.float32
            ),
        )


def collate_read_aloud_batch(samples: list[ReadAloudSample]) -> ReadAloudBatch:
    batch_size = len(samples)
    max_phones = max(sample.phone_ids.size(0) for sample in samples)
    max_words = max(sample.word_embeddings.size(0) for sample in samples)
    word_embedding_dim = samples[0].word_embeddings.size(-1)
    gop_dim = samples[0].gop_features.size(-1)
    ssl_dim = samples[0].ssl_utterance.size(-1)

    gop_features = torch.zeros(batch_size, max_phones, gop_dim, dtype=torch.float32)
    ssl_utterance = torch.zeros(batch_size, ssl_dim, dtype=torch.float32)
    phone_ids = torch.zeros(batch_size, max_phones, dtype=torch.long)
    phone_to_word = torch.full(
        (batch_size, max_phones), fill_value=-1, dtype=torch.long
    )
    phone_mask = torch.zeros(batch_size, max_phones, dtype=torch.bool)
    word_embeddings = torch.zeros(
        batch_size, max_words, word_embedding_dim, dtype=torch.float32
    )
    word_mask = torch.zeros(batch_size, max_words, dtype=torch.bool)
    phone_targets = torch.full(
        (batch_size, max_phones), fill_value=-1.0, dtype=torch.float32
    )
    word_targets = torch.zeros(
        batch_size, max_words, len(WORD_SCORE_KEYS), dtype=torch.float32
    )
    utterance_targets = torch.stack(
        [sample.utterance_targets for sample in samples], dim=0
    )

    for batch_index, sample in enumerate(samples):
        phone_len = sample.phone_ids.size(0)
        word_len = sample.word_embeddings.size(0)
        gop_features[batch_index, :phone_len] = sample.gop_features
        ssl_utterance[batch_index] = sample.ssl_utterance
        phone_ids[batch_index, :phone_len] = sample.phone_ids
        phone_to_word[batch_index, :phone_len] = sample.phone_to_word
        phone_mask[batch_index, :phone_len] = True
        word_embeddings[batch_index, :word_len] = sample.word_embeddings
        word_mask[batch_index, :word_len] = True
        phone_targets[batch_index, :phone_len] = sample.phone_targets
        word_targets[batch_index, :word_len] = sample.word_targets

    return ReadAloudBatch(
        gop_features=gop_features,
        ssl_utterance=ssl_utterance,
        phone_ids=phone_ids,
        phone_to_word=phone_to_word,
        word_embeddings=word_embeddings,
        phone_mask=phone_mask,
        word_mask=word_mask,
        phone_targets=phone_targets,
        word_targets=word_targets,
        utterance_targets=utterance_targets,
    )


def normalize_score(score: float) -> float:
    return score / 5.0
