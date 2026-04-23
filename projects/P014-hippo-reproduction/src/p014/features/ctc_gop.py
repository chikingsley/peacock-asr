"""CTC-based Goodness-of-Pronunciation feature extraction.

Reproduces Cao et al. 2024 — "A Framework for Phoneme-Level Pronunciation
Assessment Using CTC" (INTERSPEECH 2024). The reference implementation is
frank613/CTC-based-GOP at ``is24/generate-GOP-features/gop-af-feats.py``. For
each canonical phone in a target transcription, the extractor computes:

1. ``LPP`` — log posterior probability (or here, the CTC ``loss`` returned by
   ``Wav2Vec2ForCTC`` with ``labels=`` set to the canonical phone sequence).
   It is shared across all phones of the utterance.
2. ``LPR_{p->q}`` — log-probability ratio of the canonical path versus a
   substitution path where phone ``p`` is replaced by vocabulary token ``q``
   (for every ``q`` in the CTC vocab, including ``blank`` → deletion). This
   uses the forward algorithm of Graves' CTC to marginalise over all
   alignments, so the GOP is segmentation-free.

The per-phone feature vector is therefore ``[LPP, LPR_1, LPR_2, ...,
LPR_V]`` with dimension ``1 + vocab_size``.

Paper alignment
---------------
The HiPPO paper (Yan et al. 2025, Section 2.2) states ``E^gop`` has shape
``41 x N``. The cited CTC-based-GOP repository now publishes the matching
``is24`` checkpoint: a LibriSpeech-fine-tuned wav2vec2 CTC model with
``vocab_size = 40`` (39 ARPAbet phones plus blank), so the emitted GOP feature
width is exactly ``1 + 40 = 41``. This module downloads and assembles that
paper-aligned checkpoint into the local cache by default.

Users can still override ``model_id`` to point at another local directory or a
Hugging Face model, but the default path is the released paper checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from urllib.request import Request, urlopen

import torch
import torchaudio
from datasets import Audio, load_dataset
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor
from tqdm import tqdm

CTC_GOP_MODEL_ID = "paper_ctc_gop_is24_checkpoint_8000"
HF_DATASET_ID = "mispeech/speechocean762"
CACHE_FORMAT_VERSION = 1
TARGET_SAMPLE_RATE = 16_000
_PAPER_CTC_GOP_DIRNAME = "frank613_ctc_based_gop_is24_checkpoint_8000"
_PAPER_CTC_GOP_FILES: dict[str, tuple[str, int | None]] = {
    "config.json": (
        "https://raw.githubusercontent.com/frank613/CTC-based-GOP/main/is24/models/checkpoint-8000/config.json",
        None,
    ),
    "preprocessor_config.json": (
        "https://raw.githubusercontent.com/frank613/CTC-based-GOP/main/is24/models/processor_config_gop/preprocessor_config.json",
        None,
    ),
    "special_tokens_map.json": (
        "https://raw.githubusercontent.com/frank613/CTC-based-GOP/main/is24/models/processor_config_gop/special_tokens_map.json",
        None,
    ),
    "tokenizer_config.json": (
        "https://raw.githubusercontent.com/frank613/CTC-based-GOP/main/is24/models/processor_config_gop/tokenizer_config.json",
        None,
    ),
    "vocab.json": (
        "https://raw.githubusercontent.com/frank613/CTC-based-GOP/main/is24/models/processor_config_gop/vocab.json",
        None,
    ),
    "pytorch_model.bin": (
        "https://media.githubusercontent.com/media/frank613/CTC-based-GOP/main/is24/models/checkpoint-8000/pytorch_model.bin",
        1_262_066_282,
    ),
}

# Canonical phones in SpeechOcean762 are upper-case ARPAbet with stress
# digits (e.g. ``AO1``). The CTC model emits lower-case ARPAbet without
# stress (e.g. ``ao``). We strip trailing stress digits and lower-case.
_STRESS_PATTERN = re.compile(r"\d+$")


class FeatureExtractionSettings(BaseSettings):
    """Runtime tunables for CTC-GOP extraction."""

    model_config = SettingsConfigDict(env_prefix="P014_GOP_", extra="ignore")

    model_id: str = Field(default=CTC_GOP_MODEL_ID)
    dataset_id: str = Field(default=HF_DATASET_ID)
    blank_token_id: int = Field(default=0, ge=0)
    progress: bool = Field(default=True)


@dataclass(frozen=True)
class _GopCacheSidecar:
    version: int
    model_id: str
    split: str
    dataset_id: str
    num_examples: int
    gop_dim: int
    content_hash: str
    phone_source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "model_id": self.model_id,
            "split": self.split,
            "dataset_id": self.dataset_id,
            "num_examples": self.num_examples,
            "gop_dim": self.gop_dim,
            "content_hash": self.content_hash,
            "phone_source": self.phone_source,
        }


@dataclass(frozen=True)
class _GopPartsManifest:
    version: int
    model_id: str
    split: str
    dataset_id: str
    num_examples: int
    gop_dim: int
    phone_source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "model_id": self.model_id,
            "split": self.split,
            "dataset_id": self.dataset_id,
            "num_examples": self.num_examples,
            "gop_dim": self.gop_dim,
            "phone_source": self.phone_source,
        }


def _normalise_phone(phone: str) -> str:
    return _STRESS_PATTERN.sub("", phone).lower()


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "peacock-asr/p014"})
    with urlopen(request, timeout=60) as response, tmp_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    tmp_path.replace(destination)


def _ensure_paper_ctc_gop_checkpoint(cache_dir: Path) -> Path:
    model_dir = cache_dir / "models" / _PAPER_CTC_GOP_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_size) in _PAPER_CTC_GOP_FILES.items():
        target = model_dir / filename
        if target.exists():
            if expected_size is None or target.stat().st_size == expected_size:
                continue
        _download_file(url, target)
    return model_dir


def _resolve_model_source(cache_dir: Path, model_id: str) -> str:
    if model_id == CTC_GOP_MODEL_ID:
        return str(_ensure_paper_ctc_gop_checkpoint(cache_dir))
    return model_id


def _effective_vocab(vocab: dict[str, int], model_vocab_size: int) -> dict[str, int]:
    """Drop tokenizer-only entries that sit beyond the model CTC head width."""

    return {
        token: index
        for token, index in sorted(vocab.items(), key=lambda item: item[1])
        if 0 <= int(index) < model_vocab_size
    }


def _canonical_phone_ids(
    phones: list[str], vocab: dict[str, int]
) -> list[int]:
    """Map SpeechOcean phones to CTC-model phone ids.

    Unknown tokens fall back to ``[UNK]`` if present, else the first non-blank
    token. We return raw ints so callers can build tensors without casts.
    """

    fallback_id = vocab.get("[UNK]", vocab.get("<unk>", 1))
    ids: list[int] = []
    for phone in phones:
        key = _normalise_phone(phone)
        if key in vocab:
            ids.append(vocab[key])
            continue
        # Some eSpeak-style models use diphthong tokens; keep the first char
        # as a last resort so the label stays a valid vocab id.
        first_char = key[:1]
        ids.append(vocab.get(first_char, fallback_id))
    return ids


def _validate_shard_args(
    *, num_shards: int | None, shard_index: int | None
) -> None:
    if num_shards is None and shard_index is None:
        return
    if num_shards is None or shard_index is None:
        raise ValueError("num_shards and shard_index must be provided together")
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )


def _shard_suffix(
    suffix: str,
    *,
    num_shards: int | None,
    shard_index: int | None,
) -> str:
    if num_shards is None:
        return suffix
    if shard_index is None:
        raise ValueError("shard_index is required when num_shards is set")
    return f"{suffix}__shard_{shard_index:03d}_of_{num_shards:03d}"


def _selected_indices(
    total: int, *, num_shards: int | None, shard_index: int | None
) -> list[int]:
    if num_shards is None:
        return list(range(total))
    if shard_index is None:
        raise ValueError("shard_index is required when num_shards is set")
    # Matches the default strided behaviour of ``datasets.Dataset.shard``.
    return list(range(shard_index, total, num_shards))


def _write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_atomic_torch(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _parts_manifest_is_fresh(
    manifest_path: Path,
    *,
    model_id: str,
    split: str,
    dataset_id: str,
    expected_count: int,
    gop_dim: int,
    phone_source: str,
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        payload.get("version") == CACHE_FORMAT_VERSION
        and payload.get("model_id") == model_id
        and payload.get("split") == split
        and payload.get("dataset_id") == dataset_id
        and payload.get("num_examples") == expected_count
        and payload.get("gop_dim") == gop_dim
        and payload.get("phone_source") == phone_source
    )


def _prepare_parts_cache(
    *,
    feature_dir: Path,
    split: str,
    suffix: str,
    model_id: str,
    dataset_id: str,
    expected_count: int,
    gop_dim: int,
    phone_source: str,
) -> tuple[Path, Path]:
    parts_dir = feature_dir / f"{split}{suffix}.parts"
    manifest_path = feature_dir / f"{split}{suffix}.parts.json"
    if not _parts_manifest_is_fresh(
        manifest_path,
        model_id=model_id,
        split=split,
        dataset_id=dataset_id,
        expected_count=expected_count,
        gop_dim=gop_dim,
        phone_source=phone_source,
    ):
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        if manifest_path.exists():
            manifest_path.unlink()
    parts_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest = _GopPartsManifest(
            version=CACHE_FORMAT_VERSION,
            model_id=model_id,
            split=split,
            dataset_id=dataset_id,
            num_examples=expected_count,
            gop_dim=gop_dim,
            phone_source=phone_source,
        )
        _write_atomic_json(manifest_path, manifest.to_dict())
    return parts_dir, manifest_path


def _part_path(parts_dir: Path, index: int) -> Path:
    return parts_dir / f"{index:06d}.pt"


def _assemble_parts_cache(
    *,
    parts_dir: Path,
    ordered_indices: list[int],
    cache_path: Path,
    sidecar_path: Path,
    model_id: str,
    split: str,
    dataset_id: str,
    gop_dim: int,
    phone_source: str,
) -> None:
    utterance_ids: list[str] = []
    feature_tensors: list[Tensor] = []
    for index in ordered_indices:
        payload = cast(
            dict[str, object],
            torch.load(_part_path(parts_dir, index), map_location="cpu", weights_only=False),
        )
        payload_index = int(payload.get("dataset_index", index))
        if payload_index != index:
            raise RuntimeError(
                f"part file index mismatch for {parts_dir}: expected {index}, got {payload_index}"
            )
        utterance_ids.append(cast(str, payload["utterance_id"]))
        feature_tensors.append(cast(Tensor, payload["features"]))

    hasher = hashlib.sha256()
    hasher.update(model_id.encode("utf-8"))
    hasher.update(str(gop_dim).encode("utf-8"))
    hasher.update(str(len(feature_tensors)).encode("utf-8"))
    content_hash = hasher.hexdigest()

    _write_atomic_torch(
        cache_path,
        {
            "dataset_indices": ordered_indices,
            "utterance_ids": utterance_ids,
            "features": feature_tensors,
            "gop_dim": gop_dim,
            "model_id": model_id,
        },
    )
    sidecar = _GopCacheSidecar(
        version=CACHE_FORMAT_VERSION,
        model_id=model_id,
        split=split,
        dataset_id=dataset_id,
        num_examples=len(feature_tensors),
        gop_dim=gop_dim,
        content_hash=content_hash,
        phone_source=phone_source,
    )
    _write_atomic_json(sidecar_path, sidecar.to_dict())


def _ctc_log_forward(log_params: Tensor, seq: Tensor, blank: int) -> Tensor:
    """Compute CTC forward log-likelihood for a single utterance.

    ``log_params`` has shape ``(V, T)`` (log probabilities per time step).
    ``seq`` is a 1-D int tensor of label ids. Follows frank613's implementation
    but runs in log-space to avoid numerical underflow.
    """

    seq_len = int(seq.shape[0])
    time_steps = int(log_params.shape[1])
    extended_len = 2 * seq_len + 1  # insert blanks between labels

    neg_inf = torch.full((extended_len, time_steps), float("-inf"), dtype=log_params.dtype)
    alphas = neg_inf.clone()
    alphas[0, 0] = log_params[blank, 0]
    if seq_len > 0:
        alphas[1, 0] = log_params[seq[0], 0]

    for t in range(1, time_steps):
        start = max(0, extended_len - 2 * (time_steps - t))
        end = min(2 * t + 2, extended_len)
        for s in range(start, end):
            label_idx = (s - 1) // 2
            if s % 2 == 0:
                # blank position
                prev = alphas[s, t - 1]
                if s > 0:
                    prev = torch.logaddexp(prev, alphas[s - 1, t - 1])
                alphas[s, t] = prev + log_params[blank, t]
            elif s == 1 or seq[label_idx] == seq[label_idx - 1]:
                prev = torch.logaddexp(alphas[s, t - 1], alphas[s - 1, t - 1])
                alphas[s, t] = prev + log_params[seq[label_idx], t]
            else:
                prev = torch.logaddexp(alphas[s, t - 1], alphas[s - 1, t - 1])
                prev = torch.logaddexp(prev, alphas[s - 2, t - 1])
                alphas[s, t] = prev + log_params[seq[label_idx], t]

    if seq_len == 0:
        return alphas[extended_len - 1, time_steps - 1]
    return torch.logaddexp(
        alphas[extended_len - 1, time_steps - 1],
        alphas[extended_len - 2, time_steps - 1],
    )


def _load_audio(audio_payload: Any) -> Tensor:
    """Decode a HuggingFace audio payload into a mono 16 kHz float tensor."""

    if hasattr(audio_payload, "get_all_samples"):
        samples = audio_payload.get_all_samples()
        waveform = cast(Tensor, samples.data).to(dtype=torch.float32)
        sampling_rate = int(samples.sample_rate)
    else:
        payload = cast(dict[str, Any], audio_payload)
        path = payload.get("path")
        if path is None:
            # Fall back to the pre-decoded array when the path is not available.
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


def _compute_gop_for_utterance(
    waveform: Tensor,
    canonical_ids: list[int],
    processor: object,
    model: object,
    device: torch.device,
    blank_id: int,
) -> Tensor:
    """Return a ``(num_canonical_phones, 1 + vocab_size)`` tensor of GOP features."""

    typed_processor = cast(Any, processor)
    typed_model = cast(Any, model)
    processed: Any = typed_processor(
        waveform.cpu().numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    )
    input_values = cast(Tensor, processed.input_values).to(device)
    labels = torch.tensor(canonical_ids, dtype=torch.long, device=device)
    if labels.numel() == 0:
        return torch.empty(0, 0, dtype=torch.float32)

    outputs = typed_model(input_values=input_values, labels=labels)
    # The HF CTCModelOutput returns ``loss`` as a negative-log-likelihood sum
    # (``ctc_loss_reduction='sum'`` for this checkpoint), so ``-loss`` is the
    # log-likelihood of the canonical sequence. We keep sign conventions
    # matching frank613's reference, where ``log_like_total`` is the *negative*
    # of the HF loss (i.e. the canonical log-likelihood).
    log_like_total = cast(Tensor, -outputs.loss.detach()).to(torch.float64)
    logits = cast(Tensor, outputs.logits.detach()).squeeze(0).to(torch.float64)
    log_posteriors = torch.log_softmax(logits, dim=-1).transpose(0, 1)  # (V, T)
    vocab_size = int(log_posteriors.shape[0])

    features = torch.empty(len(canonical_ids), 1 + vocab_size, dtype=torch.float32)
    labels_cpu = labels.detach().to("cpu")
    log_post_cpu = log_posteriors.to("cpu")
    log_like_cpu = log_like_total.to("cpu")

    for phone_index in range(len(canonical_ids)):
        row: list[float] = [float(log_like_cpu.item())]
        for sub_id in range(vocab_size):
            if sub_id == blank_id:
                # Deletion path — drop this phone from the canonical sequence.
                new_labels = torch.cat(
                    [labels_cpu[:phone_index], labels_cpu[phone_index + 1 :]]
                )
            else:
                new_labels = labels_cpu.clone()
                new_labels[phone_index] = sub_id
            sub_log_like = _ctc_log_forward(log_post_cpu, new_labels, blank=blank_id)
            # ``LPR`` in frank613's code is ``-log_like_total - log(ctc)``, i.e.
            # ``log P(sub) - log P(canonical)``. We keep that convention.
            lpr = float((-log_like_cpu - sub_log_like).item())
            row.append(lpr)
        features[phone_index] = torch.tensor(row, dtype=torch.float32)
    return features


def _expected_example_count(
    dataset_id: str, split: str, *, max_examples: int | None
) -> int:
    # We only need a light-weight length check; ``load_dataset`` already
    # caches the metadata so this is cheap on repeat calls.
    ds: Any = load_dataset(dataset_id, split=split)
    total = len(ds)
    if max_examples is None:
        return int(total)
    return min(int(total), int(max_examples))


def _cache_is_fresh(
    cache_path: Path,
    sidecar_path: Path,
    *,
    model_id: str,
    split: str,
    dataset_id: str,
    expected_count: int,
    phone_source: str,
) -> bool:
    if not cache_path.exists() or not sidecar_path.exists():
        return False
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        sidecar.get("version") == CACHE_FORMAT_VERSION
        and sidecar.get("model_id") == model_id
        and sidecar.get("split") == split
        and sidecar.get("dataset_id") == dataset_id
        and sidecar.get("num_examples") == expected_count
        and sidecar.get("phone_source") == phone_source
    )


def phones_to_canonical_ids(
    phones: list[str], vocab: dict[str, int]
) -> list[int]:
    """Public wrapper around the phone -> CTC-vocab id translation.

    Exposed as a pure function so callers (e.g. the free-speaking loader)
    can drive GOP extraction from an externally-supplied phone sequence
    without re-implementing the normalisation rules.
    """

    return _canonical_phone_ids(phones, vocab)


def extract_gop_for_split(
    split: str,
    cache_dir: Path,
    dataset_id: str = HF_DATASET_ID,
    device: torch.device | None = None,
    settings: FeatureExtractionSettings | None = None,
    max_examples: int | None = None,
    phones_per_utterance: list[list[str]] | None = None,
    suffix: str = "",
    num_shards: int | None = None,
    shard_index: int | None = None,
) -> Path:
    """Extract CTC-GOP features for a dataset split and cache them to disk.

    Returns the path to the cached ``.pt`` file. The file contains a dict of
    ``{"utterance_ids": list[str], "features": list[Tensor]}``. A JSON sidecar
    alongside the ``.pt`` file records version/model hash so callers can
    detect stale caches.

    Args:
        phones_per_utterance: Optional per-utterance canonical phone list. When
            supplied, overrides the phones derived from the dataset's reference
            annotation (used by the free-speaking pipeline, which feeds G2P
            output of the ASR hypothesis). Length must equal the effective
            example count (clipped by ``max_examples`` when set).
        suffix: Cache filename suffix (e.g. ``"_freespeak"``) so distinct phone
            sources don't collide in the same cache directory.
        num_shards / shard_index: Optional shard assignment. When set, only the
            selected shard is computed and cached to a shard-specific ``.pt``.
    """

    _validate_shard_args(num_shards=num_shards, shard_index=shard_index)
    phone_source = "reference" if phones_per_utterance is None else "override"
    resolved_settings = settings or FeatureExtractionSettings(model_id=CTC_GOP_MODEL_ID)
    feature_dir = cache_dir / "features" / "gop"
    feature_dir.mkdir(parents=True, exist_ok=True)
    cache_suffix = _shard_suffix(
        suffix, num_shards=num_shards, shard_index=shard_index
    )
    cache_path = feature_dir / f"{split}{cache_suffix}.pt"
    sidecar_path = feature_dir / f"{split}{cache_suffix}.json"

    total_examples = _expected_example_count(
        dataset_id, split, max_examples=max_examples
    )
    if phones_per_utterance is not None and len(phones_per_utterance) != total_examples:
        raise ValueError(
            "phones_per_utterance has length "
            f"{len(phones_per_utterance)} but expected {total_examples}"
        )
    selected_indices = _selected_indices(
        total_examples, num_shards=num_shards, shard_index=shard_index
    )
    if _cache_is_fresh(
        cache_path,
        sidecar_path,
        model_id=resolved_settings.model_id,
        split=split,
        dataset_id=dataset_id,
        expected_count=len(selected_indices),
        phone_source=phone_source,
    ):
        return cache_path

    transformers_module = cast(Any, import_module("transformers"))

    resolved_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_source = _resolve_model_source(cache_dir, resolved_settings.model_id)
    tokenizer_any: Any = transformers_module.Wav2Vec2CTCTokenizer.from_pretrained(
        model_source
    )
    processor_any: Any = transformers_module.Wav2Vec2Processor.from_pretrained(
        model_source
    )
    model_any: Any = transformers_module.Wav2Vec2ForCTC.from_pretrained(model_source)
    model_any = model_any.to(resolved_device)
    model_any.eval()

    model_vocab_size = int(model_any.config.vocab_size)
    vocab = _effective_vocab(
        cast(dict[str, int], tokenizer_any.get_vocab()),
        model_vocab_size=model_vocab_size,
    )
    gop_dim = 1 + model_vocab_size

    raw_any: Any = load_dataset(dataset_id, split=split)
    raw_any = raw_any.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    total = len(raw_any)
    if max_examples is not None:
        total = min(total, int(max_examples))
    if total != total_examples:
        raise RuntimeError(
            f"dataset length mismatch while preparing GOP cache: {total} != {total_examples}"
        )

    parts_dir, manifest_path = _prepare_parts_cache(
        feature_dir=feature_dir,
        split=split,
        suffix=cache_suffix,
        model_id=resolved_settings.model_id,
        dataset_id=dataset_id,
        expected_count=len(selected_indices),
        gop_dim=gop_dim,
        phone_source=phone_source,
    )
    pending_indices = [
        index for index in selected_indices if not _part_path(parts_dir, index).exists()
    ]
    progress_iter = tqdm(
        pending_indices,
        desc=f"ctc-gop/{split}{cache_suffix}",
        disable=not resolved_settings.progress,
        total=len(selected_indices),
        initial=len(selected_indices) - len(pending_indices),
    )
    for index in progress_iter:
        example = cast(dict[str, Any], raw_any[int(index)])
        utterance_id = str(example.get("id") or example.get("speaker") or index)
        phones: list[str]
        if phones_per_utterance is not None:
            phones = list(phones_per_utterance[index])
        else:
            phones = []
            for word in cast(list[dict[str, Any]], example.get("words", [])):
                phones.extend(str(phone) for phone in cast(list[str], word["phones"]))
        if not phones:
            _write_atomic_torch(
                _part_path(parts_dir, index),
                {
                    "dataset_index": index,
                    "utterance_id": utterance_id,
                    "features": torch.empty(0, gop_dim, dtype=torch.float32),
                },
            )
            continue
        canonical_ids = _canonical_phone_ids(phones, vocab)

        with torch.no_grad():
            waveform = _load_audio(cast(dict[str, Any], example["audio"]))
            features = _compute_gop_for_utterance(
                waveform,
                canonical_ids,
                processor_any,
                model_any,
                resolved_device,
                blank_id=resolved_settings.blank_token_id,
            )
        _write_atomic_torch(
            _part_path(parts_dir, index),
            {
                "dataset_index": index,
                "utterance_id": utterance_id,
                "features": features.detach().cpu(),
            },
        )

    _assemble_parts_cache(
        parts_dir=parts_dir,
        ordered_indices=selected_indices,
        cache_path=cache_path,
        sidecar_path=sidecar_path,
        model_id=resolved_settings.model_id,
        split=split,
        dataset_id=dataset_id,
        gop_dim=gop_dim,
        phone_source=phone_source,
    )
    shutil.rmtree(parts_dir)
    manifest_path.unlink(missing_ok=True)
    return cache_path


def merge_gop_shards(
    split: str,
    cache_dir: Path,
    *,
    num_shards: int,
    dataset_id: str = HF_DATASET_ID,
    max_examples: int | None = None,
    suffix: str = "",
) -> Path:
    """Merge shard-specific GOP caches into the standard full-split cache."""

    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")

    feature_dir = cache_dir / "features" / "gop"
    feature_dir.mkdir(parents=True, exist_ok=True)
    total_examples = _expected_example_count(
        dataset_id, split, max_examples=max_examples
    )
    cache_path = feature_dir / f"{split}{suffix}.pt"
    sidecar_path = feature_dir / f"{split}{suffix}.json"

    ordered_utterance_ids: list[str | None] = [None] * total_examples
    ordered_features: list[Tensor | None] = [None] * total_examples
    merged_model_id: str | None = None
    merged_gop_dim: int | None = None
    merged_phone_source: str | None = None

    for current_shard in range(num_shards):
        shard_suffix = _shard_suffix(
            suffix, num_shards=num_shards, shard_index=current_shard
        )
        shard_cache_path = feature_dir / f"{split}{shard_suffix}.pt"
        shard_sidecar_path = feature_dir / f"{split}{shard_suffix}.json"
        shard_indices = _selected_indices(
            total_examples, num_shards=num_shards, shard_index=current_shard
        )

        if not shard_cache_path.exists() or not shard_sidecar_path.exists():
            raise FileNotFoundError(
                f"missing GOP shard cache for {split} shard {current_shard}/{num_shards}"
            )
        sidecar = json.loads(shard_sidecar_path.read_text(encoding="utf-8"))
        if not _cache_is_fresh(
            shard_cache_path,
            shard_sidecar_path,
            model_id=cast(str, sidecar["model_id"]),
            split=split,
            dataset_id=dataset_id,
            expected_count=len(shard_indices),
            phone_source=cast(str, sidecar["phone_source"]),
        ):
            raise RuntimeError(
                f"stale GOP shard sidecar for {split} shard {current_shard}/{num_shards}"
            )

        payload = cast(
            dict[str, object],
            torch.load(shard_cache_path, map_location="cpu", weights_only=False),
        )
        dataset_indices = cast(list[int], payload.get("dataset_indices", shard_indices))
        utterance_ids = cast(list[str], payload["utterance_ids"])
        features = cast(list[Tensor], payload["features"])
        gop_dim = int(payload["gop_dim"])
        model_id = cast(str, payload["model_id"])
        phone_source = cast(str, sidecar["phone_source"])

        if dataset_indices != shard_indices:
            raise RuntimeError(
                f"GOP shard indices do not match expected shard assignment for {split} shard {current_shard}/{num_shards}"
            )
        if len(utterance_ids) != len(shard_indices) or len(features) != len(shard_indices):
            raise RuntimeError(
                f"GOP shard payload length mismatch for {split} shard {current_shard}/{num_shards}"
            )
        if merged_model_id is None:
            merged_model_id = model_id
            merged_gop_dim = gop_dim
            merged_phone_source = phone_source
        elif (
            model_id != merged_model_id
            or gop_dim != merged_gop_dim
            or phone_source != merged_phone_source
        ):
            raise RuntimeError(
                f"GOP shard metadata mismatch while merging {split} shards"
            )

        for index, utterance_id, feature in zip(
            dataset_indices, utterance_ids, features, strict=True
        ):
            if ordered_utterance_ids[index] is not None:
                raise RuntimeError(
                    f"duplicate GOP shard coverage for dataset index {index}"
                )
            ordered_utterance_ids[index] = utterance_id
            ordered_features[index] = feature

    if merged_model_id is None or merged_gop_dim is None or merged_phone_source is None:
        raise RuntimeError(f"no GOP shards found to merge for split {split}")
    if any(item is None for item in ordered_utterance_ids) or any(
        item is None for item in ordered_features
    ):
        raise RuntimeError(f"incomplete GOP shard coverage for split {split}")

    hasher = hashlib.sha256()
    hasher.update(merged_model_id.encode("utf-8"))
    hasher.update(str(merged_gop_dim).encode("utf-8"))
    hasher.update(str(total_examples).encode("utf-8"))
    content_hash = hasher.hexdigest()

    _write_atomic_torch(
        cache_path,
        {
            "dataset_indices": list(range(total_examples)),
            "utterance_ids": cast(list[str], ordered_utterance_ids),
            "features": cast(list[Tensor], ordered_features),
            "gop_dim": merged_gop_dim,
            "model_id": merged_model_id,
        },
    )
    sidecar = _GopCacheSidecar(
        version=CACHE_FORMAT_VERSION,
        model_id=merged_model_id,
        split=split,
        dataset_id=dataset_id,
        num_examples=total_examples,
        gop_dim=merged_gop_dim,
        content_hash=content_hash,
        phone_source=merged_phone_source,
    )
    _write_atomic_json(sidecar_path, sidecar.to_dict())
    return cache_path
