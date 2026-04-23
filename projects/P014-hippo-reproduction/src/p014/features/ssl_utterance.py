"""Utterance-level SSL feature extraction for the HiPPO reproduction.

The paper (Yan et al. 2025, eq. (2)) fuses utterance-level embeddings from
three pretrained SSL models:

* ``facebook/wav2vec2-large-960h`` (1024-dim)
* ``facebook/hubert-large-ls960-ft`` (1024-dim)
* ``microsoft/wavlm-large`` (1024-dim)

For each utterance, we forward the raw 16 kHz waveform through each model
(frozen, ``eval`` mode), mean-pool the last hidden state over time to get a
single 1024-dim vector, then concatenate the three vectors into the 3072-dim
utterance descriptor ``[e^w2v; e^hu; e^wlm]``.

To stay under the 12 GB VRAM budget of the target machines, models are loaded
one at a time: we extract the w2v embedding for the entire split, then delete
the model, then load HuBERT, etc.
"""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import torch
import torchaudio
from datasets import Audio, load_dataset
from torch import Tensor
from tqdm import tqdm

SSL_MODEL_IDS: tuple[str, str, str] = (
    "facebook/wav2vec2-large-960h",
    "facebook/hubert-large-ls960-ft",
    "microsoft/wavlm-large",
)
SSL_HIDDEN_DIM = 1024
SSL_TOTAL_DIM = 3 * SSL_HIDDEN_DIM
CACHE_FORMAT_VERSION = 1
TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class _SslCacheSidecar:
    version: int
    model_ids: tuple[str, str, str]
    split: str
    dataset_id: str
    num_examples: int
    feature_dim: int
    content_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "model_ids": list(self.model_ids),
            "split": self.split,
            "dataset_id": self.dataset_id,
            "num_examples": self.num_examples,
            "feature_dim": self.feature_dim,
            "content_hash": self.content_hash,
        }


def _cache_is_fresh(
    cache_path: Path,
    sidecar_path: Path,
    ids_path: Path,
    *,
    model_ids: tuple[str, str, str],
    split: str,
    dataset_id: str,
    expected_count: int,
) -> bool:
    if not (cache_path.exists() and sidecar_path.exists() and ids_path.exists()):
        return False
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        sidecar.get("version") == CACHE_FORMAT_VERSION
        and tuple(sidecar.get("model_ids", [])) == model_ids
        and sidecar.get("split") == split
        and sidecar.get("dataset_id") == dataset_id
        and sidecar.get("num_examples") == expected_count
    )


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


def _extract_single_model(
    model_id: str,
    dataset: object,
    total: int,
    device: torch.device,
    progress: bool,
) -> Tensor:
    """Extract 1024-dim mean-pooled embeddings from one SSL model."""

    transformers_module = cast(Any, import_module("transformers"))
    feature_extractor: Any = transformers_module.AutoFeatureExtractor.from_pretrained(
        model_id
    )
    model_any: Any = transformers_module.AutoModel.from_pretrained(model_id)
    model_any = model_any.to(device)
    model_any.eval()

    output = torch.empty(total, SSL_HIDDEN_DIM, dtype=torch.float32)
    progress_iter = tqdm(
        range(total),
        desc=f"ssl/{model_id.split('/')[-1]}",
        disable=not progress,
    )
    typed_dataset = cast(Any, dataset)
    with torch.no_grad():
        for index in progress_iter:
            example = cast(dict[str, Any], typed_dataset[int(index)])
            waveform = _load_waveform(cast(dict[str, Any], example["audio"]))
            processed: Any = feature_extractor(
                waveform.cpu().numpy(),
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
            )
            input_values = cast(Tensor, processed.input_values).to(device)
            hidden = cast(Tensor, model_any(input_values).last_hidden_state)
            pooled = hidden.mean(dim=1).squeeze(0).detach().cpu().to(torch.float32)
            output[index] = pooled

    del model_any
    del feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def extract_ssl_utterance_for_split(
    split: str,
    cache_dir: Path,
    dataset_id: str,
    device: torch.device | None = None,
    progress: bool = True,
    max_examples: int | None = None,
) -> Path:
    """Extract utterance-level SSL features and cache to disk.

    The cache payload is a ``.pt`` file holding a ``(N, 3072)`` float32
    tensor. An ``ids.json`` sidecar maps row index to utterance id, and a
    ``.json`` sidecar records version/hash/model ids so stale caches can be
    detected.
    """

    feature_dir = cache_dir / "features" / "ssl_utt"
    feature_dir.mkdir(parents=True, exist_ok=True)
    cache_path = feature_dir / f"{split}.pt"
    ids_path = feature_dir / f"{split}_ids.json"
    sidecar_path = feature_dir / f"{split}.json"

    raw_any: Any = load_dataset(dataset_id, split=split)
    raw_any = raw_any.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    total = len(raw_any)
    if max_examples is not None:
        total = min(total, int(max_examples))

    if _cache_is_fresh(
        cache_path,
        sidecar_path,
        ids_path,
        model_ids=SSL_MODEL_IDS,
        split=split,
        dataset_id=dataset_id,
        expected_count=total,
    ):
        return cache_path

    resolved_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    utterance_ids: list[str] = []
    for index in range(total):
        example = cast(dict[str, Any], raw_any[int(index)])
        utterance_ids.append(str(example.get("id") or example.get("speaker") or index))

    slices: list[Tensor] = []
    for model_id in SSL_MODEL_IDS:
        slices.append(
            _extract_single_model(
                model_id=model_id,
                dataset=raw_any,
                total=total,
                device=resolved_device,
                progress=progress,
            )
        )
    features = torch.cat(slices, dim=-1)
    if features.shape != (total, SSL_TOTAL_DIM):
        raise RuntimeError(
            f"SSL feature tensor has wrong shape {tuple(features.shape)}; "
            f"expected ({total}, {SSL_TOTAL_DIM})"
        )

    hasher = hashlib.sha256()
    hasher.update("|".join(SSL_MODEL_IDS).encode("utf-8"))
    hasher.update(str(features.shape).encode("utf-8"))
    hasher.update(str(total).encode("utf-8"))
    content_hash = hasher.hexdigest()

    torch.save(
        {"features": features, "model_ids": list(SSL_MODEL_IDS)},
        cache_path,
    )
    ids_path.write_text(
        json.dumps({"utterance_ids": utterance_ids}, indent=2), encoding="utf-8"
    )
    sidecar = _SslCacheSidecar(
        version=CACHE_FORMAT_VERSION,
        model_ids=SSL_MODEL_IDS,
        split=split,
        dataset_id=dataset_id,
        num_examples=total,
        feature_dim=SSL_TOTAL_DIM,
        content_hash=content_hash,
    )
    sidecar_path.write_text(
        json.dumps(sidecar.to_dict(), indent=2), encoding="utf-8"
    )
    return cache_path
