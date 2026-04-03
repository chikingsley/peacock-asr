"""SpeechOcean762 dataset loaders.

Two data paths are supported:

1. ``ssl_interface=none`` loads the original phone-level last-layer SSL features.
2. ``ssl_interface=last/hconv/chconv`` loads per-utterance frame-level SSL shards from
   ``ssl_frame_store_v1`` so the interface can run before phone pooling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from p011.frame_store import load_manifest, shard_path
from p011.settings import SSLInterfaceMode
from p011.ssl_features import SSL_LAST_LAYER_FILES, SSL_MODEL_KEYS, SSLModelKey

# ── Normalization constants (librispeech acoustic model) ──────────────────────
# Source: ConPCO training script, lines 374-378
GOP_MEAN: float = 3.203
GOP_STD: float = 4.045
ENERGY_MEAN: float = 0.1697
ENERGY_STD: float = 0.4824
DUR_MEAN: float = 0.1392
DUR_STD: float = 0.0993

_DATA_SUBDIR = "seq_data_librispeech_v4"

# MDD threshold (MuFFIN §IV-A: "accuracy scores below 0.5 → mispronounced")
MDD_THRESHOLD: float = 0.5

type PhoneBatch = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


@dataclass(frozen=True)
class SharedFeatures:
    """All non-SSL tensors shared by both dataset variants."""

    gop: Tensor
    energy: Tensor
    dur: Tensor
    phn_score: Tensor
    phn_id: Tensor
    utt_label: Tensor
    word_label: Tensor
    word_id: Tensor
    mdd_label: Tensor
    diag_label: Tensor


@dataclass(frozen=True)
class FrameSample:
    """One dataset sample with frame-level SSL features."""

    gop: Tensor
    energy: Tensor
    dur: Tensor
    phn_score: Tensor
    phn_id: Tensor
    utt_label: Tensor
    word_label: Tensor
    word_id: Tensor
    mdd_label: Tensor
    diag_label: Tensor
    ssl_frames: dict[SSLModelKey, Tensor]


@dataclass(frozen=True)
class FrameBatch:
    """One collated batch with padded frame-level SSL features."""

    gop: Tensor
    energy: Tensor
    dur: Tensor
    phn_score: Tensor
    phn_id: Tensor
    utt_label: Tensor
    word_label: Tensor
    word_id: Tensor
    mdd_label: Tensor
    diag_label: Tensor
    ssl_frames: dict[SSLModelKey, Tensor]
    frame_lengths: dict[SSLModelKey, Tensor]


type Batch = PhoneBatch | FrameBatch


def _dataset_tensor(dataset: Dataset[Batch], attr: str) -> Tensor:
    """Read one tensor attribute from a dataset or Subset."""
    if isinstance(dataset, Subset):
        base = getattr(dataset.dataset, attr)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return base[indices]
    return getattr(dataset, attr)


def _load_tensor(path: Path, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Load one ``.npy`` file into a torch tensor."""
    array = np.load(path)
    tensor = torch.from_numpy(array)
    return tensor.to(dtype=dtype)


def _load_shared_features(split: str, features_dir: Path) -> SharedFeatures:
    """Load all non-SSL tensors shared by both dataset variants."""
    d = features_dir / _DATA_SUBDIR
    prefix = "tr" if split == "train" else "te"

    gop = _load_tensor(d / f"{prefix}_feat.npy")
    energy = _load_tensor(d / f"{prefix}_energy_feat.npy")
    dur = _load_tensor(d / f"{prefix}_dur_feat.npy")
    phn_label = _load_tensor(d / f"{prefix}_label_phn.npy")
    utt_label = _load_tensor(d / f"{prefix}_label_utt.npy")
    word_label = _load_tensor(d / f"{prefix}_label_word.npy")
    word_id = _load_tensor(d / f"{prefix}_word_id.npy")

    valid_mask = gop[..., 0] != 0
    gop_norm = torch.where(
        valid_mask.unsqueeze(-1),
        (gop - GOP_MEAN) / GOP_STD,
        torch.zeros_like(gop),
    )

    phn_score = phn_label[:, :, 1]
    phn_id = phn_label[:, :, 0]
    utt_label = utt_label / 5.0
    word_label = word_label.clone()
    word_label[:, :, 0:3] = word_label[:, :, 0:3] / 5.0

    mdd = torch.full_like(phn_score, fill_value=-1.0)
    valid = phn_score >= 0
    mdd[valid] = (phn_score[valid] < MDD_THRESHOLD).float()

    diag_path = d / f"{prefix}_label_diag.npy"
    diag_label = _load_tensor(diag_path, dtype=torch.long) if diag_path.exists() else phn_id.long()

    return SharedFeatures(
        gop=gop_norm,
        energy=energy,
        dur=dur,
        phn_score=phn_score,
        phn_id=phn_id,
        utt_label=utt_label,
        word_label=word_label,
        word_id=word_id,
        mdd_label=mdd,
        diag_label=diag_label,
    )


class GoPDataset(Dataset[PhoneBatch]):
    """Phone-level GOP + SSL + prosodic features for SpeechOcean762."""

    def __init__(
        self,
        split: str,
        features_dir: Path,
        ssl_model_keys: Sequence[SSLModelKey] = SSL_MODEL_KEYS,
    ) -> None:
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"
        d = features_dir / _DATA_SUBDIR
        prefix = "tr" if split == "train" else "te"
        self.ssl_model_keys = tuple(ssl_model_keys)

        shared = _load_shared_features(split, features_dir)
        ssl_tensors = [
            _load_tensor(d / f"{prefix}_{SSL_LAST_LAYER_FILES[model_key]}")
            for model_key in self.ssl_model_keys
        ]

        self.gop = shared.gop
        self.energy = shared.energy
        self.dur = shared.dur
        self.ssl = torch.cat(ssl_tensors, dim=-1)
        self.phn_score = shared.phn_score
        self.phn_id = shared.phn_id
        self.utt_label = shared.utt_label
        self.word_label = shared.word_label
        self.word_id = shared.word_id
        self.mdd_label = shared.mdd_label
        self.diag_label = shared.diag_label

    def __len__(self) -> int:
        return self.gop.shape[0]

    def __getitem__(self, idx: int) -> PhoneBatch:  # type: ignore[override]
        return (
            self.gop[idx],
            self.ssl[idx],
            self.energy[idx],
            self.dur[idx],
            self.phn_score[idx],
            self.phn_id[idx],
            self.utt_label[idx],
            self.word_label[idx],
            self.word_id[idx],
            self.mdd_label[idx],
            self.diag_label[idx],
        )


def has_frame_store_data(
    features_dir: Path,
    split: str,
    ssl_model_keys: Sequence[SSLModelKey] = SSL_MODEL_KEYS,
) -> bool:
    """Check if frame-store manifests exist for a split."""
    try:
        for model_key in ssl_model_keys:
            manifest = load_manifest(features_dir, split, model_key)
            if len(manifest.entries) == 0:
                return False
        return True
    except FileNotFoundError:
        return False


class FrameStoreDataset(Dataset[FrameSample]):
    """SpeechOcean762 with frame-level all-layer SSL features loaded per utterance."""

    def __init__(
        self,
        split: str,
        features_dir: Path,
        ssl_model_keys: Sequence[SSLModelKey] = SSL_MODEL_KEYS,
    ) -> None:
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"
        shared = _load_shared_features(split, features_dir)
        self.split = split
        self.features_dir = features_dir
        self.ssl_model_keys = tuple(ssl_model_keys)
        self.manifests = {model_key: load_manifest(features_dir, split, model_key) for model_key in self.ssl_model_keys}
        self.entries = {
            model_key: manifest.entries
            for model_key, manifest in self.manifests.items()
        }
        reference_key = self.ssl_model_keys[0]
        reference_entries = self.entries[reference_key]
        for model_key in self.ssl_model_keys[1:]:
            candidate_entries = self.entries[model_key]
            if len(candidate_entries) != len(reference_entries):
                raise ValueError(
                    f"Frame-store manifest length mismatch for {model_key}: "
                    f"{len(candidate_entries)} vs {len(reference_entries)}"
                )
            for ref_entry, cand_entry in zip(reference_entries, candidate_entries, strict=True):
                if ref_entry.utterance_id != cand_entry.utterance_id:
                    raise ValueError(
                        f"Frame-store utterance order mismatch: {reference_key}={ref_entry.utterance_id}, "
                        f"{model_key}={cand_entry.utterance_id}"
                    )

        self.gop = shared.gop
        self.energy = shared.energy
        self.dur = shared.dur
        self.phn_score = shared.phn_score
        self.phn_id = shared.phn_id
        self.utt_label = shared.utt_label
        self.word_label = shared.word_label
        self.word_id = shared.word_id
        self.mdd_label = shared.mdd_label
        self.diag_label = shared.diag_label

    def __len__(self) -> int:
        return self.gop.shape[0]

    def __getitem__(self, idx: int) -> FrameSample:  # type: ignore[override]
        ssl_frames = {}
        for model_key in self.ssl_model_keys:
            entry = self.entries[model_key][idx]
            path = shard_path(self.features_dir, self.split, model_key, entry)
            ssl_frames[model_key] = torch.from_numpy(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))
        return FrameSample(
            gop=self.gop[idx],
            energy=self.energy[idx],
            dur=self.dur[idx],
            phn_score=self.phn_score[idx],
            phn_id=self.phn_id[idx],
            utt_label=self.utt_label[idx],
            word_label=self.word_label[idx],
            word_id=self.word_id[idx],
            mdd_label=self.mdd_label[idx],
            diag_label=self.diag_label[idx],
            ssl_frames=ssl_frames,
        )


def collate_frame_samples(samples: Sequence[FrameSample]) -> FrameBatch:
    """Pad variable-length frame tensors and stack the rest."""
    if not samples:
        raise ValueError("Expected at least one sample")

    ssl_frames: dict[SSLModelKey, Tensor] = {}
    frame_lengths: dict[SSLModelKey, Tensor] = {}

    model_keys = tuple(samples[0].ssl_frames.keys())
    for sample in samples[1:]:
        if tuple(sample.ssl_frames.keys()) != model_keys:
            raise ValueError("All frame samples must contain the same SSL model keys in the same order")

    for model_key in model_keys:
        model_tensors = [sample.ssl_frames[model_key] for sample in samples]
        lengths = torch.tensor([tensor.shape[0] for tensor in model_tensors], dtype=torch.long)
        max_frames = int(lengths.max().item())
        num_layers = model_tensors[0].shape[1]
        feat_dim = model_tensors[0].shape[2]
        padded = torch.zeros(len(samples), max_frames, num_layers, feat_dim, dtype=model_tensors[0].dtype)
        for index, tensor in enumerate(model_tensors):
            padded[index, : tensor.shape[0]] = tensor
        ssl_frames[model_key] = padded
        frame_lengths[model_key] = lengths

    return FrameBatch(
        gop=torch.stack([sample.gop for sample in samples]),
        energy=torch.stack([sample.energy for sample in samples]),
        dur=torch.stack([sample.dur for sample in samples]),
        phn_score=torch.stack([sample.phn_score for sample in samples]),
        phn_id=torch.stack([sample.phn_id for sample in samples]),
        utt_label=torch.stack([sample.utt_label for sample in samples]),
        word_label=torch.stack([sample.word_label for sample in samples]),
        word_id=torch.stack([sample.word_id for sample in samples]),
        mdd_label=torch.stack([sample.mdd_label for sample in samples]),
        diag_label=torch.stack([sample.diag_label for sample in samples]),
        ssl_frames=ssl_frames,
        frame_lengths=frame_lengths,
    )


def select_mdd_holdout_indices(
    dataset: Dataset[Batch],
    *,
    holdout_size: int = 500,
    seed: int = 22,
) -> list[int]:
    """Pick a deterministic held-out set that covers correct/incorrect phones."""
    phn_id = _dataset_tensor(dataset, "phn_id").long()
    mdd_label = _dataset_tensor(dataset, "mdd_label").long()

    num_utts = int(phn_id.shape[0])
    if holdout_size <= 0 or holdout_size >= num_utts:
        raise ValueError(f"holdout_size must be in [1, {num_utts - 1}], got {holdout_size}")

    utterance_pairs: list[set[tuple[int, int]]] = []
    for utt_idx in range(num_utts):
        valid = (phn_id[utt_idx] >= 0) & (mdd_label[utt_idx] >= 0)
        pairs = {
            (int(phone_id), int(label))
            for phone_id, label in zip(
                phn_id[utt_idx][valid].tolist(),
                mdd_label[utt_idx][valid].tolist(),
                strict=True,
            )
            if 0 <= int(phone_id) < 39 and int(label) in (0, 1)
        }
        utterance_pairs.append(pairs)

    import random

    rng = random.Random(seed)
    candidates = list(range(num_utts))
    rng.shuffle(candidates)

    selected: list[int] = []
    covered: set[tuple[int, int]] = set()
    target_pairs = {(phone_id, label) for phone_id in range(39) for label in (0, 1)}
    while len(selected) < holdout_size and covered != target_pairs:
        best_idx: int | None = None
        best_gain = -1
        best_size = -1
        for idx in candidates:
            pairs = utterance_pairs[idx]
            gain = len(pairs - covered)
            size = len(pairs)
            if gain > best_gain or (gain == best_gain and size > best_size):
                best_idx = idx
                best_gain = gain
                best_size = size
        if best_idx is None or best_gain <= 0:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)
        covered |= utterance_pairs[best_idx]

    remaining = holdout_size - len(selected)
    if remaining > 0:
        selected.extend(candidates[:remaining])

    return sorted(selected)


def split_train_loader_for_mdd_threshold(
    train_loader: DataLoader[Batch],
    *,
    holdout_size: int = 500,
    seed: int = 22,
) -> tuple[DataLoader[Batch], DataLoader[Batch], list[int]]:
    """Reserve the paper's 500-utterance threshold holdout from the training loader."""
    holdout_indices = select_mdd_holdout_indices(train_loader.dataset, holdout_size=holdout_size, seed=seed)
    holdout_set = set(holdout_indices)
    train_indices = [idx for idx in range(len(train_loader.dataset)) if idx not in holdout_set]

    train_subset = Subset(train_loader.dataset, train_indices)
    holdout_subset = Subset(train_loader.dataset, holdout_indices)

    common_kwargs = {
        "batch_size": train_loader.batch_size,
        "num_workers": train_loader.num_workers,
        "pin_memory": train_loader.pin_memory,
        "persistent_workers": train_loader.persistent_workers,
        "collate_fn": train_loader.collate_fn,
    }
    train_subset_loader = DataLoader(train_subset, shuffle=True, **common_kwargs)
    holdout_loader = DataLoader(holdout_subset, shuffle=False, **common_kwargs)
    return cast(DataLoader[Batch], train_subset_loader), cast(DataLoader[Batch], holdout_loader), holdout_indices


def make_loaders(
    features_dir: Path,
    batch_size: int,
    num_workers: int = 4,
    ssl_interface: SSLInterfaceMode = SSLInterfaceMode.NONE,
    ssl_model_keys: Sequence[SSLModelKey] = SSL_MODEL_KEYS,
) -> tuple[DataLoader[Batch], DataLoader[Batch]]:
    """Return ``(train_loader, test_loader)`` for the requested SSL interface mode."""
    pin = torch.cuda.is_available()
    persistent = num_workers > 0

    if ssl_interface is SSLInterfaceMode.NONE:
        train_ds: Dataset[Batch] = GoPDataset("train", features_dir, ssl_model_keys=ssl_model_keys)
        test_ds: Dataset[Batch] = GoPDataset("test", features_dir, ssl_model_keys=ssl_model_keys)
        train_loader: DataLoader[Batch] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
        )
        test_loader: DataLoader[Batch] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
        )
        return train_loader, test_loader

    train_frame_ds = FrameStoreDataset("train", features_dir, ssl_model_keys=ssl_model_keys)
    test_frame_ds = FrameStoreDataset("test", features_dir, ssl_model_keys=ssl_model_keys)
    # PyTorch's DataLoader generic tracks dataset item type, not collate_fn output type.
    train_frame_loader = cast(
        DataLoader[Batch],
        DataLoader(
            train_frame_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            collate_fn=collate_frame_samples,
        ),
    )
    test_frame_loader = cast(
        DataLoader[Batch],
        DataLoader(
            test_frame_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            collate_fn=collate_frame_samples,
        ),
    )
    return train_frame_loader, test_frame_loader


_GOPT_DROPBOX = "https://www.dropbox.com/s/zc6o1d8rqq28vci/data.zip?dl=1"

_HF_FILES = {
    "tr_dur_feat.npy",
    "tr_energy_feat.npy",
    "tr_hubert_feat_v2.npy",
    "tr_w2v_300m_feat_v2.npy",
    "tr_wavlm_feat_v2.npy",
    "tr_word_id.npy",
    "te_dur_feat.npy",
    "te_energy_feat.npy",
    "te_hubert_feat_v2.npy",
    "te_w2v_300m_feat_v2.npy",
    "te_wavlm_feat_v2.npy",
    "te_word_id.npy",
}

_GOPT_FILES = {
    "tr_feat.npy",
    "tr_label_phn.npy",
    "tr_label_utt.npy",
    "tr_label_word.npy",
    "te_feat.npy",
    "te_label_phn.npy",
    "te_label_utt.npy",
    "te_label_word.npy",
}

_ALL_FILES = _HF_FILES | _GOPT_FILES


def download_features(features_dir: Path) -> None:
    """Download all pre-extracted SpeechOcean762 features."""
    import urllib.request
    import zipfile

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - import error is environment-specific
        raise ImportError("Install huggingface_hub: uv add huggingface_hub") from exc

    target = features_dir / _DATA_SUBDIR
    target.mkdir(parents=True, exist_ok=True)

    present = {path.name for path in target.glob("*.npy")}
    if _ALL_FILES.issubset(present):
        print(f"All features already present at {target}")
        return

    if not _HF_FILES.issubset(present):
        print("Downloading SSL/energy/dur features from HuggingFace Hub...")
        tmp_hf = features_dir / "_hf_download"
        snapshot_download(
            repo_id="a2d8a4v/SpeechOcean762_for_ConPCO",
            repo_type="dataset",
            local_dir=str(tmp_hf),
        )
        zip_path = tmp_hf / "seq_data_librispeech_v4.zip"
        if zip_path.exists():
            print("Extracting ConPCO zip...")
            with zipfile.ZipFile(zip_path, "r") as archive:
                for member in archive.namelist():
                    name = Path(member).name
                    if name.endswith(".npy") and name in _HF_FILES:
                        (target / name).write_bytes(archive.read(member))
                        print(f"  {name}")
        else:
            for src in tmp_hf.rglob("*.npy"):
                if src.name in _HF_FILES:
                    import shutil

                    shutil.copy2(src, target / src.name)

    if not _GOPT_FILES.issubset({path.name for path in target.glob("*.npy")}):
        print("Downloading GOP features + labels from GOPT (Dropbox)...")
        zip_path = features_dir / "_gopt_data.zip"
        urllib.request.urlretrieve(_GOPT_DROPBOX, zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.namelist():
                name = Path(member).name
                if name.endswith(".npy") and name in _GOPT_FILES and "librispeech" in member:
                    (target / name).write_bytes(archive.read(member))
                    print(f"  {name}")
        zip_path.unlink()

    missing = _ALL_FILES - {path.name for path in target.glob("*.npy")}
    if missing:
        raise RuntimeError(f"Download incomplete — missing: {missing}")
    print(f"All features ready at {target}")
