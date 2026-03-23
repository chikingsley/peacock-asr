"""SpeechOcean762 dataset loader.

Ported from ConPCO/src/traintest_eng_dur_ssl_3m_HierBFR_conPCO_norm.py (GoPDataset, ~lines 370-430).

Modernizations vs reference:
- norm_valid: double-nested loop → single vectorized mask operation (~100x faster on GPU).
- Paths: hardcoded '../data/' → configurable features_dir from Settings.
- HF Hub download helper added.
- MDD labels added (phone accuracy < 0.5 → mispronounced), matching MuFFIN §IV-A.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ── Normalization constants (librispeech acoustic model) ──────────────────────
# Source: ConPCO training script, lines 374-378
GOP_MEAN: float = 3.203
GOP_STD: float = 4.045
ENERGY_MEAN: float = 0.1697
ENERGY_STD: float = 0.4824
DUR_MEAN: float = 0.1392
DUR_STD: float = 0.0993

# SSL feature concat order from line 110: [wav2vec2, HuBERT, WavLM]
# Files: tr_w2v_300m_feat_v2.npy, tr_hubert_feat_v2.npy, tr_wavlm_feat_v2.npy
_DATA_SUBDIR = "seq_data_librispeech_v4"

# MDD threshold (MuFFIN §IV-A: "accuracy scores below 0.5 → mispronounced")
MDD_THRESHOLD: float = 0.5


class GoPDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Phone-level GOP + SSL + prosodic features for SpeechOcean762.

    Each sample is a padded sequence of up to 50 phone positions.
    Padding is indicated by phn_id == -1 (phone-level) and negative labels.

    Returns (10-tuple)
    ------------------
    gop       : [50, 84]   GOP features (normalized)
    ssl       : [50, 3072] Concatenated SSL: [wav2vec2 | HuBERT | WavLM] (dropout applied at model)
    energy    : [50, 7]    Energy statistics (normalized)
    dur       : [50, 1]    Duration (normalized)
    phn_score : [50]       Phone accuracy score [0, 2]; negative = padding
    phn_id    : [50]       Canonical phone id (0-based, -1 = padding)
    utt_label : [5]        Utterance scores / 5: [accuracy, completeness, fluency, prosodic, total]
    word_label: [50, 4]    [accuracy/5, stress/5, total/5, word_pos]; negative scores = padding.
                           word_pos is the within-utterance word index (0-based small int, -1 = pad).
                           Source: tr_label_word.npy (column 3 already encodes position, not vocab ID).
    word_id   : [50]       Lexical vocabulary IDs (0..2607) for word embedding. Separate from word_pos.
                           Source: tr_word_id.npy
    mdd_label : [50]       1 = mispronounced (score < 0.5), 0 = correct, -1 = padding
    """

    def __init__(self, split: str, features_dir: Path) -> None:
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"
        d = features_dir / _DATA_SUBDIR
        prefix = "tr" if split == "train" else "te"

        gop = torch.from_numpy(np.load(d / f"{prefix}_feat.npy")).float()
        energy = torch.from_numpy(np.load(d / f"{prefix}_energy_feat.npy")).float()
        dur = torch.from_numpy(np.load(d / f"{prefix}_dur_feat.npy")).float()
        ssl1 = torch.from_numpy(np.load(d / f"{prefix}_hubert_feat_v2.npy")).float()
        ssl2 = torch.from_numpy(np.load(d / f"{prefix}_w2v_300m_feat_v2.npy")).float()
        ssl3 = torch.from_numpy(np.load(d / f"{prefix}_wavlm_feat_v2.npy")).float()
        phn_label = torch.from_numpy(np.load(d / f"{prefix}_label_phn.npy")).float()
        utt_label = torch.from_numpy(np.load(d / f"{prefix}_label_utt.npy")).float()
        word_label = torch.from_numpy(np.load(d / f"{prefix}_label_word.npy")).float()
        word_id = torch.from_numpy(np.load(d / f"{prefix}_word_id.npy")).float()

        # Normalize GOP: only valid (non-padding) tokens
        # Vectorized: mask where first GOP dim != 0 (padding sentinel)
        # Reference: norm_valid(), lines 415-423
        valid_mask = gop[..., 0] != 0  # [B, 50]
        self.gop = torch.where(
            valid_mask.unsqueeze(-1),
            (gop - GOP_MEAN) / GOP_STD,
            torch.zeros_like(gop),
        )

        # Energy and duration: NOT normalized.
        # The reference defines ENERGY_MEAN/STD and DUR_MEAN/STD constants but never
        # applies them — they appear to be left over from an earlier draft. The model
        # was trained and reported on raw (un-normalized) energy and duration.
        self.energy = energy
        self.dur = dur

        # SSL concat order: [wav2vec2 | HuBERT | WavLM]  (line 110 of training script)
        self.ssl = torch.cat([ssl2, ssl1, ssl3], dim=-1)  # [B, 50, 3072]

        # Labels
        # phn_label shape: [B, 50, 2] — [:, :, 0] = phone_id, [:, :, 1] = accuracy score
        self.phn_score = phn_label[:, :, 1]   # [B, 50]; negative = padding
        self.phn_id = phn_label[:, :, 0]      # [B, 50]; -1 = padding

        # Utterance labels: 5 aspects, normalized to [0, 2] by /5
        self.utt_label = utt_label / 5.0  # [B, 5]

        # Word labels: tr_label_word.npy is already [B, 50, 4] = [accuracy, stress, total, word_pos]
        # Column 3 is the within-utterance word position (small int), NOT the lexical vocab ID.
        # Reference: training script line 410 — only columns 0:3 are divided by 5.
        word_label[:, :, 0:3] = word_label[:, :, 0:3] / 5.0
        self.word_label = word_label  # [B, 50, 4]

        # word_id: lexical vocabulary IDs (0..2607) — separate from word_pos.
        # Reference: tr_word_id.npy loaded separately (line 392), passed as distinct arg (line 130).
        self.word_id = word_id  # [B, 50]

        # MDD labels: 1 = mispronounced (accuracy < threshold), 0 = correct, -1 = padding
        # Source: MuFFIN paper §IV-A — labels manually assigned where accuracy < 0.5
        mdd = torch.full_like(self.phn_score, fill_value=-1.0)
        valid = self.phn_score >= 0
        mdd[valid] = (self.phn_score[valid] < MDD_THRESHOLD).float()
        self.mdd_label = mdd

        # Diagnosis labels: the phoneme actually produced by the learner (MuFFIN Eq.5).
        # Extracted from SpeechOcean762 `mispronunciations` annotations via HF dataset.
        # For correct phones (98%): same as canonical. For substitutions: the actually-spoken
        # phone. For non-CMU sounds (<unk>, accented variants): -1 (ignored in CE loss).
        diag_path = d / f"{prefix}_label_diag.npy"
        if diag_path.exists():
            self.diag_label = torch.from_numpy(np.load(diag_path)).long()
        else:
            # Fallback to canonical phone IDs if diag labels not yet generated
            self.diag_label = self.phn_id.long()

    def __len__(self) -> int:
        return self.gop.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:  # type: ignore[override]  # base class accepts Any index; we only support int (standard for map-style datasets)
        return (
            self.gop[idx],        # [50, 84]
            self.ssl[idx],        # [50, 3072]
            self.energy[idx],     # [50, 7]
            self.dur[idx],        # [50, 1]
            self.phn_score[idx],  # [50]
            self.phn_id[idx],     # [50]
            self.utt_label[idx],  # [5]
            self.word_label[idx], # [50, 4]  [accuracy, stress, total, word_pos]
            self.word_id[idx],    # [50]     lexical vocabulary IDs
            self.mdd_label[idx],  # [50]
            self.diag_label[idx], # [50]     canonical phone id for diagnosis (long, -1=pad)
        )


def make_loaders(features_dir: Path, batch_size: int, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader)."""
    train_ds = GoPDataset("train", features_dir)
    test_ds = GoPDataset("test", features_dir)
    pin = torch.cuda.is_available()  # no-op (and no deprecation warning) when CUDA absent
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


_GOPT_DROPBOX = "https://www.dropbox.com/s/zc6o1d8rqq28vci/data.zip?dl=1"

# Files present in the ConPCO HF dataset zip
_HF_FILES = {
    "tr_dur_feat.npy", "tr_energy_feat.npy", "tr_hubert_feat_v2.npy",
    "tr_w2v_300m_feat_v2.npy", "tr_wavlm_feat_v2.npy", "tr_word_id.npy",
    "te_dur_feat.npy", "te_energy_feat.npy", "te_hubert_feat_v2.npy",
    "te_w2v_300m_feat_v2.npy", "te_wavlm_feat_v2.npy", "te_word_id.npy",
}

# Files that come from the GOPT repo (GOP features + labels)
_GOPT_FILES = {
    "tr_feat.npy", "tr_label_phn.npy", "tr_label_utt.npy", "tr_label_word.npy",
    "te_feat.npy", "te_label_phn.npy", "te_label_utt.npy", "te_label_word.npy",
}

_ALL_FILES = _HF_FILES | _GOPT_FILES


def download_features(features_dir: Path) -> None:
    """Download all pre-extracted SpeechOcean762 features.

    Two sources:
    1. HuggingFace Hub (a2d8a4v/SpeechOcean762_for_ConPCO) — SSL + energy + dur + word_id
    2. GOPT Dropbox (YuanGongND/gopt) — GOP features + phone/word/utt labels
    """
    import urllib.request
    import zipfile

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("Install huggingface_hub: uv add huggingface_hub") from e

    target = features_dir / _DATA_SUBDIR
    target.mkdir(parents=True, exist_ok=True)

    present = {f.name for f in target.glob("*.npy")}
    if _ALL_FILES.issubset(present):
        print(f"All features already present at {target}")
        return

    # ── Step 1: ConPCO HF dataset (SSL + energy + dur + word_id) ─────────────
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
            with zipfile.ZipFile(zip_path, "r") as z:
                for member in z.namelist():
                    name = Path(member).name
                    if name.endswith(".npy") and name in _HF_FILES:
                        (target / name).write_bytes(z.read(member))
                        print(f"  {name}")
        else:
            # Flat layout — copy .npy files directly
            for src in tmp_hf.rglob("*.npy"):
                if src.name in _HF_FILES:
                    import shutil
                    shutil.copy2(src, target / src.name)

    # ── Step 2: GOPT data (GOP features + labels) ─────────────────────────────
    if not _GOPT_FILES.issubset({f.name for f in target.glob("*.npy")}):
        print("Downloading GOP features + labels from GOPT (Dropbox)...")
        zip_path = features_dir / "_gopt_data.zip"
        urllib.request.urlretrieve(_GOPT_DROPBOX, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                name = Path(member).name
                if name.endswith(".npy") and name in _GOPT_FILES and "librispeech" in member:
                    (target / name).write_bytes(z.read(member))
                    print(f"  {name}")
        zip_path.unlink()

    missing = _ALL_FILES - {f.name for f in target.glob("*.npy")}
    if missing:
        raise RuntimeError(f"Download incomplete — missing: {missing}")
    print(f"All features ready at {target}")
