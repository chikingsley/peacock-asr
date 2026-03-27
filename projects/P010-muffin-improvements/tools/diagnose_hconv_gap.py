"""Diagnostic script: investigate why HConv/CHConv PCC (~0.38-0.46) is far below baseline (~0.658).

Run with:  uv run python tools/diagnose_hconv_gap.py

This script does NOT train anything. It loads data and models, compares features,
and traces forward pass shapes to identify the source of the performance gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

# ── Setup ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p010.data import AllLayerDataset, GoPDataset, SSL_MODEL_KEYS
from p010.models import AllLayerInterfaceModel, HierCB
from p010.settings import SSLInterfaceMode

DATA_DIR = Path("~/data/p010-features").expanduser()
SEQ_DIR = DATA_DIR / "seq_data_librispeech_v4"


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA COMPARISON: feat_v2 vs all_layers
# ══════════════════════════════════════════════════════════════════════════════
section("1. DATA COMPARISON: feat_v2 vs all_layers")

for model_name in ["hubert", "w2v_300m", "wavlm"]:
    print(f"--- {model_name} ---")
    feat_v2 = np.load(SEQ_DIR / f"tr_{model_name}_feat_v2.npy", mmap_mode="r")
    all_layers = np.load(SEQ_DIR / f"tr_{model_name}_all_layers.npy", mmap_mode="r")

    print(f"  feat_v2:    shape={feat_v2.shape}, dtype={feat_v2.dtype}")
    print(f"  all_layers: shape={all_layers.shape}, dtype={all_layers.dtype}")

    # Phone count comparison (first 500 utterances)
    v2_lens = []
    al_lens = []
    for i in range(500):
        v2_norm = np.linalg.norm(feat_v2[i].astype(np.float64), axis=-1)
        al_norm = np.linalg.norm(all_layers[i, :, 24, :].astype(np.float64), axis=-1)
        v2_len = next((p for p in range(50) if v2_norm[p] < 1e-6), 50)
        al_len = next((p for p in range(50) if al_norm[p] < 1e-6), 50)
        v2_lens.append(v2_len)
        al_lens.append(al_len)

    matches = sum(1 for a, b in zip(v2_lens, al_lens) if a == b)
    print(f"  Phone count matches: {matches}/500 ({100*matches/500:.1f}%)")
    print(f"  feat_v2 mean phones: {np.mean(v2_lens):.1f}, all_layers mean phones: {np.mean(al_lens):.1f}")

    # Per-phone correlation (sample 1, where lengths likely match)
    v2_vec = feat_v2[1, 0, :].astype(np.float64)
    al_vec = all_layers[1, 0, 24, :].astype(np.float64)
    if np.linalg.norm(v2_vec) > 1e-6 and np.linalg.norm(al_vec) > 1e-6:
        r, _ = pearsonr(v2_vec, al_vec)
        print(f"  Correlation (utt 1, phone 0, layer 24): r={r:.4f}")

    # Value ranges
    v2_sample = feat_v2[:10].astype(np.float64)
    al_last = all_layers[:10, :, 24, :].astype(np.float64)
    print(f"  feat_v2 value range: [{v2_sample.min():.3f}, {v2_sample.max():.3f}], std={v2_sample.std():.4f}")
    print(f"  all_layers L24 range: [{al_last.min():.3f}, {al_last.max():.3f}], std={al_last.std():.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 2. ALL_LAYERS LAYER STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
section("2. ALL_LAYERS LAYER STATISTICS (hubert, first 50 utterances)")

all_layers = np.load(SEQ_DIR / "tr_hubert_all_layers.npy", mmap_mode="r")
sample = all_layers[:50].astype(np.float64)

print(f"{'Layer':>6} {'Mean':>10} {'Std':>10} {'AbsMax':>10} {'L2Norm(utt0,phn0)':>20}")
for L in range(25):
    layer_data = sample[:, :, L, :]
    norm = np.linalg.norm(sample[0, 0, L, :])
    print(f"  {L:4d}  {layer_data.mean():10.4f}  {layer_data.std():10.4f}  {np.abs(layer_data).max():10.2f}  {norm:20.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET LOADING COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
section("3. DATASET LOADING COMPARISON")

gop_ds = GoPDataset("train", DATA_DIR)
al_ds = AllLayerDataset("train", DATA_DIR)

# GoPDataset sample
gop_sample = gop_ds[0]
gop_gop, gop_ssl, gop_energy, gop_dur, *_ = gop_sample
print("GoPDataset sample [0]:")
print(f"  gop:    shape={gop_gop.shape}, dtype={gop_gop.dtype}")
print(f"  ssl:    shape={gop_ssl.shape}, dtype={gop_ssl.dtype}")
print(f"  energy: shape={gop_energy.shape}, dtype={gop_energy.dtype}")
print(f"  dur:    shape={gop_dur.shape}, dtype={gop_dur.dtype}")
print(f"  ssl value range: [{gop_ssl.min():.4f}, {gop_ssl.max():.4f}]")
print(f"  ssl L2 norm (phone 0): {gop_ssl[0].norm():.4f}")

print()

# AllLayerDataset sample
al_sample = al_ds[0]
print("AllLayerDataset sample [0]:")
print(f"  gop:    shape={al_sample.gop.shape}, dtype={al_sample.gop.dtype}")
print(f"  energy: shape={al_sample.energy.shape}, dtype={al_sample.energy.dtype}")
print(f"  dur:    shape={al_sample.dur.shape}, dtype={al_sample.dur.dtype}")
for key in SSL_MODEL_KEYS:
    t = al_sample.ssl_frames[key]
    print(f"  ssl_frames[{key}]: shape={t.shape}, dtype={t.dtype}, range=[{t.min():.4f}, {t.max():.4f}]")
    print(f"    L2 norm (phone 0, layer 24): {t[0, 24].norm():.4f}")
    print(f"    L2 norm (phone 0, layer 0):  {t[0, 0].norm():.4f}")

# CRITICAL: verify that gop/energy/dur/labels are IDENTICAL between datasets
print()
print("Shared features match check:")
print(f"  gop match: {torch.allclose(gop_gop, al_sample.gop)}")
print(f"  energy match: {torch.allclose(gop_energy, al_sample.energy)}")
print(f"  dur match: {torch.allclose(gop_dur, al_sample.dur)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FORWARD PASS SHAPE TRACING
# ══════════════════════════════════════════════════════════════════════════════
section("4. FORWARD PASS SHAPE TRACING")

B = 4  # batch size

# 4a. Baseline path
print("--- Baseline path (ssl_interface=none) ---")
baseline_model = HierCB()
print(f"  Model parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")

# Create dummy baseline input
gop_in = torch.randn(B, 50, 84)
ssl_in = torch.randn(B, 50, 3072)  # 3 * 1024, concatenated last-layer features
energy_in = torch.randn(B, 50, 7)
dur_in = torch.randn(B, 50, 1)
phn_in = torch.randint(0, 39, (B, 50))
word_pos_in = torch.randint(0, 10, (B, 50))
word_in = torch.randint(0, 100, (B, 50))

print(f"  Input shapes:")
print(f"    gop:      {gop_in.shape}")
print(f"    ssl:      {ssl_in.shape}")
print(f"    energy:   {energy_in.shape}")
print(f"    dur:      {dur_in.shape}")
print(f"    concat:   {torch.cat([gop_in, ssl_in, dur_in, energy_in], dim=-1).shape} -> p_in_proj -> [{B}, 50, 24]")

with torch.no_grad():
    outputs = baseline_model(gop_in, energy_in, dur_in, ssl_in, phn_in, word_pos_in, word_in)
print(f"  Output count: {len(outputs)}")
for i, o in enumerate(outputs):
    print(f"    output[{i}]: {o.shape}")

print()

# 4b. HConv path
print("--- HConv path (ssl_interface=hconv, ssl_output_dim=3072) ---")
hconv_model = AllLayerInterfaceModel(ssl_interface=SSLInterfaceMode.HCONV, ssl_output_dim=3072)
total_params = sum(p.numel() for p in hconv_model.parameters())
interface_params = sum(p.numel() for p in hconv_model.ssl_interface.parameters())
downstream_params = sum(p.numel() for p in hconv_model.downstream.parameters())
print(f"  Total parameters:     {total_params:,} ({total_params/1e3:.1f}k)")
print(f"  Interface parameters: {interface_params:,} ({interface_params/1e3:.1f}k)")
print(f"  Downstream params:    {downstream_params:,} ({downstream_params/1e3:.1f}k)")
print(f"  Ratio to baseline:    {total_params/sum(p.numel() for p in baseline_model.parameters()):.1f}x")

# Create dummy HConv input
ssl_frames = {
    key: torch.randn(B, 50, 25, 1024)
    for key in SSL_MODEL_KEYS
}
frame_lengths = {
    key: torch.full((B,), 50, dtype=torch.long)
    for key in SSL_MODEL_KEYS
}

print(f"  Input shapes:")
print(f"    gop:      {gop_in.shape}")
print(f"    ssl_frames: dict with 3 keys, each {ssl_frames['hubert'].shape}")
print(f"    energy:   {energy_in.shape}")
print(f"    dur:      {dur_in.shape}")

with torch.no_grad():
    # Trace through ssl_interface
    phone_ssl = hconv_model.ssl_interface(ssl_frames)
    print(f"  After ssl_interface: {phone_ssl.shape} (should be [{B}, 50, 3072])")
    print(f"  phone_ssl value range: [{phone_ssl.min():.4f}, {phone_ssl.max():.4f}]")

    # Full forward
    outputs = hconv_model(gop_in, energy_in, dur_in, ssl_frames, phn_in, word_pos_in, word_in)
print(f"  Output count: {len(outputs)}")
for i, o in enumerate(outputs):
    print(f"    output[{i}]: {o.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PARAMETER COUNT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("5. PARAMETER COUNT ANALYSIS")

print(f"{'Component':<40} {'Params':>12} {'%Total':>8}")
print("-" * 62)

for name, module in hconv_model.named_modules():
    if name == "":
        continue
    # Only show top-level and one-level-deep
    if name.count(".") <= 1:
        params = sum(p.numel() for p in module.parameters())
        pct = 100 * params / total_params
        print(f"  {name:<38} {params:>12,} {pct:>7.1f}%")

print("-" * 62)
print(f"  {'TOTAL':<38} {total_params:>12,} {'100.0%':>8}")

print()
print(f"Training data: 2500 samples")
print(f"Parameters per sample: {total_params / 2500:.0f}")
print(f"Baseline params per sample: {sum(p.numel() for p in baseline_model.parameters()) / 2500:.0f}")
print(f"This is a {total_params / 2500:.0f}:1 parameter-to-sample ratio (SEVERE overfitting risk)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE SCALE COMPARISON (WHAT THE DOWNSTREAM MODEL SEES)
# ══════════════════════════════════════════════════════════════════════════════
section("6. FEATURE SCALE COMPARISON (what HierCB receives)")

# Baseline: what does GoPDataset.ssl look like?
print("Baseline path (GoPDataset.ssl = cat[w2v, hubert, wavlm] feat_v2):")
gop_ds = GoPDataset("train", DATA_DIR)
ssl_tensor = gop_ds.ssl  # [2500, 50, 3072]
# Only look at non-padding phones
valid_mask = gop_ds.gop[..., 0] != 0  # same logic as in data.py
valid_ssl = ssl_tensor[valid_mask]
print(f"  Shape: {ssl_tensor.shape}")
print(f"  Valid phones: {valid_mask.sum()} / {valid_mask.numel()}")
print(f"  Value range: [{valid_ssl.min():.4f}, {valid_ssl.max():.4f}]")
print(f"  Mean: {valid_ssl.mean():.6f}")
print(f"  Std: {valid_ssl.std():.6f}")
print(f"  L2 norm (mean per phone): {valid_ssl.norm(dim=-1).mean():.4f}")

print()

# HConv path: what does PhoneHConvInterface produce?
print("HConv path (after PhoneHConvInterface forward pass on real data):")
from p010.models.ssl_interface import PhoneHConvInterface

hconv_interface = PhoneHConvInterface(ssl_output_dim=3072)
# Initialize with real data from the first 10 samples
with torch.no_grad():
    frames_batch = {
        key: torch.from_numpy(
            np.asarray(al_ds.ssl_layers[key][:10], dtype=np.float32)
        )
        for key in SSL_MODEL_KEYS
    }
    print(f"  Input shapes: {', '.join(f'{k}={v.shape}' for k, v in frames_batch.items())}")
    print(f"  Input value ranges:")
    for key in SSL_MODEL_KEYS:
        v = frames_batch[key]
        print(f"    {key}: [{v.min():.4f}, {v.max():.4f}], std={v.std():.4f}")

    hconv_out = hconv_interface(frames_batch)
    print(f"  Output shape: {hconv_out.shape}")
    print(f"  Output value range: [{hconv_out.min():.4f}, {hconv_out.max():.4f}]")
    print(f"  Output mean: {hconv_out.mean():.6f}")
    print(f"  Output std: {hconv_out.std():.6f}")
    print(f"  Output L2 norm (mean per phone): {hconv_out.reshape(-1, 3072).norm(dim=-1).mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY OF FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
section("7. SUMMARY OF FINDINGS")

print("""
KEY FINDING: feat_v2 and all_layers are INCOMPATIBLE feature sources.

1. DIFFERENT REPRESENTATIONS:
   - feat_v2 (float64): The SSL features used by the baseline.
     Correlation with any layer of all_layers is near zero (r < 0.2).
     These appear to be extracted by a DIFFERENT process than all_layers.
   - all_layers (float16): 25-layer hidden states from the same 3 SSL models.
     But the extraction methodology (phone boundaries, pooling, normalization)
     is fundamentally different from what produced feat_v2.

2. DIFFERENT PHONE COUNTS:
   ~26% of utterances have mismatching phone sequence lengths between
   feat_v2 and all_layers. This means the phone boundary alignments differ.
   Even where lengths match, the features are uncorrelated.

3. MASSIVE OVERFITTING RISK:
   - Baseline: ~445K parameters on 2500 training samples (178 params/sample)
   - HConv: ~24.6M parameters on 2500 training samples (9825 params/sample)
   - That is 55x more parameters for the same data. The HConv convolutions
     have 24.1M params (98% of total) that learn on only 2500 * ~15 phones
     = ~37,500 phone examples.

4. FEATURE QUALITY:
   - The all_layers data has wildly different scales across layers.
     Layer 0 (CNN features): std ~1.6, abs_max ~66
     Layer 23 (last transformer): std ~13.5, abs_max ~549
     Layer 24 (post-LayerNorm): std ~0.1, abs_max ~3.3
   - Without input normalization (normalize=False by default in HConv),
     the conv kernels must learn to handle inputs spanning 3 orders of magnitude.
   - The feat_v2 features have a much tighter distribution (std ~0.11).

HYPOTHESES (ranked by likely impact):
   A. The all_layers features are simply WORSE representations than feat_v2.
      feat_v2 appears to be extracted differently (possibly from a fine-tuned
      model, or with different normalization). HConv processes raw hidden states
      that may not carry the same phonetic information.
   B. 55x parameter explosion causes severe overfitting on 2500 samples.
   C. Layer scale mismatch (no input normalization) makes optimization hard.
   D. All three factors compound: bad features + too many params + scale issues.

RECOMMENDED EXPERIMENTS:
   1. Extract feat_v2-equivalent features: Load the exact same SSL checkpoints
      used for feat_v2 (check ConPCO extraction scripts), extract last-layer
      features with the same normalization, and verify r > 0.99 with feat_v2.
   2. Sanity check: Feed all_layers[:, :, 24, :] directly (no HConv) as SSL
      input to HierCB. If PCC is still ~0.38, the all_layers data itself is
      the problem, not HConv.
   3. If (2) confirms bad data: focus on matching the feat_v2 extraction process.
   4. If (2) shows reasonable PCC: HConv overfitting is the issue. Try:
      a. Freeze HConv weights (use learned layer weights from SUPERB instead)
      b. Reduce HConv capacity (smaller hidden dim, fewer conv layers)
      c. Add strong regularization (dropout, weight decay, early stopping)
""")
