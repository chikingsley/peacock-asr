# P011: Paper-Faithful HConv Plan

## Context

We're replicating MuFFIN (Yan et al. 2025, PCC **0.742** on SpeechOcean762) and isolating a
**paper-faithful HConv** path where hierarchical convolution is applied on frame-level SSL
hidden states before phone pooling. The current execution record is
[EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md).

**What MuFFIN is**: HierCB (hierarchical Branchformer: phone→word→utterance) + ConPCO loss + joint APA+MDD training + PhnVar for class imbalance. Input features: 84 (GOP) + 7 (energy) + 1 (duration) + 3×1024 (SSL: wav2vec2 + HuBERT + WavLM last-layer) = 3,164 dims per phone. Model is tiny (~tens of thousands params, embed_dim=24).

**What the ConPCO repo gives us**: HierCB + ConPCO (targets 0.701). Missing: joint MDD heads (binary correct/incorrect per phone) which push it to 0.742. We implement those.

**What P011 changes**: Replace the old phone-level approximation with frame-level HConv so
the operation order matches Shih & Harwath 2024: conv over SSL layers first, then pool to
the phone sequence consumed by MuFFIN.

**Current implementation status**: the repo now supports explicit SSL model subsets
(`ssl_models`), derived SSL widths, optional `ssl_output_dim` overrides for branch-preserving
ablations, gradient accumulation (`grad_accum_steps`), unique checkpoint roots, and per-run
manifests so experiments do not overwrite each other.

---

## Module Structure

```text
src/p011/
├── __init__.py
├── settings.py              # pydantic-settings, all hyperparams
├── data.py                  # ~180 lines — dataset, download, normalization
├── models/
│   ├── __init__.py
│   ├── blocks.py            # ~200 lines — BlockCNN (Branchformer), Attention, MLP
│   ├── hiercb.py            # ~300 lines — HierCB + MDD heads (full MuFFIN)
│   └── hconv.py             # ~80 lines — HConv + CHConv interfaces (Phase 2)
├── losses.py                # ~120 lines — ConPCO, masked MSE, MDD BCE
├── trainer.py               # ~250 lines — train loop, validation, W&B, checkpoints
├── eval.py                  # ~120 lines — PCC, per-phone/word/utt metrics
└── cli.py                   # ~120 lines — click: download, train, sweep, eval
tests/
├── test_data.py
├── test_models.py
├── test_losses.py
├── test_hconv.py
└── test_eval.py
```

---

## Phase 1: Replicate MuFFIN (target 0.742 PCC)

### Step 1: Data pipeline (`data.py`)

- `download_features()`: pull `a2d8a4v/SpeechOcean762_for_ConPCO` from HF Hub (`seq_data_librispeech_v4/`)
- Port `GoPDataset` from `third_party/ConPCO/src/traintest_*.py:370-430`
  - Replace hardcoded `'../data/'` paths with configurable root from Settings
  - Vectorize the double-nested-loop normalization
  - Keep exact normalization constants: GOP mean=3.203, std=4.045; energy mean=0.1697, std=0.4824; dur mean=0.1392, std=0.0993
  - Keep utt/word label normalization (/ 5)
- SSL concat order must match: `torch.cat([ssl_w2v, ssl_hubert, ssl_wavlm], dim=-1)`
- Need MDD labels: phone-level binary correct/incorrect from SpeechOcean762 (derive from phone accuracy scores: score >= threshold → correct)

### Step 2: Model blocks (`models/blocks.py`)

- Port from `third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py`
- `BlockCNN`: self-attention branch + depthwise conv branch + learned merge
- `Attention`, `MLP`, `MultiHeadedAttention` (for word masking)
- `W2UFeatGen`, `AttentionPooling`
- Replace hand-rolled `trunc_normal_` with `torch.nn.init.trunc_normal_`

### Step 3: HierCB model with MDD (`models/hiercb.py`)

- Port `HierCB` class, then add MDD heads:
  - Phone-level MDD: binary classifier head parallel to `mlp_head_phn`
  - Architecture: input_dim=92 + 3072 SSL → embed_dim=24
  - Phone: 3 × BlockCNN, Word: 2 × BlockCNN, Utterance: 1 × BlockCNN
  - Phone embedding: 40 classes, Word embedding: 2607 classes
  - Returns: (u1..u5, p, w1..w3, mdd_pred, phn_audio_feats, phn_text_feats)

### Step 4: Losses (`losses.py`)

- Port `ContrastivePhonemicOrdinalRegularizer` from `third_party/ConPCO/src/models/conPCO_norm.py`
  - Diversity term (maximize inter-phone center distance)
  - Tightness term (ordinal-weighted intra-phone scatter)
  - CLAP contrastive alignment (audio ↔ text embeddings)
- `masked_mse_loss()` — handles padding mask + scale correction
- MDD loss: `nn.BCEWithLogitsLoss` with masking
- Combined: `w_phn*MSE + w_utt*MSE + w_word*MSE + w_pco*ConPCO + w_clap*CLAP + w_mdd*BCE`

### Step 5: Evaluation (`eval.py`)

- Port and vectorize `valid_phn`, `valid_utt`, `valid_word`
- Add MDD evaluation: F1, precision, recall
- PCC as primary metric (matches all published SpeechOcean762 work)

### Step 6: Trainer (`trainer.py`)

- `train_one_config(settings, model, train_loader, test_loader)`
- Adam(lr=1e-3, weight_decay=5e-7, betas=(0.95, 0.999))
- Warmup: 100 steps linear, then ReduceLROnPlateau(patience=10)
- W&B: init per run, log all metrics, log config from Settings
- Checkpoint: save best by test phone MSE
- 100 epochs per run

### Step 7: Settings (`settings.py`)

- Model: embed_dim=24, p_depth=3, w_depth=2, u_depth=1, num_heads=1, ssl_drop=0.2
- Training: lr=1e-3, batch_size=25, grad_accum_steps=1, n_epochs=100
- Loss weights: all 1.0
- ConPCO: pco_ld=0.5, pco_lt=0.1, pco_mg=1.0, clap_t2a=0.5
- MDD: use_mdd=True, loss_w_mdd=1.0
- W&B: project="p011-muffin", entity from env
- SSL controls: `ssl_models` defaults to all 3 streams; `ssl_output_dim=None` resolves to the selected raw SSL width

### Step 8: CLI (`cli.py`)

- `uv run p011 download` — pull HF features
- `uv run p011 train --seed 22 --use-conpco --use-mdd` — single run
- `uv run p011 sweep --seeds 22,33,44,55,66` — multi-seed
- `uv run p011 eval --checkpoint path/to/model.pth`
- `--ssl-models hubert` — honest one-stream control
- `--ssl-models hubert --ssl-output-dim 3072` — shape-preserving one-stream ablation
- `--batch-size 5 --grad-accum-steps 5` — low-VRAM run with effective batch 25

### Step 9: Verification runs

| Run | Config | Target PCC | Seeds |
|-----|--------|-----------|-------|
| 1a | HierCB (MSE only, no ConPCO, no MDD) | ~0.680 | 5 |
| 1b | HierCB + ConPCO | ~0.701 | 5 |
| 1c | HierCB + ConPCO + MDD (full MuFFIN) | **~0.742** | 5 |

Seeds: 22, 33, 44, 55, 66 (matching their seed list).

---

## Phase 2: CHConv Layer Aggregation

### Step 10: All-layer feature extraction (`extract.py`)

- Download SpeechOcean762 audio
- For each model (HuBERT-Large, wav2vec2-large, WavLM-Large):
  - `model.config.output_hidden_states = True`
  - Extract all 24 layers (1024-dim each)
  - Average over phone durations using forced alignment timestamps
  - Save: `[N_utterances, 50, 24, 1024]` per model as .npy

### Step 11: HConv + CHConv (`models/hconv.py`)

- **HConv**: 1D conv over layer dimension (kernel=5, stride=3), stacked `floor(log3(L))` times
  - Input: [batch, T, L, D] → conv over L → [batch, T, D]
- **CHConv**: Multi-model fusion — concatenate models along feature dim, then HConv
  - Input: 3 models × [batch, T, L, D] → concat → [batch, T, L, 3D] → HConv → [batch, T, D']

### Step 12: Modified HierCB

- `HierCB_CHConv`: replace `nn.Linear(input_dim + 3072, embed_dim)` with CHConv interface
- CHConv replaces last-layer concat with learned all-layer aggregation
- Downstream `ssl_dim` is now derived from the selected SSL subset unless an explicit `ssl_output_dim` override is requested
- Everything downstream unchanged

---

## Phase 3: Experiments

| ID | Config | Target | Seeds |
|----|--------|--------|-------|
| 1a | HierCB (MSE only) | ~0.680 | 5 |
| 1b | HierCB + ConPCO | ~0.701 | 5 |
| 1c | Full MuFFIN (HierCB + ConPCO + MDD) | ~0.742 | 5 |
| 2.1 | MuFFIN + HConv (per-model, then concat) | > 0.742 | 3 |
| 2.2 | MuFFIN + CHConv (fused multi-model) | > 0.742 | 3 |
| 2.3 | MuFFIN + CHConv (best config) | best | 5 |

---

## Critical Reference Files

| Purpose | File |
|---------|------|
| HierCB model | `third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py` |
| Training script | `third_party/ConPCO/src/traintest_eng_dur_ssl_3m_HierBFR_conPCO_norm.py` |
| ConPCO loss | `third_party/ConPCO/src/models/conPCO_norm.py` |
| Training hyperparams | `third_party/ConPCO/src/train_hierCB.sh` |
| HConv specification | `docs/papers/asr-backbones/2406.12209-.../sections/method.tex` |
| CHConv specification | `docs/papers/asr-backbones/2511.08389-.../latex-source/sections/interface_definition.tex` |
| SSL Interface code | `github.com/atosystem/SSL_Interface` |

## Implementation Order

```text
data.py → blocks.py → hiercb.py → losses.py → eval.py → trainer.py → settings.py → cli.py → tests → Phase 1 runs → extract.py → hconv.py → Phase 2/3 runs
```

Each step is a natural commit point.
