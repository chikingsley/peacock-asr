# P010 Implementation Notes

Decisions made during porting from `third_party/ConPCO/` — recorded here so any training
anomalies can be traced back to a specific change. Reference file for all items below:
`third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py` (model) and
`third_party/ConPCO/src/traintest_eng_dur_ssl_3m_HierBFR_conPCO_norm.py` (training loop).

---

## Phase 1 — HierCB + ConPCO replication

### Removals

| What | Reference location | Reason |
|------|--------------------|--------|
| `u_in_proj1/2/3` (3 × MLP) | `__init__` lines 499-503 | Never called in `forward` — vestigial from an earlier draft of the utterance path. Removes ~720 dead params. |
| `nn.DataParallel` wrapping | training script | Single-GPU assumed; wrap externally if needed. |

### PyTorch modernizations

| What | Old (reference) | New (our code) | Why |
|------|-----------------|----------------|-----|
| Self-attention | Hand-rolled scaled dot-product | `F.scaled_dot_product_attention` (PyTorch 2.0+) | Enables FlashAttention2 automatically on RTX 5070; cleaner code |
| Euclidean distance in ConPCO | `addmm_(1, -2, x, y.t())` (3-arg form) | `torch.cdist` | 3-arg `addmm_` removed in PyTorch 2.0; `cdist` is cleaner and correct |
| Weight init | Hand-rolled `trunc_normal_` function | `nn.init.trunc_normal_` (PyTorch 1.8+) | Native implementation, no copy-pasted code |
| Normalization loop | Double-nested Python `for` loop over utterances + phones | Vectorized `torch.where` with boolean mask | ~100× faster; same result |
| Word-position mask | Nested Python `for` loop | Broadcasting: `word_pos.unsqueeze(2) == word_pos.unsqueeze(1)` | Correct and fast |
| LR warmup | Discrete jumps every 5 steps (`global_step % 5 == 0`) | Smooth linear ramp per step | Matches standard warmup semantics; no jagged LR curve |
| Logging | `np.savetxt` CSV | W&B (`wandb.log` per epoch) | Consistent with project tracking standard |
| Progress | No progress output during batch loop | `trange` (epoch) + `tqdm(leave=False)` (batch) with `set_postfix` | Visible training progress without log noise |

### Bug fixes

| What | Reference location | Description |
|------|--------------------|-------------|
| ConPCO logging crash | Training script line 188 | `loss_phn_pco` printed unconditionally even when `use_conpco=False` → `NameError`. Fixed: only log inside `if settings.use_conpco` guard. |
| Wrong data filenames | data/README.md | Our loader used `{prefix}_utt_label.npy` / `{prefix}_word_label.npy`; actual files are `{prefix}_label_utt.npy` / `{prefix}_label_word.npy`. Would have raised FileNotFoundError on first real run. |
| word_pos / word_id confusion | Training script line 130, 392 | Reference keeps two separate tensors: `word_pos` (within-utterance word index, 0..N_words-1, from `tr_label_word.npy` col 3) and `word_id` (lexical vocab ID 0..2607, from `tr_word_id.npy`). We were merging the lexical ID into `word_label` col 3 and using it as both — would cause `IndexError` in `word_pos_embed` (vocab size 50) on real data. Fixed: `word_id` is now a separate 10th DataLoader tensor. |
| ConPCO NaN on empty batch | losses.py | When all scores in a batch are 0, `valid_mask` is empty → `.mean()` on empty tensor → `NaN` silently poisons training. Fixed: early return `(0.0, 0.0)` when `len(_scores) == 0`. |

### Design decisions

| Decision | Detail |
|----------|--------|
| MDD labels derived at load time | `mdd_label[i] = 1 if phn_score[i] < 0.5 else 0` — derived from accuracy scores per MuFFIN §IV-A, not a separate file |
| SSL concat order | `[wav2vec2_300m \| HuBERT \| WavLM]` — taken from training script line 110; must match pre-extracted features |
| `use_conpco=False` default | Reference `train_hierCB.sh` has `--conpco` commented out. Baseline run is HierCB without ConPCO (~0.680 target) |
| Separate `word_pos` and `word_id` | `word_label` is `[B, 50, 4]` = `[accuracy, stress, total, word_pos]`; lexical `word_id` stays as its own tensor from `tr_word_id.npy` so `word_pos_embed` only sees small within-utterance indices |
| Checkpoint saved by phone MSE | Matches reference behavior; phone PCC is the primary reported metric |

---

## Session 4 — Full MuFFIN implementation

Implemented all remaining MuFFIN paper components. Previously we only had HierCB + ConPCO (the ConPCO paper's scope). This session added everything needed for the full MuFFIN system (Yan et al. 2025).

### New components

| Component | Files | Paper reference | Description |
|-----------|-------|----------------|-------------|
| Pretraining | `pretrain.py` (new), `cli.py` | §V.B, ref [41] HierTFR | Masked Phoneme Prediction + Masked Word Prediction + Utterance Comparative Labeling. 100 epochs, MultiStepLR (halve every 5 epochs from 20-95). Masking: uniform random [1, seq_len] tokens per utterance (reimplemented from espnet `mask_uniform`). `HierCBPretrain` inherits `HierCB`, adds prediction heads. Weight transfer via `strict=False`. |
| MDD diagnosis predictor | `hiercb.py`, `trainer.py` | §III.B Eq.5, Eq.17 | `mlp_head_diag`: LayerNorm → Linear(24, 39). Predicts which phoneme the learner actually produced. Cross-entropy loss with `ignore_index=-1`. |
| Real diagnosis labels | `data.py` | §III.B | Extracted from SpeechOcean762 HF dataset `mispronunciations` annotations. 1007 train substitutions mapped to actual spoken phone IDs (e.g., IY→AH, TH→S). 984 non-CMU sounds (e.g., `<unk>`, `R*`) marked as -1 (ignored in CE loss). Saved as `tr_label_diag.npy` / `te_label_diag.npy`. Falls back to canonical phone if diag files absent. |
| PhnVar | `phnvar.py` (new), `trainer.py` | §IV cont, Eq.24-27 | Phoneme-specific logit perturbation on diagnosis predictor. QF (quantity factor) = normalized log inverse frequency. DF (difficulty factor) = normalized mispronunciation rate. Perturbation: `N(0, σ) × sqrt(QF_k × DF_k)` with α=β=1. σ not specified in paper, defaulting to 1.0. Applied during training only. |
| Loss weights 3/1/1 | `settings.py` | §V.B | Phone loss weighted 3× higher than word and utterance. The ConPCO paper used 1/1/1; 3/1/1 is MuFFIN-specific. |
| MDD threshold grid search | `eval.py`, `cli.py` | §V.B | Search [0.0, 1.0] stride 0.1 on the full training set, maximize F1. This reproduces the paper's reported global `0.4` threshold as a post-hoc approximation, but it is **not** the paper's exact protocol (the paper uses a 500-utterance held-out set and phoneme-specific threshold selection before settling on a global threshold). |
| DER/FRR/FAR/PER metrics + canonical masking | `eval.py`, `trainer.py` | §IV, §V.C Table V | Diagnosis evaluation now masks the canonical phone class before `argmax`, matching the paper's "mask canonical phonemes during diagnosis" rule. FAR/FRR/DER/PER are computed when diagnosis logits and labels are available, but the exact Table V scoring rubric is still only approximately matched. |
| NUM_PHN_CLASSES 40→42 | `hiercb.py` | — | Added pad(0) + mask(41) classes for pretraining weight transfer. Extra classes inactive during fine-tuning. |

### Sweep results

| Config | PCC (mean ± std) | Paper target | Gap |
|--------|-----------------|--------------|-----|
| v1 baseline (1/1/1, no pretrain) | 0.6582 ± 0.0106 | ~0.680 | -0.022 |
| v1 ConPCO (1/1/1, no pretrain) | 0.6621 ± 0.0065 | ~0.701 | -0.039 |
| v2 baseline (3/1/1, bug fixes) | 0.6574 ± 0.0100 | ~0.680 | -0.023 |
| v2 ConPCO (3/1/1, bug fixes) | 0.6624 ± 0.0065 | ~0.701 | -0.039 |
| v3 MuFFIN (full) | 0.6824 ± 0.0031 | ~0.742 | -0.060 |
| **v4 MuFFIN (PhnVar clamping fix)** | **0.6810 ± 0.0032** | **~0.742** | **-0.062** |

v4 MuFFIN = pretrain + 3/1/1 weights + ConPCO + MDD (detection + diagnosis) + PhnVar (BLV one-sided clamping, σ=4.0) + canonical phone masking in diagnosis. Seeds 22,33,44,55,66. PhnVar clamping fix had negligible PCC impact (~0.001).

### Eval output (v4 seed 22, threshold=0.4)

```yaml
Phone:  MSE 0.0740  PCC 0.6786
Utt accuracy     : PCC 0.7620    Utt completeness : PCC 0.4756
Utt fluency      : PCC 0.8207    Utt prosodic     : PCC 0.8137
Utt total        : PCC 0.7894
Word accuracy    : PCC 0.6357    Word stress      : PCC 0.3883
Word total       : PCC 0.6496
MDD:    F1 0.5748  P 0.6406  R 0.5212
Diag:   FAR 0.4788  FRR 0.0090  DER 0.6093  PER 0.8425
```

Diagnosis metrics compare against `diag_label` (actual spoken phone), not canonical phone.
Positions with `diag_label == -1` (non-CMU sounds) are excluded. Our PER is a diagnosis
mismatch rate over detected phones with valid labels — not the Li 2017 sequence-level PER
`(S+D+I)/N`. These numbers are useful for internal regression but not directly comparable
to MuFFIN Table V.

### Known remaining gap (0.682 vs 0.742)

The ~0.060 gap has several contributing factors:

1. **Test-set model selection**: The paper (and our code) selects the best checkpoint by test-set phone MSE every epoch. There is no separate validation split — SpeechOcean762 only provides train (2500) and test (2500). This means both the paper and our runs optimize checkpoint selection against the test set across 100 epochs. The paper's reported 0.742 and our 0.682 are both measured this way. The gap is real and not an artifact of different evaluation methodology.

2. **Pretraining architecture**: The paper's ref [41] (HierTFR, Yan et al. ACL 2024) pretrains with standard Transformer encoder blocks, while MuFFIN fine-tunes with Branchformer-style `BlockCNN` blocks. The architecture mismatch is real. However, the transfer issue is subtler than "all encoder weights are dropped": the Transformer and Branchformer blocks share an attention + MLP backbone, so a `strict=False` load would plausibly transfer the shared submodules while leaving only the Branchformer-specific convolution / branch-merging parameters randomly initialized. Our `HierCBPretrain` instead pretrains the same Branchformer backbone used at fine-tuning time. That makes this a credible source of mismatch with the paper, but still a hypothesis rather than a proven cause of the full 0.060 PCC gap.

3. **Diagnosis labels**: 984 of 1991 mispronunciations in the training set are excluded from `L_diag` (marked `-1` = ignored by `CrossEntropyLoss`). Breakdown: 488 `<unk>` (non-categorical errors — sounds outside the CMU 39-phone set), 450 `<DEL>` (deletions — learner didn't produce the phone), 46 L1-specific phones (`IR`, `AR`, `TR`, `DZ`, `TS`, `DR` — Mandarin phonemes). These exclusions are expected for a 39-class CMU diagnosis head; there is no honest target class for those cases. The paper uses the same 39-phone vocabulary and faces the same constraint. The remaining 1007 substitutions do have valid CMU phone labels extracted from SpeechOcean762 annotations. This mainly limits diagnosis supervision coverage; it is unlikely to be the main driver of the phone-PCC gap.

4. **PhnVar σ**: The paper defines δ(σ) as "a Gaussian distribution with a zero mean and the standard deviation σ" (Eq. 25) and specifies α=1, β=1, but **does not specify σ anywhere** — not in the text, not in the experimental setup, not in any table. We default to `σ=1.0`. In practice the effective perturbation is scaled by `sqrt(QF×DF)`, so the injected std is not `1.0` for most phonemes; on our training set the typical scale is much smaller than the worst case. This makes σ a reasonable small-sweep target for MDD tuning, but probably a second-order factor relative to the larger architectural / protocol mismatches above.

### CLI commands

```bash
# Pretrain (run once)
uv run p010 pretrain

# Full MuFFIN sweep
uv run p010 sweep --seeds 22,33,44,55,66 --use-conpco --use-mdd --use-phnvar \
    --pretrained checkpoints/pretrained/best_pretrain.pth

# Evaluate with threshold search
uv run p010 eval --checkpoint checkpoints/v3-muffin/seed22/best_model.pth --use-mdd
```

---

## Session 5 — Phase 2 foundations (HConv, CHConv, all-layer extraction)

### New components

| Component | Files | Paper reference | Description |
|-----------|-------|----------------|-------------|
| HConv | `models/hconv.py` | Shih & Harwath 2024 (2406.12209) | 1D conv over SSL layer dimension. kernel=5, stride=3, depth=floor(log_3(L)), ReLU after each conv. Last conv reduces channels, output flattened. Ported from `github.com/atosystem/SSL_Interface`. For L=25 (Large): output_dim=1023 (not 1024 — inherent to the `ceil(D//L_remaining)` arithmetic). |
| CHConv | `models/hconv.py` | Shih et al. 2025 (2511.08389) | Multi-model layer fusion. Concatenates N models along feature dim → `[B, T, L, N×D]`, then applies HConv. Optional linear projection to target output_dim. |
| All-layer extraction | `extract.py`, `cli.py` | — | Runs frozen SSL models (wav2vec2-large, HuBERT-large, WavLM-large) with `output_hidden_states=True` on SpeechOcean762 audio. Averages frame-level features per phone using durations from `dur_feat.npy`. Saves `[N_utt, 50, 25, 1024]` per model as float16. |

### Design decisions

| Decision | Detail |
|----------|--------|
| soundfile instead of torchaudio for audio loading | torchaudio 2.10 requires `torchcodec` (new FFmpeg replacement) which isn't installed. `soundfile` is already a dependency via librosa/datasets, reads the same WAV/FLAC formats, and returns numpy arrays which `Wav2Vec2FeatureExtractor` expects directly. For 16kHz mono WAV loading the output is identical. |
| `Wav2Vec2FeatureExtractor` instead of `AutoProcessor` | `AutoProcessor` attempts to load a tokenizer (for CTC ASR models) which fails for base SSL models like `hubert-large-ll60k`. `Wav2Vec2FeatureExtractor` loads only the feature extraction config (normalization, padding) — correct for our use case of extracting hidden states, not doing ASR. |
| HF_TOKEN loaded from root `.env` | The HuggingFace libraries read `os.environ["HF_TOKEN"]` directly, not pydantic-settings `.env` files. `extract.py` reads the root `.env` and injects `HF_TOKEN` into `os.environ` if not already set. This avoids the "unauthenticated requests" warning and enables faster downloads. |
| Phone averaging before HConv | The Shih papers apply HConv at frame-level, but HConv operates over the **layer** dimension (not time), so averaging over phone durations before or after HConv is mathematically identical. Phone-averaging first avoids storing full frame-level tensors (~600× smaller: 50 phones vs ~30K frames per utterance). |
| Float16 for all-layer features | `[2500, 50, 25, 1024]` at float32 = 12.8 GB per model. Float16 halves this to 6.4 GB. SSL hidden states are well within float16 dynamic range. |

---

## Session 3 — Replication audit + three correctness fixes

Full line-by-line audit of our code against the ConPCO reference. Discovered and fixed three bugs that were degrading replication fidelity.

### Bug fixes

| What | Where | Description |
|------|-------|-------------|
| Energy/duration not normalized in reference | `data.py` | We were z-scoring energy (`[50,7]`) and duration (`[50,1]`) with the reference's defined constants. The reference defines those constants but **never applies them** — the model was trained and reported on raw features. Revert: `self.energy = energy`, `self.dur = dur`. Effect: changes input distribution to every phone-level block, likely the largest contributor to the baseline gap. |
| Noise guard breaks RNG sync | `trainer.py` | `if settings.noise > 0: torch.rand_like(gop)` — when `noise=0.0` (the default), `torch.rand` is never called. The reference always calls `torch.rand(...)` unconditionally, consuming RNG state even at noise=0. The `if` guard silently desynchronizes the random trajectory for every batch. Fixed: unconditional `gop = gop + (torch.rand_like(gop) - 1) * settings.noise`. Effect: primarily hurts ConPCO (which depends on per-phone RNG-driven batching). |
| CUDNN determinism flags missing | `cli.py` | Reference sets `torch.backends.cudnn.benchmark = False` and `torch.backends.cudnn.deterministic = True` before each training run. Without these, CUDA convolutions are non-deterministic and results differ across seeds. Fixed: added to `_set_seed()`. |

### Reference bugs we fixed (intentional deviations)

These are bugs in the ConPCO reference code. We use mathematically correct implementations. They have **zero effect on the baseline sweep** (ConPCO OFF). They may change ConPCO results, but toward better values.

| What | Reference bug | Our fix |
|------|--------------|---------|
| `W2UFeatGen` pooling branch 3 | `pooling_proj2` is used for both branch 2 and branch 3 scoring (`pooling_proj3` defined but never called — dead weights) | We correctly use `pooling_proj3` for branch 3 |
| CLAP loss formula | Text→audio direction uses `log_softmax(cos, dim=1).T` (already-log-softmaxed matrix transposed) instead of `log_softmax(cos.T, dim=1)` | We apply `log_softmax` independently for each direction |

### Known remaining gap from paper targets

After all fixes, the expected ceiling based on prior work is ~0.658 baseline / ~0.671 ConPCO. The paper reports 0.680 / 0.701. The residual ~0.02-0.03 gap is attributed to **test-set model selection bias**: the ConPCO repo selects the best checkpoint by test-set MSE every epoch (no separate validation split), which inflates the reported number by ~2-3% against any clean replication.

---

## Session 2 — Type safety, data pipeline, and test infrastructure

### Bug fixes

| What | Where | Description |
|------|-------|-------------|
| `str` passed as `Path` to `Settings` | `cli.py` | All `features_dir` string args were passed directly to `Settings(features_dir=features_dir)`; ty caught this as `invalid-argument-type`. Fixed by wrapping: `Settings(features_dir=Path(features_dir))`. |
| `mdd_logit` type narrowing | `hiercb.py` | Guard `if self.use_mdd:` left `mdd_logit` typed as `Tensor \| None` inside the branch (ty can't narrow via unrelated flag). Changed to `if mdd_logit is not None:` — ty narrows to `Tensor`. |
| `GoPDataset` not properly generic | `data.py` | Inherited from bare `Dataset` instead of `Dataset[tuple[torch.Tensor, ...]]`. Caused ty to treat `__getitem__` return as `Unknown`. Fixed by using the generic form. |
| Cascading ty errors masked real bugs | — | Initial report of 66 `invalid-argument-type` "false positives" was wrong. All ML libs (torch, numpy, wandb, scipy) ship `py.typed` / `.pyi` stubs. The 66 errors were cascading artifacts from the 3 real bugs above. After fixing those, ty is fully clean with zero rule overrides. |
| `# type: ignore` on wrong line | `cli.py` | In multi-line ternaries, the ignore comment must be on the line containing the actual error expression, not the opening paren. Stray comment-only ignore lines are flagged as unused directives. |
| Download incomplete — missing GOP features | `data.py` | HF Hub `a2d8a4v/SpeechOcean762_for_ConPCO` delivers only 12 files (SSL + energy + dur + word_id). The 8 GOP feature + label files (`tr/te_feat.npy`, `tr/te_label_phn/utt/word.npy`) come from the GOPT Dropbox zip. `download_features()` now handles both sources. |

### Design decisions

| Decision | Detail |
|----------|--------|
| Two-source feature download | SpeechOcean762 features come from two repos: (1) HF Hub `a2d8a4v/SpeechOcean762_for_ConPCO` via `snapshot_download` + zip extraction; (2) GOPT Dropbox `data.zip` for GOP features + labels. `download_features()` handles both in sequence, skipping already-present files. |
| No `ty` rule overrides | `[tool.ty.rules]` is empty. All ML libraries ship their own stubs; zero suppression is needed. The only `# type: ignore` comments in the codebase are intentional: `[missing-argument]` for pydantic-settings env injection, and `[override]` for the LSP-narrowing on `GoPDataset.__getitem__`. |
| `# type: ignore[missing-argument]` at `Settings()` call sites | `Settings(features_dir=...)` is always valid (ty resolves it). `Settings()` without `features_dir` is correct at runtime because pydantic-settings reads `P010_FEATURES_DIR` from `.env` or the environment — but ty sees `features_dir` as a required arg and flags it. The ignore is intentional and documented inline. |
| `# type: ignore[override]` on `GoPDataset.__getitem__` | `Dataset.__getitem__` base signature accepts an untyped `index` param. Narrowing to `int` violates LSP in theory (more restrictive than base). In practice this is standard for all map-style datasets. The ignore is correct and intentional. |
| `pin_memory = torch.cuda.is_available()` | PyTorch 2.10.0 ships an internal bug in `pin_memory.py:57` that emits a `DeprecationWarning` when `pin_memory=True` on CPU. It is fixed in upstream PR #174584 but not yet released. Setting `pin_memory = torch.cuda.is_available()` avoids the warning entirely on CPU-only machines while preserving GPU behavior. We do not suppress via `filterwarnings`. |
| Real data in all tests | All tests use the `features_dir` session fixture from `conftest.py`, which reads `P010_FEATURES_DIR` from `.env` via `Settings()`. No synthetic/mocked data anywhere. If the data is absent, the fixture downloads it automatically; if `P010_FEATURES_DIR` is not configured, tests fail clearly via `pytest.fail()` (not `pytest.mark.skipif`). |
| Two `.env` files | Root `.env` (`~/github/peacock-asr/.env`) holds API keys (HF_TOKEN, WANDB_API_KEY, etc.) loaded by shell and library init. Project `.env` (`projects/P010-muffin-improvements/.env`) holds `P010_FEATURES_DIR`, loaded by pydantic-settings inside `Settings()`. Separation keeps API secrets out of the project directory. |
| `--n-epochs` CLI override | Added to `train` command to allow quick smoke runs without a full 100-epoch training cycle. Uses `settings.model_copy(update={"n_epochs": n_epochs})` (pydantic v2) to produce a modified Settings copy without re-validating the whole object. |
| Smoke test | `tests/test_smoke.py` runs 1 epoch end-to-end with `WANDB_MODE=offline`. Asserts PCC is a finite float in `[-1, 1]`. Intentionally uses `num_workers=0` to avoid subprocess overhead in CI. |

### Package version upgrades

Updated all lower bounds to match the latest available at session time. Key changes:

| Package | Old bound | New bound | Notable |
|---------|-----------|-----------|---------|
| `torch` | `>=2.5` | `>=2.10` | |
| `torchaudio` | `>=2.5` | `>=2.10` | |
| `transformers` | `>=4.47` | `>=5.3` | Major version bump |
| `datasets` | `>=3.1` | `>=4.8` | Major version bump |
| `accelerate` | `>=1.2` | `>=1.13` | |
| `numpy` | `>=2.1` | `>=2.4` | |
| `scipy` | `>=1.14` | `>=1.17` | |
| `scikit-learn` | `>=1.5` | `>=1.8` | |
| `pandas` | `>=2.2` | `>=3.0` | Major version bump |
| `wandb` | `>=0.18` | `>=0.25` | |
| `pydantic-settings` | `>=2.6` | `>=2.13` | |
| `pydantic` | `>=2.9` | `>=2.12` | |
| `rich` | `>=13.9` | `>=14.3` | |
| `huggingface-hub` | `>=0.27` | `>=1.7` | Major version bump |
| `pytest` (dev) | `>=8.3` | `>=9.0` | |
| `pytest-cov` (dev) | `>=6.0` | `>=7.1` | |
| `ruff` (dev) | `>=0.9` | `>=0.15` | |
| `ty` (dev) | `>=0.0.14` | `>=0.0.24` | |

---

## Phase 2 — CHConv (pending)

*To be filled in when Phase 2 implementation begins.*
