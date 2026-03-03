# Architecture & Design Decisions

## Audio Decoding: soundfile over torchcodec

**Decision**: Disable HuggingFace datasets 4.x auto-decoding (torchcodec) and decode audio manually with `soundfile`.

**Why**: datasets 4.x switched from soundfile to torchcodec as the default audio decoder. torchcodec decodes audio files one-at-a-time through FFmpeg, which is extremely slow for our use case (5000 small wav files). Parsing the full SpeechOcean762 dataset took 15+ minutes with torchcodec vs ~30 seconds with soundfile.

**How**: `dataset.cast_column("audio", Audio(decode=False))` disables auto-decoding. We then use `soundfile.read(io.BytesIO(raw_bytes))` to decode each audio file manually.

**Sources**:

- [HuggingFace datasets audio.py](https://github.com/huggingface/datasets/blob/main/src/datasets/features/audio.py) — confirms torchcodec is hardcoded, no backend switch
- [HuggingFace audio loading docs](https://huggingface.co/docs/datasets/en/audio_load) — `decode=False` is the official way to skip auto-decoding
- CTC-based-GOP reference code uses `.map(batched=True, batch_size=100)` to avoid per-sample overhead
- ZIPA reference code uses `soundfile.read()` directly

**Alternative considered**: `dataset.decode(num_threads=N)` for multithreaded torchcodec — but this only works on `IterableDataset` (streaming), not regular `Dataset`.

---

## GOP Algorithm: Ported from CTC-based-GOP

**Decision**: Port the GOP-SF-SD-Norm algorithm directly from `CTC-based-GOP/taslpro26/gop_sf_sd_norm.py`.

**Why**: This is the exact algorithm from the paper "Segmentation-Free Goodness of Pronunciation" (IEEE TASLP 2026). It computes GOP scores using CTC forward-backward without forced alignment.

**Source**: `CTC-based-GOP/taslpro26/gop_sf_sd_norm.py` — functions `ctc_loss`, `ctc_loss_denom`, `single_process`

**Changes from original**:

- Split `ctc_loss_denom` into 5 helper functions (`_step_denom`, `_step_blank`, `_step_normal`, `_step_after_arb`, `_step_arb`) to satisfy ruff PLR0912/PLR0915 complexity limits
- No algorithmic changes

---

## Evaluation Protocol: Per-phone polynomial regression + PCC

**Decision**: Use the same evaluation protocol as the CTC-based-GOP paper.

**Source**: `CTC-based-GOP/is24/evaluation/spo762/evaluate_gop_scalar.py`

**Protocol**:

1. Train a per-phone polynomial regression (order 2) on train split
2. Use balanced sampling to oversample minority score classes
3. Predict on test split, round to nearest integer
4. Compute Pearson Correlation Coefficient (PCC) with 95% CI

---

## Three Backends

### 1. original (checkpoint-8000)

- wav2vec2-xlsr-53 fine-tuned on LibriSpeech with 39 ARPABET phones
- Loaded from local CTC-based-GOP repo files
- This is the paper's own model, used as the baseline

### 2. xlsr-espeak

- `facebook/wav2vec2-xlsr-53-espeak-cv-ft` from HuggingFace (387 IPA tokens)
- Same base model (xlsr-53) but fine-tuned on CommonVoice with IPA phones
- Requires ARPABET-to-IPA mapping to work with SpeechOcean762
- Requires `espeak-ng` system package and `phonemizer` Python package

### 3. zipa

- ZIPA-CR large model via ONNX Runtime (127 IPA tokens)
- `anyspeech/zipa-large-crctc-ns-800k` from HuggingFace
- Takes 80-dim FBank features (via lhotse), NOT raw audio
- Requires `onnxruntime` and `lhotse` packages
- Source: ZIPA repo `inference/inference.py` and `inference/utils.py`

---

## ZIPA FBank Features

**Decision**: Use lhotse `Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))` for feature extraction.

**Why**: The ZIPA ONNX model expects `[N, T, 80]` FBank input (not raw audio). This is confirmed by:

- ONNX model input: `x` shape `['N', 'T', 80]`, type `tensor(float)`
- ZIPA `inference/utils.py`: `FbankConfig(num_filters=80, dither=0.0, snip_edges=False)`
- ZIPA `inference/inference.py`: `extractor.extract_batch([audio_tensor], sampling_rate=16000)`

---

## ZIPA Blank Token

**Decision**: Use `<blk>` (index 0) as the CTC blank token for ZIPA.

**Why**: The ZIPA `tokens.txt` file has `<blk> 0` as the first entry. Our code originally looked for `<blank>` which doesn't exist in the ZIPA vocab.

---

## Dataset: SpeechOcean762

**Decision**: Pin to revision `f95618ea1353303f34cf186b9c310fa2c1eb02c8`.

**Why**: Reproducibility. This is the revision cached from our first successful download.

**Source**: `mispeech/speechocean762` on HuggingFace. 5000 utterances (2500 train, 2500 test), phone-level ARPABET transcriptions with accuracy scores (0-2).

---

## ARPABET Stress Stripping

**Decision**: Strip stress digits from ARPABET phones (e.g., `AA1` -> `AA`).

**Why**: SpeechOcean762 uses stressed ARPABET (AA0, AA1, AA2) but the CTC models use plain ARPABET (AA). The evaluation protocol in the paper also strips stress.

---

## Dependencies

- `torchcodec` — intentionally **not** required; we bypass datasets auto-decode with `Audio(decode=False)` and decode via `soundfile`
- `protobuf` — required by HuggingFace tokenizers at runtime
- `phonemizer` + `espeak-ng` (system) — required by xlsr-espeak tokenizer
- `lhotse` — required for ZIPA FBank extraction
- `onnxruntime` — optional, only for ZIPA backend
- `soundfile` — used for fast audio decoding (replaces torchcodec in practice)

---

## noqa Comments

Intentional lint suppressions:

- `PLC0415` — lazy imports of heavy libraries (torch, transformers, datasets) for fast CLI startup
- `T201` — print statements are intentional (CLI tool output)
- `FBT001` — boolean `verbose` param is standard CLI pattern
- `BLE001` — broad exception catch in `compare` command (intentional, catches any backend failure)
- `PLR0913` — GOP helper functions need many params due to algorithm structure
- `TC002` — numpy needed at runtime (not just type-checking) for `torch.from_numpy`
- `ARG001` — `big_p` param kept for clarity even if unused in some helpers

---

## ZIPA Character-Level Vocab Incompatibility

**Problem**: ZIPA scored PCC 0.0668 (near zero) on Run 1.

**Root cause**: ZIPA uses a 127-token character-level IPA vocab — each token
is a single IPA character, not a phoneme. This causes two issues:

1. **Unicode bugs** (fixable): ER mapped to U+025D (not in vocab), should be
   U+025C. G mapped to U+0261 (not in vocab), should be ASCII 'g'.

2. **Structural incompatibility** (not fixable without fine-tuning): 7/39
   ARPABET phones are diphthongs/affricates that map to multi-character IPA
   strings (e.g., AW → aʊ, CH → tʃ). ZIPA has no single token for these —
   they get dropped from GOP scoring.

**Source evidence**:

- ZIPA `ipa_simplified/tokens.txt`: 127 entries, each is one IPA character
- ZIPA `scripts/evaluate.py`: evaluation uses phonetic feature edit distance
  (panphon), not per-phoneme scoring — confirms ZIPA is an ASR model, not
  designed for phoneme-level pronunciation assessment
- CTC-based-GOP `evaluate_gop_scalar.py` line 44-51: enforces strict 1:1
  phone-to-score mapping with `len(label_phoneme) != len(seq_score)` assertion
- CTC-based-GOP GOP algorithm: each phone = exactly 1 vocab index, no
  multi-token support

**Decision**: Fix the unicode bugs (ER, G), accept that 7 diphthong/affricate
phones are dropped. Run 2 will benchmark ZIPA on the 32/39 phones that do
map to single IPA characters. This gives a real measurement of ZIPA's
acoustic quality on the phones it can handle.

**Unmappable phones**: AW, AY, CH, EY, JH, OW, OY

**Future**: Adding a phoneme-level CTC head on top of ZIPA (fine-tuning a
small projection layer: 127 char logits → 39 phoneme logits) would resolve
this. That's the approach planned for Options 4-5.

**Conclusion**: ZIPA cannot be used for GOP-based pronunciation scoring
without fine-tuning. See [EXPERIMENTS.md](EXPERIMENTS.md) for full run results.

---

## Experiment Ops Stack (2026-03-02)

**Decision**: Keep MLflow as the experiment tracker and use Hugging Face Hub for checkpoint hosting.

**Why**: These solve different problems cleanly:

1. MLflow handles run metadata/metrics/comparison.
2. HF Hub handles model artifact versioning and distribution.

**Decision**: Add a YAML-driven batch orchestration layer in-repo (instead of ad-hoc shell loops) as the next execution path.

**Why**: Reproducibility and stability. Shell one-offs have already caused avoidable run failures (shell-specific behavior).

**Execution environment**:

1. Local machine: dev iteration and smaller/full eval runs.
2. RunPod: heavier training runs, using the same repo-runner path.

---

## RunPod Training Stabilization (2026-03-03)

**Context**: `peacock-asr train-preflight` repeatedly failed on RunPod L4 before first stable completion.

**Root causes found**:

1. `torch` runtime mismatch produced CUDA GEMM failures in wav2vec2-bert attention:
   - `RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE`.
   - **Correction**: These CUBLAS crashes were caused by FP16 (5-bit exponent) overflow on
     wav2vec2-bert activations, not by missing BF16 hardware support. The original fix
     incorrectly assumed L4 lacked BF16 and fell back to FP32.
2. HF datasets audio auto-decoding path invoked `torchcodec`, which failed on the pod (ABI/FFmpeg issues) and blocked dataset loading.
3. A prior run ended with exit code 143 (`SIGTERM`) due process interruption, not model logic.

**Decisions/Fixes**:

1. Pin `torch==2.8.0` for this training stack.
2. Remove `torchcodec` dependency and avoid all auto-decode paths in training data loading:
   - `Audio(decode=False)` for streaming and non-streaming splits.
   - Decode audio manually with `soundfile`.
3. **Use `torch.cuda.is_bf16_supported()` for BF16 detection** instead of a GPU name allowlist.
   The NVIDIA L4 (Ada Lovelace, 4th-gen Tensor Cores) has native BF16 support at 242 TFLOPS
   ([L4 datasheet](https://www.nvidia.com/en-us/data-center/l4/)). The previous allowlist
   (`A100`, `H100`, `H200`, `B100`, `B200`) incorrectly excluded L4, forcing FP32 and wasting
   ~50% compute. The runtime query covers all current and future GPU architectures correctly.
4. Re-enable gradient checkpointing (was disabled as part of the L4 stability investigation;
   safe to restore now that BF16 is used instead of FP16).
5. Keep HF dataset loading cache under project settings cache dir (`settings.data_dir / "hf-datasets"`).

**Validation result**:

- RunPod `train-preflight` completed end-to-end (Run ID `3aa093bd111e45aeba9681451c986595`), with final `Done.` in `logs/train-preflight.log`.

**Notes on expected non-fatal messages**:

- `MISSING/UNEXPECTED` model keys from `from_pretrained(..., add_adapter=True, vocab_size=...)` are expected for newly initialized adapter/CTC head params.
- MLflow message `Skip logging GPU metrics` is informational in current server/runtime setup; run-level metrics still log.
