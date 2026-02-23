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

- `torchcodec` — required by datasets 4.x even though we don't use it for decoding
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
