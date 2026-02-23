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

---

## Benchmark Run 1 Results (scalar GOP + polynomial regression)

All three backends on SpeechOcean762 test split, using GOP-SF-SD-Norm
with per-phone polynomial regression (order 2) trained on the train split.

| Backend | PCC | 95% CI | MSE |
|---------|-----|--------|-----|
| xlsr-espeak | 0.3197 | [0.3115, 0.3278] | 0.6656 |
| original | 0.3104 | [0.3022, 0.3185] | 0.6872 |
| zipa | 0.0668 | [0.0573, 0.0762] | 0.9291 |

**Context**: The CTC-based-GOP paper (Table III) reports TGOP-TDNN baseline
PCC of 0.361 ± 0.008 for scalar GOP + polynomial regression. Our original
(0.31) is in the right ballpark. Higher numbers (0.58-0.65) require GOP
feature vectors + SVR/GOPT transformer, which is the next pipeline stage.

**ZIPA 0.07**: See "ZIPA Character-Level Vocab Incompatibility" below.

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

---

## Benchmark Run 2 Results (ZIPA with ER/G unicode fixes)

ZIPA only, with two fixes applied:
- ER: U+025D → U+025C (correct IPA character now in vocab)
- G: U+0261 → ASCII 'g' (correct character now in vocab)
- 7 diphthong/affricate phones excluded (32/39 phones scored)

| Backend | PCC | 95% CI | MSE | Phones scored |
|---------|-----|--------|-----|---------------|
| zipa (run 1) | 0.0668 | [0.0573, 0.0762] | 0.9291 | ~50K (with broken ER/G) |
| zipa (run 2) | 0.0749 | [0.0656, 0.0842] | 0.9253 | 43,827 (32/39 phones) |

**Marginal improvement only.** The unicode fixes helped slightly but the
fundamental issue is not the mapping — ZIPA's character-level posteriors
are not calibrated for pronunciation scoring.

**Per-phone PCC (run 2, sorted):**
| Phone | PCC | Phone | PCC |
|-------|-----|-------|-----|
| NG | 0.4242 | R | 0.0840 |
| ZH | 0.3391 | ER | 0.0801 |
| V | 0.2718 | EH | 0.0563 |
| S | 0.2647 | Z | 0.0512 |
| F | 0.2070 | T | 0.0432 |
| P | 0.1615 | AE | 0.0408 |
| G | 0.1607 | B | 0.0386 |
| TH | 0.1391 | UW | 0.0369 |
| M | 0.1269 | AO | 0.0354 |
| D | 0.1092 | UH | 0.0332 |
| Y | 0.0869 | L | 0.0266 |
| W | 0.0868 | DH | 0.0201 |
| K | 0.0848 | N | 0.0169 |
| | | SH | 0.0081 |
| | | IH | 0.0068 |
| | | IY | -0.0125 |
| | | AH | -0.0232 |
| | | AA | -0.0287 |

**Analysis**: A few phones show moderate correlation (NG 0.42, ZH 0.34,
V/S/F ~0.20-0.27) but most are near zero or negative. Three phones (IY, AH,
AA) have negative PCC, meaning the model's confidence is inversely correlated
with human scores — worse pronunciation actually gets higher posterior
probability for these phones.

**Interpretation**: ZIPA's posteriors reflect transcription confidence ("did I
hear this sound?"), not pronunciation quality ("how well was this sound
produced?"). These are related but distinct signals. A strongly accented but
recognizable phone might get high transcription confidence but low pronunciation
score. The model was trained on multilingual ASR data (17K hours across 88
languages), so it learned to be tolerant of phonetic variation — the opposite
of what pronunciation scoring needs.

**Conclusion**: ZIPA cannot be used for GOP-based pronunciation scoring
without fine-tuning. The pretrained encoder captures useful acoustic
information (evidenced by NG and ZH showing real correlation), but the CTC
head is calibrated for recognition, not assessment. Fine-tuning a new
phoneme-level head on pronunciation-scored data would be the path forward.

---

## All Results Summary

| Backend | Vocab | PCC | 95% CI | MSE | Phones |
|---------|-------|-----|--------|-----|--------|
| xlsr-espeak | 387 IPA phonemes | **0.3197** | [0.3115, 0.3278] | 0.6656 | 39/39 |
| original | 39 ARPABET | 0.3104 | [0.3022, 0.3185] | 0.6872 | 39/39 |
| zipa | 127 IPA chars | 0.0749 | [0.0656, 0.0842] | 0.9253 | 32/39 |

**Winner**: xlsr-espeak (wav2vec2-xlsr-53-espeak-cv-ft) edges out the original
checkpoint-8000 by a small margin. Both are in the expected range for scalar
GOP + polynomial regression (paper baseline: 0.361). ZIPA is not viable
without fine-tuning.
