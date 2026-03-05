# Track 07 Evidence Ledger: Training from Scratch

Scope:

- From-scratch Conformer/Zipformer phoneme CTC training
- Comparison with fine-tuned SSL encoders (Track 05)
- Data requirements and compute cost analysis

Citation policy:

- Use numbered citations in text: `[1]`, `[2]`, ...
- Use `./refs.bib` as canonical bib source.

---

## 1. Claim Map

| ID | Claim | Evidence Status | Primary Citations |
|----|-------|----------------|------------------|
| C1 | ZIPA (Zipformer from scratch on 17K hours) achieves 2.71 PFER on seen languages | Paper result | [1] |
| C2 | ZIPA's 127-char IPA vocabulary breaks GOP scoring (diphthongs as single tokens) | Architectural analysis | [1], internal |
| C3 | PRiSM finds encoder-CTC is most stable approach for phone recognition | Paper result | [3] |
| C4 | POWSM (multi-task Whisper-style 350M) outperforms ZIPA and wav2vec2-phoneme | Paper result | [2] |
| C5 | icefall TIMIT TDNN-LSTM achieves 17.66% PER | Paper result | [7] |
| C6 | Conformer is the standard modern ASR architecture (CNN+Transformer hybrid) | Widely established | [4] |
| C7 | From-scratch training needs 1K+ hours minimum for competitive phone posteriors | Estimate from ZIPA data scale | [1], [2] |
| C8 | Fine-tuned w2v-BERT 2.0 achieves PCC 0.648 on SpeechOcean762 GOP scoring | Internal result (Track 05) | Internal |
| C9 | LibriSpeech 960h phone alignments available (gilkeyio/librispeech-alignments) | Dataset exists | [6] |
| C10 | Zipformer outperforms Conformer in icefall benchmarks | icefall documentation | [7] |

---

## 2. External Reference Papers

| # | Paper | Key Result | Status |
|---|-------|-----------|--------|
| [1] | Zhu et al. (2025) "ZIPA: Efficient Multilingual Phone Recognition" | Zipformer on 17K hours, 2.71 PFER | PDF in papers/, code at github.com/lingjzhu/zipa |
| [2] | Bigi et al. (2025) "POWSM: Phonetic Open Whisper-Style Speech Foundation Model" | 350M multi-task, beats ZIPA on phone recognition | PDF in papers/ |
| [3] | (2025) "PRiSM: Benchmarking Phone Realization in Speech Models" | encoder-CTC most stable for phone recognition | PDF in papers/ |
| [4] | Gulati et al. (2020) "Conformer: Convolution-augmented Transformer for Speech Recognition" | CNN+Transformer hybrid, standard ASR architecture | PDF in papers/ |
| [5] | Conneau et al. (2021) "Simple and Effective Zero-Shot Cross-Lingual Phoneme Recognition" | wav2vec2 zero-shot phoneme transfer | PDF in papers/ |
| [6] | Pratap et al. (2023) "Scaling Speech Technology to 1000+ Languages" (MMS) | Massive multilingual phone recognition | PDF in papers/ |
| [7] | k2-fsa/icefall | TIMIT TDNN-LSTM 17.66% PER, Zipformer LibriSpeech recipes | github.com/k2-fsa/icefall |
| [8] | Yao et al. (2025) "Towards Accurate Phonetic Error Detection Through Phoneme Similarity Modeling" | Phoneme similarity for error detection | PDF in papers/ (2507.14346) |
| [9] | (2025) "LCS-CTC: Leveraging Soft Alignments" | CTC variant with soft alignments | PDF in papers/ (2508.03937) |
| [10] | Gutkin et al. (2022) "ByT5 model for massively multilingual G2P" | G2P across 100 languages, needed for data labeling | PDF in papers/ |

---

## 3. Internal Evidence Anchors

- Track 05 best result (w2v-BERT 2.0 fine-tuned, PCC 0.648): `runs/` directory
- GOP feature extraction pipeline: `/home/simon/github/peacock-asr/src/peacock_asr/gop.py`
- CTC posterior extraction: `/home/simon/github/peacock-asr/src/peacock_asr/ctc_gop_original.py`
- GOPT scorer: `/home/simon/github/peacock-asr/src/peacock_asr/gopt_model.py`
- LibriSpeech alignments: gilkeyio/librispeech-alignments on HuggingFace (960h, ARPABET)

---

## 4. Key Technical Details

### ZIPA Architecture

- Model: Zipformer (k2-fsa variant of Conformer)
- Training data: IPAPack++ 17K hours, 88 languages
- Output vocabulary: 127 IPA characters
- PFER: 2.71 on seen languages, 0.66 on English
- Critical limitation: IPA character-level vocabulary means diphthongs (AW, AY, EY, OW, OY)
  and affricates (CH, JH) are split into multiple tokens, breaking ARPABET-based GOP scoring

### POWSM Architecture

- Model: Whisper-style encoder-decoder, 350M parameters
- Training: multi-task (phone recognition + ASR + G2P + P2G)
- Framework: ESPnet
- Result: outperforms ZIPA and wav2vec2-phoneme on phone recognition tasks

### PRiSM Benchmark Findings

- Compared encoder-CTC, encoder-decoder, and other approaches for phone recognition
- Finding: encoder-CTC is the most stable architecture for phone-level output
- This supports using CTC (not seq2seq) for our GOP posterior extraction

### Conformer Architecture (Gulati 2020)

- Combines CNN and Multi-Head Self-Attention in each block
- CNN captures local acoustic patterns; MHSA captures global context
- Standard modern ASR architecture, basis for Zipformer and FastConformer
- icefall has Conformer LibriSpeech recipes but only BPE/word-level output

### Data Labeling Pipeline

For large unlabeled datasets, G2P labeling is needed:

1. LibriSpeech 960h: pre-labeled ARPABET via forced alignment (gilkeyio/librispeech-alignments)
2. CommonVoice: text available, needs G2P (ByT5-based multilingual G2P [10])
3. IPAPack++: ZIPA's dataset, G2P-labeled via CharsiuG2P — IPA output, would need ARPABET conversion

### GOP Posterior Format Requirement

The GOP pipeline (`gop.py`, `ctc_gop_original.py`) expects:

- Frame-level log-probabilities from a CTC output layer
- Shape: `(T, num_phones)` where `num_phones` is ARPABET vocabulary size (~40)
- No beam search or decoding — raw CTC output layer activations
- Any from-scratch model must produce this format for the pipeline to work

### Compute Estimates (rough)

| Training | Data | Estimated GPU-hours (RTX 5070 12GB) |
|---------|------|-------------------------------------|
| TIMIT TDNN-LSTM | 5h | ~1-2 hours |
| Conformer-S on LS-100h | 100h | ~8-24 hours |
| Conformer-M on LS-960h | 960h | ~3-7 days |
| Zipformer-M on LS-960h | 960h | ~3-5 days |

Note: These are rough estimates. Actual compute depends on batch size, gradient accumulation,
and whether the RTX 5070 can fit the model in 12GB VRAM. Conformer-M may need smaller batch
sizes than larger GPU configurations in the literature.

---

## 5. Open Questions

- Q1: Can Conformer-M fit in 12GB VRAM at a practical batch size?
- Q2: Is gilkeyio/librispeech-alignments the best source for ARPABET labels, or should we
  use a G2P + forced alignment pipeline?
- Q3: What is the minimum PER threshold that predicts useful GOP posteriors for scoring?
- Q4: Does icefall's k2 graph compilation work with ARPABET vs BPE vocabularies without
  significant recipe changes?
- Q5: Can we reuse ZIPA's trained Zipformer encoder with a new ARPABET head
  (hybrid approach) to avoid full from-scratch training?
