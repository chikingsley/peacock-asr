# Track 10 Evidence Ledger: Compact Backbones

Scope:

- Backbone size vs pronunciation scoring quality
- wav2vec2-base, HuBERT-base, Citrinet-256 as GOP backbone alternatives
- HMamba, HiPAMA as alternative scoring heads

Citation policy:

- Use numbered citations in text: `[1]`, `[2]`, ...
- Use `./refs.bib` as canonical bib source.

---

## 1. Claim Map

| ID | Claim | Evidence Status | Primary Citations |
|---|---|---|---|
| C1 | wav2vec2-base (95M) can serve as GOP backbone with acceptable PCC | Needs experiment (Phase 1B) | [1] |
| C2 | HuBERT-base (95M) matches or exceeds wav2vec2-base for GOP | Needs experiment (Phase 1C) | [2], [3] |
| C3 | 95M backbone loses < 5% PCC vs 300M xlsr-53 | Hypothesis, needs P1 | [1], [2] |
| C4 | Citrinet-256 (10M) can be adapted for phoneme-level GOP | Needs experiment (Phase 2) | [4] |
| C5 | Scoring head choice (GOPT vs HMamba) matters less than backbone | Hypothesis, needs P3 | [5], [6] |
| C6 | No published paper tests wav2vec2-base or HuBERT-base as GOP backbone | Supported by literature search | [3] |
| C7 | Our xlsr-53 + GOPT baseline (0.677) already exceeds HIA SOTA (0.657) | Supported (Track 05 data vs [7]) | Internal, [7] |

---

## 2. External Reference Papers

| # | Paper | Key Result | Status |
|---|---|---|---|
| [1] | Baevski et al. (NeurIPS 2020) wav2vec 2.0 | Base model: 95M, Large: 317M | Architecture reference |
| [2] | Hsu et al. (IEEE TASLP 2021) HuBERT | Base: 95M, same arch as wav2vec2 | Architecture reference |
| [3] | Kim et al. (Interspeech 2022) SSL pronunciation | wav2vec2-base UTT PCC 0.65, HuBERT-base 0.75 (SSL-only, not GOP) | Key prior work |
| [4] | Li et al. (ASRU 2021) Citrinet | 10M params, SentencePiece vocab, WER 3.8% | Backbone candidate |
| [5] | Chao & Chen (NAACL 2025) HMamba | Mamba scoring head, hierarchical, arXiv 2502.07575 | Scorer alternative |
| [6] | Do et al. (ICASSP 2023) HiPAMA | Multi-aspect attention, ~32K head params | Scorer alternative |
| [7] | Han et al. (AAAI 2026) HIA | Phone PCC 0.657 (best published GOP-Kaldi) | Comparison target |
| [8] | Gong et al. (ICASSP 2022) GOPT | Phone PCC 0.612 with Kaldi GOP | Our scorer baseline |
| [9] | Cao et al. (TASLP 2026) GOP-SF | GOP-SF algorithm (our feature extractor) | Feature source |

---

## 3. Internal Evidence Anchors

- GOPT baseline runs: Track 05 Phase 1 (`runs/2026-03-03_001037_track05_phase1_baseline/`)
- GOPT model: `/home/simon/github/peacock-asr/src/peacock_asr/gopt_model.py`
- GOP feature extraction: `/home/simon/github/peacock-asr/src/peacock_asr/gop.py`
- Backend interface: `/home/simon/github/peacock-asr/src/peacock_asr/backends/`
- Current backbone: xlsr-53 (300M) via `original` backend

---

## 4. Key Technical Details (from research)

### Backbone Parameter Counts

| Model | Params | Pre-training | Vocab (stock) | Fine-tune Needed? |
|---|---|---|---|---|
| xlsr-53 (ours) | 300M | 53 languages | 39 ARPABET | Already done |
| wav2vec2-base | 95M | English 960h | CTC on LibriSpeech | Yes (41 ARPABET) |
| wav2vec2-large | 317M | English 960h | CTC on LibriSpeech | Yes (41 ARPABET) |
| HuBERT-base | 95M | English 960h | None (masked pred.) | Yes (CTC head + 41 ARPABET) |
| HuBERT-large | 317M | English 960h | None (masked pred.) | Yes (CTC head + 41 ARPABET) |
| Citrinet-256 | 10M | English 960h | 256 SentencePiece | Yes (new CTC head, 41 ARPABET) |

### Published PCC Comparison (SpeechOcean762)

| System | Phone PCC | Paradigm | Backbone |
|---|---|---|---|
| GOPT (Kaldi) | 0.612 | GOP + transformer | Kaldi chain |
| HiPAMA | 0.616 | GOP + multi-aspect | Kaldi chain |
| Gradformer | 0.646 | GOP + SSL | Kaldi + SSL |
| HIA | 0.657 | GOP + interactive attn | Kaldi chain |
| **Ours (xlsr-53 + GOPT)** | **0.677** | **GOP-SF + transformer** | **xlsr-53 300M** |
| HierCB + ConPCO | >0.657 (exact unknown) | SSL + ordinal loss | HuBERT/WavLM 3072-dim |

### Key Gap in Literature

No published paper uses wav2vec2-base or HuBERT-base as a CTC backbone for
GOP-based phone-level scoring on SpeechOcean762. All GOP papers use Kaldi models.
All SSL papers use embeddings directly (not GOP). Our paper would be the first
systematic comparison of modern SSL backbones in the GOP pipeline.

### HMamba Architecture Notes

- Replaces GOPT transformer blocks with Mamba (selective state space) blocks
- Hierarchical: phone → word → utterance (like HierTFR)
- Input: GOP features + prosodic features + SSL representations
- Code available: <https://github.com/Fuann/hmamba>
- Built on GOPT codebase (confirms it's a scoring head swap)
- "0.83 PCC" likely refers to utterance-level, not phone-level

### Citrinet-256 Technical Challenges

1. **Vocabulary mismatch**: 256 SentencePiece tokens ≠ 41 ARPABET phonemes
2. **Time reduction**: 8x stride (80ms frames) vs wav2vec2's 20ms frames
3. **GOP-SF compatibility**: GOP-SF algorithm assumes ~20ms frame resolution
4. **NeMo ecosystem**: Different model format, needs conversion or wrapper

Solutions:

- Replace CTC head (256 → 41 tokens) and fine-tune on LibriSpeech
- May need to reduce stride or interpolate posteriors for GOP-SF compatibility
- Use NeMo's `change_vocabulary()` API
