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
| C1 | wav2vec2-base (95M) can serve as GOP backbone with acceptable PCC | **CONFIRMED: PCC 0.640 ± 0.009** (5 seeds) | [1] |
| C2 | HuBERT-base (95M) matches or exceeds wav2vec2-base for GOP | **CONFIRMED: PCC 0.6489 +/- 0.0093** (5 seeds), above wav2vec2-base | [2], [3] |
| C3 | 95M backbone loses < 5% PCC vs 300M xlsr-53 | **CONFIRMED via HuBERT-base: 4.2% drop** (0.6489 vs 0.6774) | [1], [2] |
| C4 | Citrinet-256 (10M) can be adapted for phoneme-level GOP, but current P2-B quality is clearly below the SSL CTC backbones | **CONFIRMED: PCC 0.5574 +/- 0.0133** (5 seeds), adaptation works but is not competitive yet | [4] |
| C5 | Scoring head choice (GOPT vs HMamba) matters less than backbone | Hypothesis, needs P3 | [5], [6] |
| C6 | No published paper tests wav2vec2-base or HuBERT-base as GOP backbone | Supported by literature search | [3] |
| C7 | Our xlsr-53 + GOPT baseline (0.677) already exceeds HIA SOTA (0.657) | Supported (Track 05 data vs [7]) | Internal, [7] |
| C8 | w2v-BERT-2.0 (600M) reaches near-parity with our xlsr-53 + GOPT baseline | **CONFIRMED: PCC 0.6755 +/- 0.0066** (5 seeds) | [1], [2] |

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

- GOPT baseline runs: final `P001` aggregate table
  (`projects/P001-gop-baselines/experiments/final/results/aggregate_summary.tsv`)
- GOPT model: `/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/p003_compact/gopt_model.py`
- GOP feature extraction: `/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/p003_compact/gop.py`
- Backend loader: `/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/p003_compact/backend_loader.py`
- P003 HF backend: `/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/p003_compact/backends/hf_ctc.py`
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
| **Ours (xlsr-53 + GOPT)** | **0.677 ± 0.003** | **GOP-SF + transformer** | **xlsr-53 300M** |
| **Ours (wav2vec2-base + GOPT)** | **0.640 ± 0.009** | **GOP-SF + transformer** | **wav2vec2-base 95M** |
| **Ours (HuBERT-base + GOPT)** | **0.649 +/- 0.009** | **GOP-SF + transformer** | **HuBERT-base 95M** |
| **Ours (w2v-BERT-2.0 + GOPT)** | **0.676 +/- 0.007** | **GOP-SF + transformer** | **w2v-BERT-2.0 600M** |
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

---

## 5. Experiment Results

### E1: wav2vec2-base (95M) — Phase 1B

- **Date**: 2026-03-05
- **Model**: `Peacockery/wav2vec2-base-phoneme-en` (HF Hub)
- **Training**: 3 epochs on LibriSpeech 960h, CTC loss, 41 ARPABET phones
  - GPU: RTX 2000 Ada 16GB (RunPod), ~10h
  - Sweep: `projects/P003-compact-backbones/experiments/sweeps/final/train_wav2vec2_base.yaml`
  - W&B training run: `peacockery/w2v-bert-phoneme-en/runs/0rt9dwf1`
- **Eval**: GOPT scoring head, 5 seeds, SpeechOcean762
  - Sweep: `projects/P003-compact-backbones/experiments/sweeps/final/eval_wav2vec2_base.yaml`
  - W&B sweep: `peacockery/peacock-asr/sweeps/s6tqenxi`
  - Backend: `hf:Peacockery/wav2vec2-base-phoneme-en`

| Seed | PCC | MSE |
|------|-----|-----|
| 1 | 0.6312 | — |
| 2 | 0.6366 | — |
| 3 | 0.6397 | — |
| 4 | 0.6544 | — |
| 5 | 0.6379 | — |
| **Mean ± std** | **0.640 ± 0.009** | **0.080** |

**vs baseline**: xlsr-53 (300M) GOPT PCC = 0.677 ± 0.003 → **5.5% relative drop, 3.3x fewer params**

Claims supported: C1 (confirmed), C3 (marginal — 5.5% > 5% threshold, but close)

### E2: HuBERT-base (95M) — Phase 1C

- **Date**: 2026-03-06
- **Model**: `Peacockery/hubert-base-phoneme-en` (HF Hub)
- **Training**: 3 epochs on LibriSpeech 960h, CTC loss, 41-token ARPABET vocabulary
  - Local run completed successfully
  - Canonical train sweep:
    `projects/P003-compact-backbones/experiments/sweeps/final/train_hubert_base.yaml`
  - W&B training run:
    `peacockery/hubert-base-phoneme-en/runs/qe7scuxw`
  - Best checkpoint selected for export: `checkpoint-8500`
  - Best tracked trainer metric:
    - `eval_per = 0.9988901220865705`
    - `eval_loss = 0.11406530439853668`
  - Final trainer eval at step `13000`:
    - `eval_loss = 0.10874085873365402`
    - `eval_per = 0.9992600813910469`
- **Eval**: GOPT scoring head, 5 seeds, SpeechOcean762
  - Canonical eval sweep:
    `projects/P003-compact-backbones/experiments/sweeps/final/eval_hubert_base.yaml`
  - Finished W&B sweep:
    `peacockery/peacock-asr-p003-compact-backbones/sweeps/w9wu57e8`
  - Backend:
    `hf:Peacockery/hubert-base-phoneme-en`

| Seed | PCC | MSE |
|------|-----|-----|
| 1 | 0.6402 | 0.0812 |
| 2 | 0.6456 | 0.0811 |
| 3 | 0.6535 | 0.0791 |
| 4 | 0.6628 | 0.0772 |
| 5 | 0.6423 | 0.0811 |
| **Mean ± std** | **0.6489 +/- 0.0093** | **0.0800 +/- 0.0018** |

Interpretation:

- Improves over `wav2vec2-base` (`0.640 +/- 0.009`)
- Narrows the gap to the `xlsr-53 + GOPT` baseline to about `4.2%`
- Still sits clearly below `w2v-BERT-2.0` (`0.6755 +/- 0.0066`) and slightly below
  the `P001` `xlsr-53 + GOPT` baseline (`0.6774 +/- 0.0127`)

Claims supported: `C2`, `C3`

### E3: w2v-BERT-2.0 (600M) — Phase 0A

- **Date**: 2026-03-07
- **Model**: `Peacockery/w2v-bert-phoneme-en` (HF Hub)
- **Training**: completed on LibriSpeech 960h with 41-token ARPABET CTC head
  - Canonical train artifact: `Peacockery/w2v-bert-phoneme-en`
  - Canonical eval sweep:
    `projects/P003-compact-backbones/experiments/sweeps/final/eval_w2v_bert.yaml`
  - W&B eval sweep:
    `peacockery/peacock-asr-p003-compact-backbones/sweeps/hciltqza`
  - Backend:
    `hf:Peacockery/w2v-bert-phoneme-en`

| Seed | PCC | MSE |
|------|-----|-----|
| 1 | 0.6739 | 0.0752 |
| 2 | 0.6820 | 0.0742 |
| 3 | 0.6828 | 0.0747 |
| 4 | 0.6684 | 0.0776 |
| 5 | 0.6706 | 0.0763 |
| **Mean ± std** | **0.6755 +/- 0.0066** | **0.0756 +/- 0.0014** |

Interpretation:

- Strongly improves over `wav2vec2-base` (`0.640 +/- 0.009`)
- Effectively matches our `xlsr-53 + GOPT` baseline (`0.6774 +/- 0.0127`)
- Does not produce a decisive gain over the strong `P001` baseline, but does
  validate the fine-tuning pipeline and establishes a 600M Pareto point

Claims supported: `C8`

### E4: Citrinet-256 (10M) — Phase 2B

- **Date**: 2026-03-07
- **Model**: `Peacockery/citrinet-256-phoneme-en` (NeMo artifact / `nemo:` backend)
- **Training**: local `train_clean_100` fine-tune completed
  - Full artifact:
    `projects/P003-compact-backbones/experiments/citrinet/checkpoints/citrinet_256_p2b_trainclean100_local_full_s3/artifacts/citrinet_256_p2b_trainclean100_local_full_s3.nemo`
  - Final training report:
    `projects/P003-compact-backbones/experiments/citrinet/checkpoints/citrinet_256_p2b_trainclean100_local_full_s3/report.json`
- **Eval**: GOPT scoring head, 5 seeds, SpeechOcean762
  - Sweep log:
    `projects/P003-compact-backbones/experiments/citrinet/logs/sweep_eval_citrinet_hanv3zeq.log`
  - Backend:
    `nemo:Peacockery/citrinet-256-phoneme-en`

| Seed | PCC | MSE |
|------|-----|-----|
| 1 | 0.5629 | 0.0971 |
| 2 | 0.5715 | 0.0952 |
| 3 | 0.5660 | 0.0946 |
| 4 | 0.5429 | 0.1011 |
| 5 | 0.5436 | 0.1003 |
| **Mean ± std** | **0.5574 +/- 0.0133** | **0.0977 +/- 0.0029** |

Interpretation:

- The NeMo adaptation path works end-to-end.
- Current Citrinet quality is well below `wav2vec2-base` (`0.640 +/- 0.009`)
  and `HuBERT-base` (`0.6489 +/- 0.0093`).
- This makes Citrinet a valid extreme-compression branch, but not a competitive
  default backbone in its current form.

Claims supported: `C4`
