# TODO

## Immediate: Close the PCC Gap with Feature Vectors

- [ ] Extract full GOP feature vectors (41-dim: LPP + LPR + expected count)
  - Currently using scalar GOP scores → PCC 0.320
  - Paper gets PCC 0.581 (poly) / 0.648 (SVR) with feature vectors
  - This is the single biggest PCC improvement available right now
- [ ] Train SVR or GOPT on feature vectors, evaluate PCC
- [ ] Compare: scalar poly regression vs SVR vs GOPT on same features

## Phase 1: w2v-BERT 2.0 Phoneme Head (Lowest Risk)

- [ ] Build ARPABET vocabulary (39 phones + blank + pad = 41 tokens)
- [ ] Prepare LibriSpeech with phoneme labels (text → ARPABET via CMU dict / G2P)
- [ ] Fine-tune w2v-BERT 2.0 with CTC on LibriSpeech (A100 80GB, ~6-12h)
  - Follow HuggingFace blog recipe exactly, swap vocab only
  - See docs/research/05_PHONEME_HEADS.md Phase 1 for details
- [ ] Create new backend (`w2v_bert_phoneme.py`), plug into benchmark
- [ ] Compare PCC with xlsr-espeak baseline (0.320 scalar / TBD feature vectors)

## Phase 2: omniASR Head Swap (Alternative Path)

- [ ] Swap omniASR-CTC-1B final_proj (9812 chars → 41 phones)
- [ ] Fine-tune on LibriSpeech (~2-4h)
- [ ] Note: fairseq2 ecosystem — may need HuggingFace port

## Phase 3: Ablations & Analysis

- [ ] Per-phone improvement analysis
- [ ] Data efficiency study (100h vs 460h vs 960h LibriSpeech)
- [ ] Temporal resolution experiment (downsample posteriors to 80ms)
- [ ] Logit-based GOP-SF (pre-softmax values in CTC forward) — arXiv: 2506.12067

## Research Infrastructure

- [ ] Build paper management system (Zotero replacement)
  - Postgres backend with pgvector embeddings + tsvector full-text search
  - Store papers as markdown with structured metadata
  - Citation graph / relationship tracking

## Key Papers to Process

- [x] ZIPA (2505.23170) — saved to docs/papers/
- [x] POWSM (2510.24992) — saved to docs/papers/
- [x] PRiSM (2601.14046) — saved to docs/papers/
- [x] CTC-based-GOP (2507.16838v3) — saved to docs/papers/
- [ ] Xu et al. 2022 — wav2vec2-xlsr-53-espeak-cv-ft paper (arXiv: 2109.11680)
- [ ] Allosaurus (ICASSP 2020) — universal phone recognizer
- [ ] Enhancing GOP with Phonological Knowledge (2506.02080)
- [ ] Logit-based GOP Scores (2506.12067)
- [ ] Original GOP paper (Witt & Young 2000)
- [ ] GOPT Transformer paper (Gong et al. ICASSP 2022)
- [ ] SpeechOcean762 dataset paper (2104.01378)

## Completed

- [x] Implement GOP-SF algorithm (scalar scores)
- [x] Build backend architecture with pluggable phoneme models
- [x] Benchmark three backends (original, xlsr-espeak, zipa)
- [x] Identify ZIPA character-level vocab incompatibility
- [x] Document research landscape (05_PHONEME_HEADS.md)
- [x] Collect ZIPA run 2 results (32/39 phones, ER/G fixes)
- [x] Compare all three backends side-by-side
- [x] Document findings in DECISIONS.md
