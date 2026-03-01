# TODO

Current PCC: **0.548** | Paper target: **0.648** | See [EXPERIMENTS.md](EXPERIMENTS.md) for full run history.

## Next: GOPT Transformer

Reproduce the paper's main result by training their GOPT transformer on our feature vectors.
Reference code: `references/gopt-transformer/`

- [ ] Adapt GOPT model input_dim (84 → 42) for our feature vectors
- [ ] Restructure data pipeline from per-phone tuples to per-utterance batches
- [ ] Train GOPT on SpeechOcean762 (MSE loss, 100 epochs, ~minutes on CPU)
- [ ] Compare PCC with SVR baseline (0.548) and paper target (0.648)

## Then: Better Backends (the actual research)

Once GOPT reproduction validates our features, swap in better phoneme models.

- [ ] Compare: original vs xlsr-espeak backends with feature vectors + GOPT
- [ ] Build ARPABET vocabulary (39 phones + blank + pad = 41 tokens)
- [ ] Prepare LibriSpeech with phoneme labels (text → ARPABET via CMU dict / G2P)
- [ ] Fine-tune w2v-BERT 2.0 with CTC on LibriSpeech (A100 80GB, ~6-12h)
- [ ] Create new backend (`w2v_bert_phoneme.py`), plug into benchmark
- [ ] Compare PCC with original baseline

## Later

- [ ] omniASR head swap (9812 chars → 41 phones, fairseq2 ecosystem)
- [ ] Per-phone improvement analysis across backends
- [ ] Data efficiency study (100h vs 460h vs 960h LibriSpeech)
- [ ] Logit-based GOP-SF (pre-softmax values in CTC forward) — arXiv: 2506.12067
- [ ] Feature normalization (z-score per feature dimension before SVR)

## Research Infrastructure

- [ ] Build paper management system (Zotero replacement)

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
