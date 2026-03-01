# TODO

Current PCC: **0.648** (GOPT, matched paper target) | See [EXPERIMENTS.md](EXPERIMENTS.md) for full run history.

## Next

- [ ] Compare: original vs xlsr-espeak backends with GOPT (fast now with caching)

### Done (Cleanup + Caching)

- [x] Add feature caching to disk (`~/.cache/peacock-asr/features/`, `--no-cache` to bypass)
- [x] Extract shared PCC computation in evaluate.py (`_compute_pcc()` helper)
- [x] Remove dead `GOPResult.phones` field
- [x] Move env var `GOPT_BENCH_CTC_BACKEND` → `settings.ctc_feature_backend`
- [x] Replace `Path(__file__).parents[3]` with `_find_repo_root()` (walks up to `pyproject.toml`)
- [x] `_step_denom` helpers — kept split (each handles a distinct CTC case, re-merging triggers ruff)

## Then: Better Backends (the actual research)

Once caching enables fast iteration, swap in better phoneme models.

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
- [x] GOPT Transformer paper (Gong et al. ICASSP 2022)
- [ ] SpeechOcean762 dataset paper (2104.01378)
