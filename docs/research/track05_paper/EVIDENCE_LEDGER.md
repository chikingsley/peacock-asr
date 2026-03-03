# Track 1 Evidence Ledger (Phoneme CTC Heads)

This ledger is the evidence backbone for writing
`05_PHONEME_HEADS.md` as a paper-style draft.

Scope:
- Pronunciation scoring pipeline centered on CTC phoneme posteriors, GOP-SF, and GOPT-style scoring.
- Sources are marked with local PDF paths whenever available.

Citation policy:
- Use standard numbered citations in text: `[1]`, `[2]`, ...
- Prefer local PDF citations whenever available.
- Use `docs/research/track05_paper/refs.bib` as the canonical bib source for citekeys and metadata.

---

## 1. Claim Map (Current)

| ID | Claim in 05_ narrative | Evidence status | Primary citations |
|---|---|---|---|
| C1 | Segmentation-free GOP from CTC posteriors is the core scoring method and performs strongly on Speechocean762/CMU Kids. | Supported | [1] |
| C2 | GOP-SF feature vectors (LPP/LPR/etc.) are suitable as downstream model inputs. | Supported | [1] |
| C3 | Restricting substitutions/confusions is a viable GOP enhancement direction. | Supported | [3] |
| C4 | Logit-based GOP variants can outperform probability-only GOP for assessment tasks. | Partially supported (mixture variant only) | [4], [15] |
| C5 | A GOPT-style transformer scorer is a strong downstream baseline for pronunciation scoring. | Supported | [2], [13], [14] |
| C6 | SSL + CTC fine-tuning is a valid route for pronunciation assessment features. | Supported | [5] |
| C7 | SpeechOcean762 is a valid benchmark dataset and should be used as central eval set. | Supported | [6] |
| C8 | Modern multilingual phone recognizers (ZIPA/POWSM/PRiSM context) support phone-centric modeling direction. | Supported | [7], [8], [9] |
| C9 | Character-level IPA vocab mismatches can harm ARPABET-targeted scoring pipelines. | Partially supported (needs explicit experiment citation in Methods/Results) | [7], [13], [14] |
| C10 | Our current repo-level benchmark numbers (e.g., GOPT PCC 0.662) are reproducible internal evidence. | Supported (internal evidence, not literature) | [13], [14], [15] |

Notes:
- `C9` is strong internally but should cite a specific experiment artifact in the future paper’s Results section.

---

## 2. Internal Evidence Anchors (Repo Artifacts)

These are not external papers; they are reproducibility anchors for our own claims:

- Run logs for current GOPT experiments:
  - `/home/simon/github/peacock-asr/runs/2026-03-02_mlflow_batch/original_gopt.log`
  - `/home/simon/github/peacock-asr/runs/2026-03-02_mlflow_batch/xlsr-espeak_gopt.log`
- Phase-1 baseline batch artifacts:
  - `/home/simon/github/peacock-asr/runs/2026-03-03_001037_track05_phase1_baseline/summary.tsv`
  - `/home/simon/github/peacock-asr/runs/2026-03-03_001037_track05_phase1_baseline/aggregates.tsv`
- Phase-2 scalar logit batch artifacts:
  - `/home/simon/github/peacock-asr/runs/2026-03-03_045426_track05_phase2_logit_scalar/summary.tsv`
  - `/home/simon/github/peacock-asr/runs/2026-03-03_045426_track05_phase2_logit_scalar/aggregates.tsv`
- Phase-2b dense alpha sweep artifacts:
  - `/home/simon/github/peacock-asr/runs/2026-03-03_080157_alpha_sweep_xlsr-espeak__wav2vec2-xlsr-53-espeak-cv-ft/alpha_sweep.tsv`
  - `/home/simon/github/peacock-asr/runs/2026-03-03_080157_alpha_sweep_xlsr-espeak__wav2vec2-xlsr-53-espeak-cv-ft/alpha_sweep_meta.json`
- 05_ source narrative:
  - `/home/simon/github/peacock-asr/docs/research/05_PHONEME_HEADS.md`
- GOP implementation used in repo:
  - `/home/simon/github/peacock-asr/src/gopt_bench/gop.py`

Recommendation for paper rewrite:
- Treat these as `Method implementation evidence` and `Experimental evidence`.
- Keep them separate from bibliography citations.

Phase-2 scalar logit ablation snapshot (2026-03-03):
- `B1 gop_sf`: PCC `0.3195`, MSE `0.6655`
- `B2 logit_margin`: PCC `0.1849`, MSE `0.8177`
- `B3 logit_combined a=0.25`: PCC `0.3452`, MSE `0.5981`
- `B4 logit_combined a=0.50`: PCC `0.3222`, MSE `0.6322`
- `B5 logit_combined a=0.75`: PCC `0.2664`, MSE `0.7131`

Interpretation note:
- In this stack, pure `logit_margin` underperforms baseline scalar GOP-SF.
- A low-weight mixture (`a=0.25`) improves over baseline.
- Dense sweep (`a=0.00..1.00`, step `0.05`) confirms the best point at
  `a=0.25` with PCC `0.3452` and MSE `0.5981`.

---

## 3. Coverage Update (2026-03-02)

Newly mirrored local PDFs since the initial ledger draft:
- [2] GOPT ICASSP 2022 (`2205.03432`)
- [10] Xu et al. (`2109.11680`)
- [11] Pratap et al. (`2305.13516`)
- [12] Li et al. (`2002.11800`)
- [7] ZIPA (`2505.23170`)
- [8] POWSM (`2510.24992`)
- [9] PRiSM (`2601.14046`)

Current status:
- Core Track 05 citation set is now local-PDF backed.
- Remaining rigor gap is not source availability; it is tightening claim-to-result linkage in Methods/Results.

---

## 4. Proposed Citation Set (PDF-first)

### Core Pipeline (must-cite in Methods/Results)

[1] X. Cao, Z. Fan, T. Svendsen, and G. Salvi, “Segmentation-Free Goodness of Pronunciation,” 2026.  
Local PDF: [2507.16838.pdf](../../papers/core_segmentation_free/2507.16838.pdf)

[2] Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and J. Glass, “Transformer-Based Multi-Aspect Multi-Granularity Non-native English Speaker Pronunciation Assessment,” ICASSP 2022.  
Local PDF: [2205.03432_gopt_transformer_multi_aspect_pronunciation_assessment.pdf](../../papers/end_to_end_assessment/2205.03432_gopt_transformer_multi_aspect_pronunciation_assessment.pdf)

[3] A. Parikh et al., “Enhancing GOP in CTC-Based MDD with Phonological Knowledge,” 2025.  
Local PDF: [2506.02080.pdf](../../papers/core_segmentation_free/2506.02080.pdf)

[4] “Evaluating Logit-Based GOP Scores,” 2025.  
Local PDF: [2506.12067.pdf](../../papers/gop_methods/2506.12067.pdf)

[5] J.-H. Kim et al., “Automatic Pronunciation Assessment using Self-Supervised Speech Representation Learning,” 2022.  
Local PDF: [2204.03863.pdf](../../papers/end_to_end_assessment/2204.03863.pdf)

[6] J. Zhang et al., “Speechocean762: An Open-Source Non-native English Speech Corpus for Pronunciation Assessment,” 2021.  
Local PDF: [2104.01378.pdf](../../papers/datasets/2104.01378.pdf)

### Adjacent Phone-Model Literature (for Related Work)

[7] J. Zhu et al., “ZIPA,” 2025.  
Local PDF: [2505.23170_zipa_multilingual_phone_recognition.pdf](../../papers/phoneme_recognition/2505.23170_zipa_multilingual_phone_recognition.pdf)

[8] C.-J. Li et al., “POWSM,” 2026.  
Local PDF: [2510.24992_powsm_phonetic_open_whisper_style_speech_foundation_model.pdf](../../papers/phoneme_recognition/2510.24992_powsm_phonetic_open_whisper_style_speech_foundation_model.pdf)

[9] S. Bharadwaj et al., “PRiSM,” 2026.  
Local PDF: [2601.14046_prism_benchmarking_phone_realization_in_speech_models.pdf](../../papers/phoneme_recognition/2601.14046_prism_benchmarking_phone_realization_in_speech_models.pdf)

### Supporting Phoneme Recognition References

[10] Q. Xu, A. Baevski, and M. Auli, “Simple and Effective Zero-shot Cross-lingual Phoneme Recognition,” 2021.  
Local PDF: [2109.11680_simple_effective_zero_shot_cross_lingual_phoneme_recognition.pdf](../../papers/phoneme_recognition/2109.11680_simple_effective_zero_shot_cross_lingual_phoneme_recognition.pdf)

[11] V. Pratap et al., “Scaling Speech Technology to 1,000+ Languages,” 2023.  
Local PDF: [2305.13516_scaling_speech_technology_to_1000_languages_mms.pdf](../../papers/phoneme_recognition/2305.13516_scaling_speech_technology_to_1000_languages_mms.pdf)

[12] X. Li et al., “Universal Phone Recognition with a Multilingual Allophone System,” 2020.  
Local PDF: [2002.11800_universal_phone_recognition_multilingual_allophone_system.pdf](../../papers/phoneme_recognition/2002.11800_universal_phone_recognition_multilingual_allophone_system.pdf)

### Internal Experimental Evidence (non-bibliographic)

[13] Our GOPT benchmark logs (2026-03-02):  
`/home/simon/github/peacock-asr/runs/2026-03-02_mlflow_batch/original_gopt.log`  
`/home/simon/github/peacock-asr/runs/2026-03-02_mlflow_batch/xlsr-espeak_gopt.log`

[14] `05_PHONEME_HEADS.md` tracked benchmark statement:  
`/home/simon/github/peacock-asr/docs/research/05_PHONEME_HEADS.md`

[15] Track05 reproducible batch tables (2026-03-03):  
`/home/simon/github/peacock-asr/runs/2026-03-03_001037_track05_phase1_baseline/summary.tsv`  
`/home/simon/github/peacock-asr/runs/2026-03-03_001037_track05_phase1_baseline/aggregates.tsv`  
`/home/simon/github/peacock-asr/runs/2026-03-03_045426_track05_phase2_logit_scalar/summary.tsv`  
`/home/simon/github/peacock-asr/runs/2026-03-03_045426_track05_phase2_logit_scalar/aggregates.tsv`  
`/home/simon/github/peacock-asr/runs/2026-03-03_080157_alpha_sweep_xlsr-espeak__wav2vec2-xlsr-53-espeak-cv-ft/alpha_sweep.tsv`  
`/home/simon/github/peacock-asr/runs/2026-03-03_080157_alpha_sweep_xlsr-espeak__wav2vec2-xlsr-53-espeak-cv-ft/alpha_sweep_meta.json`

---

## 5. Recommendation for Full Paper Draft

Use this ledger as the single source of truth for claims:
- `Methods`: cite [1], [2], [5] plus implementation anchors.
- `Results`: cite [6], [13], [14] for our numbers; [1], [2] for comparative context.
- `Related Work`: cite [3], [4], [7], [8], [9], [10], [11], [12].

This keeps the draft rigorous while making evidence provenance explicit.
