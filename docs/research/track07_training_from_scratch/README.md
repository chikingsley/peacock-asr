# Track 07 Paper Workspace: Training a Phoneme Recognition Model from Scratch

Working title:

- **From-Scratch Phoneme CTC: Is Conformer Training Competitive with Fine-Tuned SSL Encoders?**

Purpose:

- Train a full phoneme recognition model end-to-end (not fine-tuning a pretrained SSL encoder).
- Determine whether from-scratch training can match or exceed Track 05 fine-tuned w2v-BERT 2.0
  for the GOP posterior quality that matters for pronunciation scoring.
- Follow the lab methodology: start with smallest viable experiment, one change at a time.

Source of truth:

- Evidence ledger: `./EVIDENCE_LEDGER.md`
- Bibliography: `./refs.bib`
- Ablation plan: `./ABLATION_PLAN.md`

Draft files:

- `manuscript.md` (primary writing file)

Citation convention:

- Use Pandoc/Quarto citekeys, e.g. `[@gulati2020conformer]`.
- All citekeys are in `./refs.bib`.

Process:

1. Phase 1: Reproduce icefall TIMIT TDNN-LSTM recipe (learning sandbox, no paper value).
2. Phase 2: Adapt icefall Conformer recipe for phoneme CTC on LibriSpeech.
3. Phase 3: Head-to-head vs fine-tuned w2v-BERT 2.0 on SpeechOcean762 GOP quality.
4. Phase 4 (conditional): Zipformer phoneme CTC if Conformer shows promise.

Key external references:

- ZIPA: Zipformer trained on 17K hours, 2.71 PFER — our reference for from-scratch ceiling
- PRiSM: benchmarks encoder-CTC as the most stable approach
- icefall: k2-fsa/icefall recipes (TIMIT TDNN-LSTM is only phoneme CTC recipe)

Papers (PDFs in `./papers/`):

- `2505.23170_zipa_multilingual_phone_recognition.pdf` — ZIPA (Zipformer from scratch)
- `2510.24992_powsm_phonetic_open_whisper_style_speech_foundation_model.pdf` — POWSM (multi-task)
- `2601.14046_prism_benchmarking_phone_realization_in_speech_models.pdf` — PRiSM benchmark
- `2109.11680_simple_effective_zero_shot_cross_lingual_phoneme_recognition.pdf` — wav2vec2 zero-shot
- `2305.13516_scaling_speech_technology_to_1000_languages_mms.pdf` — MMS scaling
- `2507.14346.pdf` — phoneme similarity modeling
- `2508.03937.pdf` — LCS-CTC soft alignments
- `2005.08100_conformer_convolution_augmented_transformer.pdf` — Conformer (Gulati 2020)
- `2204.03067_byt5_massively_multilingual_g2p.pdf` — multilingual G2P for data labeling

Status: stub — architecture identified, no experiments started.

Last updated: 2026-03-02
