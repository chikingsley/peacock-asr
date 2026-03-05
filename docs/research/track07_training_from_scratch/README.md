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

- `[Zhu et al, 2025]-family-of-efficient-models-for-multilingual-phone-recognition.pdf` — ZIPA (Zipformer from scratch)
- `[Li et al, 2025]-phonetic-open-whisper-style-speech-foundation-model.pdf` — POWSM (multi-task)
- `[Bharadwaj et al, 2026]-benchmarking-phone-realization-in-speech-models.pdf` — PRiSM benchmark
- `[Xu et al, 2021]-simple-effective-zero-shot-cross-lingual-phoneme-recognition.pdf` — wav2vec2 zero-shot
- `[Pratap et al, 2023]-scaling-speech-technology-to-1000-languages-mms.pdf` — MMS scaling
- `[Zhou et al, 2025]-phonetic-error-detection-similarity-modeling.pdf` — phoneme similarity modeling
- `[Ye et al, 2025]-lcs-ctc-phonetic-transcription-alignment.pdf` — LCS-CTC soft alignments
- `[Gulati et al, 2020]-conformer-speech-recognition.pdf` — Conformer (Gulati 2020)
- `[Zhu et al, 2022]-byt5-massively-multilingual-g2p.pdf` — multilingual G2P for data labeling

Status: stub — architecture identified, no experiments started.

Last updated: 2026-03-02
