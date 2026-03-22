---
title: "GigaAM: Efficient Self-Supervised Learner for Speech Recognition"
authors:
  - "Aleksandr Kutsakov"
  - "Alexandr Maximenko"
  - "Georgii Gospodinov"
  - "Pavel Bogomolov"
  - "Fyodor Minkin"
citation_author: "Kutsakov et al"
year: 2025
doi: null
pages: 5
source_pdf: "paper.pdf"
extraction_method: "Manual rewrite from the local paper.pdf."
extracted_at: "2026-03-15"
llm_friendly: true
---

# GigaAM: Efficient Self-Supervised Learner for Speech Recognition

## Metadata

- Authors: Aleksandr Kutsakov, Alexandr Maximenko, Georgii Gospodinov, Pavel Bogomolov, Fyodor Minkin
- Citation author: Kutsakov et al
- Year: 2025
- DOI: Not stated in the local PDF
- Pages: 5
- Source PDF: `paper.pdf`
- Venue/status: arXiv preprint (`arXiv:2506.01192v1`, `eess.AS`)

## TL;DR

This paper proposes a Russian-focused SSL recipe, `HuBERT-CTC`, that uses
targets derived from a supervised CTC ASR model rather than the usual lower-level
acoustic clustering targets. It pairs that target design with dynamic chunk-size
training so the same pretraining run can support both full-context and streaming
fine-tuning.

On the open Russian benchmarks shown in the paper, the released `CTC` and `RNNT`
variants clearly beat Whisper-large-v3 and the listed FastConformer baseline.
The strongest reported model is the `RNNT` variant.

## Abstract

The paper argues that stable SSL methods such as BEST-RQ are efficient but learn
targets that are too low-level, while classic HuBERT-style methods can still
waste representation capacity on less semantically meaningful signals. The
proposed fix is to build masked-prediction targets from a CTC ASR teacher so the
student learns representations that are more directly useful for recognition.
The authors also add dynamic chunk-size sampling so one pretraining recipe can
later support both long-context and streaming ASR.

## Research Question

Can a CTC-teacher-derived SSL pretraining recipe produce better Russian ASR
representations than HuBERT / BEST-RQ style baselines while also supporting
streaming and full-context fine-tuning from the same pretrained model?

## Method

- Core SSL recipe: `HuBERT-CTC`
- Target construction: K-means clustering on the last-layer states of a
  supervised CTC ASR model, rather than on lower-level acoustic features
- Backbone family: Conformer
- Standard model size in the main setup: `240M` parameters
- Data preprocessing: VAD filtering removes one-minute chunks with more than
  `60%` silence; the paper says this keeps about `80%` of the original data and
  improves performance by roughly `5-10%`
- Streaming support: chunkwise attention plus dynamic chunk-size sampling during
  pretraining so the same base model can later be fine-tuned for either
  full-context or streaming use
- Scaling analysis:
  - pretraining data from `1k` to `100k` hours
  - fine-tuning data from `2` to `2000` hours
  - model size from `30M` to `500M`

## Data

- Main SSL pretraining setup in the paper: `100k` hours of Russian audio,
  `400k` pretraining steps, virtual batch size of `9` hours
- Fine-tuning setup for the released Russian ASR models: Golos, Russian MCV-19,
  Russian LibriSpeech, and SOVA, totaling about `2k` hours
- The paper frames the approach as monolingual Russian pretraining on purpose,
  arguing that domain-matched monolingual data can beat broader multilingual
  pretraining for the downstream task

## Results

Open-source benchmark table in the paper:

- Golos Farfield:
  - Whisper-large-v3: `16.6`
  - FastConformer-RNNT: `6.6`
  - Ours (CTC): `4.3`
  - Ours (RNNT): `3.9`
- Russian MCV-19:
  - Whisper-large-v3: `5.5`
  - FastConformer-RNNT: `5.7`
  - Ours (CTC): `3.1`
  - Ours (RNNT): `2.7`
- Russian LibriSpeech:
  - Whisper-large-v3: `9.5`
  - FastConformer-RNNT: `11.3`
  - Ours (CTC): `5.5`
  - Ours (RNNT): `5.5`

Broader takeaways reported in the paper:

- `HuBERT-CTC` beats HuBERT, BEST-RQ, and wav2vec2-style baselines in the
  authors' Russian setup
- Performance stabilizes once pretraining data reaches roughly `6k` hours,
  suggesting the teacher-target setup is fairly data-efficient after that point
- Dynamic chunking is reported as the most flexible pretraining strategy across
  both long-form and streaming fine-tuning settings

## Limitations / Notes

- The paper is short and very task-focused; it does not provide a broad audit of
  cross-lingual transfer beyond Russian
- The strongest practical story is Russian ASR, not universal multilingual phone
  modeling
- The main paper is only `5` pages, so many implementation details are concise
- The model card advertises a broader `GigaAM-v3` product family than the core
  arXiv paper itself; keep paper claims and model-card claims separate

## Relevance To Peacock

- Highly relevant if Peacock wants a serious open Russian ASR reference point
  rather than another English-centric benchmark
- Less relevant as a direct drop-in for the current `phoneme posterior -> GOP-SF
  -> GOPT` path, because this is an ASR-first family rather than a phone-label
  posterior model designed around pronunciation scoring
- Most relevant immediate use:
  - external baseline for Russian ASR
  - evidence input for `P009` Russian data strategy
  - inspiration for any future Russian-side SSL / streaming work
