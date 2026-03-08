---
title: "End-to-End Mispronunciation Detection and Diagnosis From Raw Waveforms"
authors:
  - "Bi-Cheng Yan"
  - "Berlin Chen"
citation_author: "Yan"
year: 2021
doi: "10.23919/EUSIPCO54536.2021.9615987"
pages: "61-65"
source_pdf: "paper.pdf"
extraction_method: "Manually curated from the local PDF, with nearby OCR text used only to recover result tables and implementation details."
extracted_at: "2026-03-07T20:02:23-08:00"
llm_friendly: true
---

# End-to-End Mispronunciation Detection and Diagnosis From Raw Waveforms

## Metadata
- Authors: Bi-Cheng Yan, Berlin Chen
- Year: 2021
- Venue: EUSIPCO 2021
- DOI: 10.23919/EUSIPCO54536.2021.9615987
- Pages: 61-65
- Task: Mispronunciation detection and diagnosis from raw speech

## TL;DR
This paper asks whether raw-waveform front ends can replace hand-crafted acoustic features for end-to-end mispronunciation detection and diagnosis. The main contribution is adding a SincNet front end before a hybrid CTC-attention recognizer so the model learns task-specific acoustic filters directly from waveform input.

SincNet gives the best phone error rate and competitive detection performance, while a raw-waveform CNN front end gives the best diagnosis accuracy. Relative to GOP, all end-to-end variants are much stronger on detection.

## Abstract
The authors frame MDD as end-to-end free-phone recognition directly from learner waveforms. They argue that raw waveforms preserve acoustic detail that standard features may discard, and they use SincNet because it has fewer parameters and more interpretable filters than ordinary CNN front ends. Experiments on L2-ARCTIC show strong phone recognition and diagnosis performance with raw-waveform models.

## Research Question
Can a raw-waveform front end, especially SincNet, learn useful pronunciation representations for MDD and match or exceed feature-based end-to-end systems and GOP baselines?

## Method
- Core recognizer: hybrid CTC-attention model used as an end-to-end free-phone recognizer.
- Front-end variants:
- MFCC input.
- FBANK input.
- Raw waveform plus standard CNN front end.
- Raw waveform plus SincNet front end.
- SincNet uses parametrized sinc-based band-pass filters instead of fully learned convolution kernels in the first layer.
- Evaluation includes three views:
- Phone recognition quality on correctly pronounced utterances.
- Mispronunciation detection.
- Mispronunciation diagnosis accuracy.

## Data
- Non-native speech: L2-ARCTIC, containing correctly pronounced and mispronounced utterances from 24 L2 English speakers across six L1s.
- Native English bootstrap data: TIMIT.
- Phone inventory: 39 canonical phones aligned to the CMU dictionary setup.
- The local PDF text confirms train/test splits following the earlier L2-ARCTIC MDD recipe, but the OCR rendering of Table I is noisy enough that I would not rely on every numeric cell without re-extraction.

## Results
- Phone error rate on correctly pronounced utterances:
- MFCC: 9.25.
- FBANK: 8.45.
- Raw + CNN: 6.44.
- Raw + SincNet: 5.50.
- Mispronunciation detection F1:
- GOP with FBANK: 42.42.
- CTC-ATT with MFCC: 53.59.
- CTC-ATT with FBANK: 53.83.
- CTC-ATT with raw + CNN: 51.10.
- CTC-ATT with raw + SincNet: 52.57.
- Diagnosis accuracy (`DAR`):
- Leung et al. CNN-RNN-CTC baseline: 32.10.
- CTC-ATT with FBANK: 59.84.
- CTC-ATT with raw + CNN: 62.08.
- CTC-ATT with raw + SincNet: 60.96.
- Qualitative claim from the paper:
- SincNet filters concentrate below 2 kHz and appear to capture interpretable vowel- and pitch-related regions.
- The authors also report that multilingual SincNet filters adapt quickly to different L1 groups.

## Limitations / Notes
- Raw-waveform modeling does not clearly beat the best FBANK system on detection F1; the main empirical wins are stronger PER and solid diagnosis accuracy.
- The best diagnosis result comes from raw + CNN, not raw + SincNet.
- The paper is still built around forced text prompts and read speech.
- Some corpus-statistics cells in the local OCR output are garbled, so the note keeps the dataset description high-confidence rather than repeating uncertain counts.

## Relevance To Peacock
- This is directly relevant if Peacock wants to compare learned waveform front ends against standard acoustic features for pronunciation-sensitive ASR.
- SincNet is interesting mainly as a lightweight, interpretable front end rather than a guaranteed end-task winner.
- The paper suggests that raw-waveform models can improve phone discrimination and diagnosis quality even when final detection F1 gains are small.
