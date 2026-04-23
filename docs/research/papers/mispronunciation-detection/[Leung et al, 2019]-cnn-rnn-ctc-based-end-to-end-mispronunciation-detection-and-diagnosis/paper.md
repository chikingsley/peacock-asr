---
title: "CNN-RNN-CTC Based End-to-end Mispronunciation Detection and Diagnosis"
authors:
  - "Wai-Kim Leung"
  - "Xunying Liu"
  - "Helen Meng"
citation_author: "Leung et al"
year: 2019
venue: "ICASSP 2019"
doi: "10.1109/ICASSP.2019.8682654"
pages: "8132-8136"
source_pdf: "paper.pdf"
extraction_method: "Manually summarized from the published PDF; no public LaTeX source was located."
extracted_at: "2026-03-22"
llm_friendly: true
---

## Metadata

- Authors: Wai-Kim Leung, Xunying Liu, Helen Meng
- Venue: ICASSP 2019
- DOI: 10.1109/ICASSP.2019.8682654
- Pages: 8132-8136
- Task: end-to-end phone-level mispronunciation detection and diagnosis

## TL;DR

This paper is an early end-to-end MDD baseline built from CNN, RNN, and CTC. Its main point is that you can do mispronunciation detection and diagnosis without explicit phonemic or graphemic modeling and without forced alignment. It reports a best F-measure of 74.65% and outperforms ERN, APM, AGM, and APGM on the reported benchmark.

## Abstract

The authors propose an end-to-end speech-recognition-style approach for MDD. A CNN front end and RNN sequence model are trained with CTC, and the resulting system is used to detect and diagnose pronunciation errors directly. The paper argues that this avoids the need for explicit phonological rules, grapheme inputs, or forced alignment.

## Method

- Use a CNN front end to extract acoustic features.
- Feed the result into an RNN sequence model.
- Train with CTC for end-to-end sequence modeling.
- Evaluate detection and diagnosis without hand-built phonemic or graphemic alignment pipelines.
- Compare against ERN, state-level acoustic model, APM, AGM, and APGM baselines.

## Results

- Best reported MDD F-measure: `74.65%` with 1024 hidden units.
- Relative improvement over ERN (S-AM): `32.28%` in F-measure.
- Relative improvement over APM, AGM, and APGM: `9.57%`, `5.04%`, and `2.77%` respectively.
- The paper also reports detection accuracy `89.38%` and diagnosis accuracy `83.24%` at the best hidden size.

## Relevance To Peacock

This is a strong historical baseline for end-to-end MDD. It is particularly relevant when comparing newer SSL or LLM-based pronunciation models against a simpler CTC pipeline that already removed forced alignment and phonological rule engineering.
