---
title: "Developing an Automatic Pronunciation Scorer: Aligning Speech Evaluation Models and Applied Linguistics Constructs"
authors:
  - "Danwei Cai"
  - "Ben Naismith"
  - "Maria Kostromitina"
  - "Zhongwei Teng"
  - "Kevin P. Yancey"
  - "Geoffrey T. LaFlair"
citation_author: "Cai et al."
year: 2025
doi: "10.1111/lang.70000"
pages: 34
source_pdf: "paper.pdf"
extraction_method: "manual-curated from local paper.pdf and nearby repository markdown"
extracted_at: "2026-03-07"
llm_friendly: true
---

# Developing an Automatic Pronunciation Scorer: Aligning Speech Evaluation Models and Applied Linguistics Constructs

## Metadata

- Authors: Danwei Cai, Ben Naismith, Maria Kostromitina, Zhongwei Teng, Kevin P. Yancey, Geoffrey T. LaFlair
- Year: 2025
- DOI: 10.1111/lang.70000
- Pages: 34
- Source PDF: `paper.pdf`
- Context: Duolingo English Test / high-stakes open-response pronunciation scoring

## TL;DR

- This paper is less about a novel CAPT loss and more about building a construct-aligned, operational pronunciation scorer for open-response English assessment.
- The key claim is that training on diverse, CEFR-aligned human ratings matters more than training on a convenient CAPT corpus like Speechocean762.
- The resulting scorer reaches Spearman 0.82 and QWK 0.81 against expert ratings, outperforming GOP, Whisper-ASR confidence, Microsoft Pronunciation Assessment, and a scorer trained on Speechocean762.

## Abstract

The paper develops an automatic pronunciation scorer for high-stakes English testing and explicitly ties it to applied-linguistics constructs such as intelligibility, comprehensibility, segmental control, and prosody. Instead of relying on opaque operational proxies alone, it builds a bespoke human-rated dataset aligned to CEFR descriptors and recent pronunciation theory. The scorer is then evaluated for predictive validity against expert ratings, for comparative performance against strong baselines, and for subgroup fairness via differential feature functioning (DFF).

## Research Question

Can an automatic pronunciation scorer for open-response English speech be built so that it is both construct-aligned with human expert judgment and measurably fair across major test-taker subgroups?

## Method

- The scorer adapts a hierarchical pronunciation model originally suited to read-aloud tasks for open-response speech.
- Pipeline:
- transcribe the response with large Whisper
- segment the transcript into 10-35 word chunks
- force-align each chunk with audio
- score each segment
- compute a duration-weighted response-level score
- Training target: holistic human pronunciation ratings on a 1-6 CEFR-aligned scale grounded in overall phonological control, sound articulation, and prosodic features.
- Baselines include GOP, Whisper ASR confidence, Microsoft Pronunciation Assessment (prosody and pronunciation scores), and the same scorer trained on Speechocean762 rather than the bespoke DET dataset.

## Data

- Human-ratings dataset: 2,624 speaking samples from 1,312 test takers
- Composition:
- Pilot subset: 312 samples
- Main L2 English subset: 2,060 samples
- Main L1 English subset: 252 samples
- Responses come from operational DET sessions between February 2022 and March 2023.
- Samples are approximately 30 seconds long and selected at utterance boundaries.
- The paper groups L1 backgrounds into 13 language-family/region groups and balances for gender and proficiency as much as possible.
- Interrater agreement on valid double-rated items:
- QWK 0.85
- Spearman 0.84
- ICC 0.85
- Evaluation setup: 5-fold cross-validation
- Average segment duration after chunking: 11.9 seconds; average total segmented duration per response: 26.5 seconds

## Results

- Human interrater benchmark: Spearman 0.86, QWK 0.87
- Proposed scorer: Spearman 0.82 ± 0.02, QWK 0.81 ± 0.01
- Baselines:
- GOP scorer: 0.66 / 0.60
- Whisper (medium) ASR scorer: 0.72 / 0.69
- MPA prosody scorer: 0.77 / 0.71
- MPA pronunciation scorer: 0.75 / 0.70
- Same scorer trained on Speechocean762: 0.71 / 0.65
- Steiger tests against all listed baselines are reported as `p < .001`.
- DFF findings:
- No meaningful gender bias
- Consistent negative bias for Windows recordings versus Mac recordings
- Negative bias for two language-family groups

## Limitations / Notes

- The scorer still shows subgroup bias tied to recording environment and some language-family effects.
- The paper explicitly links the Windows effect to lower average audio quality and dataset imbalance, not to pronunciation itself.
- The text clearly gives Indo-Aryan as one biased language-family example, but the second flagged family is not recoverable with high confidence from text extraction alone because it is encoded mainly in a figure.
- Building this kind of scorer requires expensive human rating, diverse sampling, and fairness analysis; it is not a lightweight CAPT-style benchmark recipe.

## Relevance To Peacock

- Very relevant if Peacock needs pronunciation scoring that is defensible as an assessment construct rather than just a proxy model.
- Strong evidence that dataset design and rating rubric quality matter as much as model choice.
- The fairness section is especially useful for Peacock if the end goal is real evaluation or feedback at scale, not just benchmark performance on a single L1 read-aloud corpus.
