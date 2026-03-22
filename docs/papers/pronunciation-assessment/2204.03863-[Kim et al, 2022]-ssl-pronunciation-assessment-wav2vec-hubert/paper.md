---
title: "Automatic Pronunciation Assessment using Self-Supervised Speech Representation Learning"
authors:
  - "Eesung Kim"
  - "Jae-Jin Jeon"
  - "Hyeji Seo"
  - "Hoon Kim"
citation_author: "Kim et al"
year: 2022
doi: null
pages: "1411-1415"
source_pdf: "paper.pdf"
extraction_method: "Manual rewrite from the local LaTeX source (main.tex) and bibliography (main.bbl); no auto-generation."
extracted_at: "2026-03-22"
llm_friendly: true
---

# Automatic Pronunciation Assessment using Self-Supervised Speech Representation Learning

## Metadata

- Authors: Eesung Kim, Jae-Jin Jeon, Hyeji Seo, Hoon Kim
- Citation author: Kim et al
- Year: 2022
- Venue: Interspeech 2022
- ArXiv identifier: 2204.03863
- Pages: 1411-1415
- DOI: Not stated in the local source bundle
- Source: local LaTeX source (`main.tex`) and `main.bbl`
- Task: utterance-level automatic pronunciation scoring for non-native English speech
- Model family: fine-tuned `wav2vec 2.0` and `HuBERT` encoders with layer-wise representations

## TL;DR

The paper fine-tunes pre-trained SSL speech encoders on learner speech with CTC, then uses layer-averaged transformer representations plus the reference text to predict pronunciation scores with a BLSTM scorer. On both the in-house Korean ESL children corpus and Speechocean762, the fine-tuned SSL models beat GOP-style and handcrafted-feature baselines, and `HuBERT Large` is the best model reported.

## Abstract

The authors argue that self-supervised speech representations can capture pronunciation-relevant structure well enough to improve automatic pronunciation assessment. Their method adapts pre-trained `wav2vec 2.0` and `HuBERT` encoders to non-native speech, extracts representations from all transformer layers, and combines the acoustic features with text information in a BLSTM scoring module. The main empirical result is that fine-tuned SSL models outperform both traditional baselines and frozen SSL encoders on sentence-level pronunciation scoring.

## Research Question

Can SSL encoders, after ASR-style fine-tuning on learner speech, produce better utterance-level pronunciation scores than GOP-style and handcrafted acoustic features?

## Method

- Fine-tune `wav2vec 2.0` and `HuBERT` on non-native speech with CTC so the encoder adapts to the target pronunciation domain.
- Extract hidden states from every transformer layer and average them to build a layer-wise context representation.
- Feed the acoustic representation into a BLSTM scoring branch.
- Encode the reference transcript with a text embedding branch and another BLSTM, then combine audio and text context with global average pooling and a final linear regressor.
- Compare pre-trained SSL, fine-tuned SSL, and conventional baselines such as GOP, aggregate features, and sequence features.
- The experiments use `wav2vec2-base-960h`, `wav2vec2-large-960h`, `wav2vec2-large-robust`, `HuBERT-base-ls960h`, and `HuBERT-large-ls960h` checkpoints from Fairseq.
- The setup uses a speaker-level 10-fold split and selects the fine-tuned checkpoint with the lowest development WER.

## Data

- KESL: 17,800 utterances from 300 Korean ESL children aged 10-12, with five expert sentence-level ratings on a 1-5 scale.
- KESL scoring: the evaluation score is the average of the five expert ratings.
- Speechocean762: 5,000 utterances from 250 non-native speakers, with sentence-level labels for accuracy, completeness, fluency, and prosody.
- The paper evaluates KESL holistic scores and Speechocean762 fluency and prosody scores.
- Audio is sampled at 16 kHz.

## Results

- Traditional baselines: `GOP` 0.63/0.65/0.64; `Agg + Seq` 0.55/0.51/0.59; `Agg + Seq + GOP` 0.64/0.67/0.66.
- Pre-trained SSL already beats the handcrafted baselines: `wav2vec2 Robust` reaches 0.76/0.73/0.73; `HuBERT Large` reaches 0.75/0.75/0.74.
- Fine-tuning improves every SSL family: `wav2vec2 Robust` rises to 0.79/0.75/0.74 and `HuBERT Large` rises to 0.82/0.78/0.77, which are the best scores in the paper.
- Ablation shows all-layer averaging is better than using only the local convolutional output (0.56/0.60/0.62) or a single transformer layer such as Layer 20 (0.81/0.76/0.76).
- The main pattern is that higher transformer layers carry more pronunciation-relevant information, and ASR-style fine-tuning helps more than using frozen SSL features.

## Limitations / Notes

- The method is sentence-level and transcript-conditioned, so it is not an open-response or transcript-free scorer.
- The evaluation is limited to two datasets, one of which is in-house, so generalization is not fully established.
- The paper reports PCC as the main metric, but not calibration or fine-grained error analysis.
- The approach is stronger for scoring than for diagnostic feedback, since it does not directly produce phone-level explanations.

## Relevance To Peacock

- This is a strong reference for using SSL encoders in pronunciation scoring.
- The all-layer averaging result is useful if we want a compact scorer without committing to a single transformer layer.
- The paper also reinforces a transcript-conditioned scoring design, which fits closed-response CAPT workflows better than open-ended assessment.
