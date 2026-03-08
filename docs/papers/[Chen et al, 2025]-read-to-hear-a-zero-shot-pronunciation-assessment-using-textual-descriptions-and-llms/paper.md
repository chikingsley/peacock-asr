---
title: "Read to Hear: A Zero-Shot Pronunciation Assessment Using Textual Descriptions and LLMs"
authors: "Yu-Wen Chen; Melody Ma; Julia Hirschberg"
citation_author: "Chen et al"
year: 2025
doi: "10.18653/v1/2025.emnlp-main.134"
pages: 13
source_pdf: "paper.pdf"
extraction_method: "manual-curated from paper.pdf and local arXiv source"
extracted_at: "2026-03-07"
llm_friendly: true
---

# Title

Read to Hear: A Zero-Shot Pronunciation Assessment Using Textual Descriptions and LLMs

## Metadata

- Authors: Yu-Wen Chen, Melody Ma, Julia Hirschberg
- Citation author: Chen et al
- Venue: EMNLP 2025, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing
- Pages: 13
- DOI: 10.18653/v1/2025.emnlp-main.134
- Affiliation: Department of Computer Science, Columbia University
- Core task: zero-shot English pronunciation assessment with interpretable feedback
- Main metric: Pearson correlation coefficient (PCC)

## TL;DR

TextPA assesses pronunciation without supervised audio-score training by converting speech into text-like cues: ASR transcript, recognized IPA sequence, recognized CMU phoneme sequence with pause markers, and a separate IPA similarity score. An LLM then predicts accuracy and fluency and gives reasoning.

The method is most compelling on open-ended speech. On the MultiPA dataset, TextPA with `gpt-4o-mini` outperforms both `gpt-4o-mini-audio` and a supervised acoustic baseline on accuracy, and a simple fusion of TextPA with the acoustic model is best overall. On short scripted read-aloud speech, TextPA stays competitive with zero-shot audio models but remains clearly behind in-domain supervised systems.

## Abstract

The paper argues that pronunciation assessment does not have to be done purely with acoustic models trained on human scores. Instead, it proposes turning speech into human-readable symbolic descriptions that an LLM can interpret. The main claim is that this produces useful zero-shot accuracy and fluency scores, plus natural-language reasoning, at lower cost than audio-token-based large audio-language models. The authors also show that the text-based system can complement a conventional audio model on out-of-domain data.

## Research Question

Can an LLM perform useful zero-shot pronunciation assessment from textualized speech cues, rather than raw audio, and can this provide a cheaper and more explainable alternative or complement to supervised audio-score models?

## Method

TextPA has four main steps:

1. Convert speech into textual cues using pretrained components.
2. Prompt an LLM to score pronunciation accuracy and fluency on a 1-5 scale and produce reasoning.
3. Compute a separate pronunciation-accuracy signal by matching recognized IPA against a canonical IPA sequence derived from the transcript.
4. Average the normalized LLM accuracy score with the normalized IPA match score.

Key components used in the paper:

- Transcript: Whisper `large-v3-en`
- Recognized IPA sequence: a wav2vec2-based phoneme recognition model from Xu et al. (2021)
- Recognized CMU phoneme sequence plus pause durations: Charsiu predictive aligner
- Transcript-to-canonical-IPA mapping: Phonemize with `EspeakBackend("en-us")`
- Similarity algorithm for IPA matching: Smith-Waterman local alignment

Why each cue matters:

- Transcript: semantic incoherence or filler/repetition can indicate poor pronunciation or disfluency.
- IPA / CMU sequences: expose phone-level mismatches that ASR transcripts may hide.
- CMU pause markers: help the LLM reason about fluency.
- IPA match score: catches articulation mismatches the LLM may miss.

## Data

- Speechocean762:
  Scripted English read-aloud speech from native Mandarin speakers. The dataset has 5,000 utterances from 250 speakers with sentence-, word-, and phoneme-level annotations. The paper uses only the 2,500-utterance test split because TextPA is zero-shot. Typical utterance length is about 2-20 seconds.
- MultiPA:
  50 open-ended English speech clips from about 20 anonymous users interacting with a dialogue chatbot. Clips are about 10-20 seconds long.
- Both datasets:
  Focus on English speech by native Mandarin speakers. The paper cites a CC BY 4.0 license.
- Evaluation:
  Sentence-level accuracy and fluency are the main targets; prosody is explored separately and performs poorly.

## Results

- MultiPA (free speech):
  `TextPA (gpt-4o-mini)` reaches `0.728` PCC on accuracy and `0.650` on fluency.
- MultiPA comparison:
  `GPT-4o-mini-audio` gets `0.674 / 0.648`, and the supervised `MultiPA model` gets `0.618 / 0.683`.
- MultiPA fusion:
  A simple normalized average of `MultiPA model + TextPA (gpt-4o-mini)` reaches `0.769 / 0.784`, the best result reported for that dataset.
- Speechocean (scripted speech):
  `TextPA (gemini-2.0-flash)` reaches `0.532` accuracy and `0.557` fluency, while `Gemini-2.0-flash-audio` gets `0.562 / 0.556`.
- Speechocean in-domain references:
  The supervised `MultiPA model` gets `0.705 / 0.772`; prior cited systems report `0.72` accuracy or `0.795` fluency. TextPA is competitive for zero-shot use, but not close to trained in-domain systems.
- Ablation:
  Transcript-only prompting is weakest. Adding IPA and CMU cues helps, and IPA match scoring is a strong extra signal for accuracy. CMU sequence matching alone performs poorly.
- ASR quality:
  A weaker ASR model (`Whisper tiny`) helps when using transcript-only prompting, but the stronger ASR model (`large-v3-en`) works better inside the full TextPA pipeline.
- Prompting:
  Detailed rubric text is not consistently better than simpler instructions and can hurt both performance and cost.
- Reasoning quality:
  In a manual annotation study, the paper reports that about 76% of `gpt-4o-mini` reasoning content and more than 90% of `gemini-2.0-flash` reasoning content was labeled correct or constructive.
- Prosody:
  The text-only prosody attempt is weak. On MultiPA, adding prosody as an explicit target drops accuracy and fluency and yields only `0.243` PCC on prosody.

## Limitations / Notes

- The approach is much stronger for accuracy and fluency than for prosody.
- Results depend on both ASR output quality and LLM behavior; the paper states results are from a single run with default API settings.
- The authors note occasional hallucinated or irrelevant reasoning.
- Budget constraints limited evaluation of stronger LLMs and broader audio-model comparisons.
- The paper does not model accent variation and explicitly warns about overemphasizing General American or native-like pronunciation over intelligibility.
- Inference from the method description: because final accuracy uses min-max normalization across the test set, deployment for single-utterance or streaming scoring would need an additional calibration choice that the paper does not fully specify.

## Relevance To Peacock

This paper is directly relevant if Peacock wants low-cost, interpretable pronunciation feedback without needing a large scored-audio training set. The most reusable idea is not the exact LLM choice, but the decomposition of speech into transcript, phone sequence, and pause cues, then using an LLM for explanation and a symbolic phone-match score for extra stability.

It also supports a hybrid strategy. The paper's strongest practical result is that TextPA complements an audio-trained model on out-of-domain free speech, which is likely more useful for real learner interactions than short scripted prompts. The main cautions for Peacock are weak prosody handling, possible reasoning hallucinations, accent-bias risk, and unclear online calibration for the normalized IPA-match component.
