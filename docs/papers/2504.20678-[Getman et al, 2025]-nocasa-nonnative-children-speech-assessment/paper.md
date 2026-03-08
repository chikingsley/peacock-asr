---
title: "Non-native Children's Automatic Speech Assessment Challenge (NOCASA)"
authors:
  - "Yaroslav Getman"
  - "Tamás Grósz"
  - "Mikko Kurimo"
  - "Giampiero Salvi"
citation_author: "Getman et al."
year: 2025
doi: null
pages: 6
source_pdf: "paper.pdf"
extraction_method: "Manually curated from the local PDF, with section-level summarization and cleanup of extraction artifacts."
extracted_at: "2026-03-07"
llm_friendly: true
---

# Non-native Children's Automatic Speech Assessment Challenge (NOCASA)

## Metadata

- Type: Benchmark / shared-task paper.
- Venue: IEEE MLSP 2025 workshop challenge paper.
- Topic: Automatic pronunciation assessment for young L2 learners of Norwegian.

## TL;DR

NOCASA introduces a difficult low-resource benchmark for rating children's single-word Norwegian pronunciations on a 1-5 scale. After cleanup, the released task contains 7,857 training utterances and 1,460 test utterances from children aged 5-12. The best official baseline, a multitask wav2vec 2.0 system, reaches only 36.37% UAR, which makes the paper most useful as a benchmark statement about data scarcity, imbalance, and practical latency constraints.

## Abstract

The paper presents NOCASA, a shared task on automatic pronunciation assessment for non-native children learning Norwegian. It releases a rated single-word speech dataset together with two baselines, one traditional SVM system and one multitask wav2vec 2.0 system. The stated goal is to accelerate comparable research on low-resource, highly imbalanced child-speech pronunciation scoring.

## Research Question

How can a public benchmark help researchers build and compare automatic pronunciation assessment systems for young L2 learners under low-resource and class-imbalanced conditions?

## Method

- Challenge framing:
  - Predict a 1-5 pronunciation score ("stars") for each child utterance.
  - Primary metric: unweighted average recall (UAR).
  - Additional metrics: accuracy and mean absolute error (MAE).
- Official baselines:
  - `SVM` with `ComParE 16` OpenSMILE features (`6,373` features), one-vs-rest classification, and class weighting.
  - Multitask `wav2vec 2.0` based on a Norwegian ASR model, keeping the CTC head and adding a pronunciation-rating head.
- wav2vec 2.0 training details reported in the paper:
  - 20 epochs.
  - Batch size 10.
  - Peak learning rate `2e-4`.
  - 10% validation split during tuning, then retraining on full training data.
- The paper also analyzes latency, confidence, and interpretability of the baselines.

## Data

- Source corpus:
  - TeflonNorL2 / TEFLON-related child Norwegian pronunciation data.
- Population:
  - Children aged 5-12.
  - Includes native Norwegian speakers, beginner L2 learners of Norwegian, and children with no prior Norwegian exposure.
- Task unit:
  - Single-word pronunciations of `205` distinct Norwegian words.
- Initial split before cleanup:
  - Training: `10,334` recordings from `44` speakers.
  - Test: `1,930` recordings from `8` speakers.
- Final released split after removing score-0 samples and duplicate same-word attempts per speaker:
  - Training: `7,857` audio files.
  - Test: `1,460` audio files.
- Additional details:
  - All recordings are roughly 5-8 seconds long.
  - File names were randomized to hide speaker identity.
  - The split was designed to approximately preserve score, gender, age, and language-background distributions.

## Results

- Official baselines on the test set:
  - `SVM`: `22.14%` UAR, `32.74%` accuracy, `1.05` MAE.
  - `MT wav2vec 2.0`: `36.37%` UAR, `54.45%` accuracy, `0.55` MAE.
- Reported `95%` confidence interval for the wav2vec UAR: `34.30 - 38.81`.
- On high-scored utterances (scores 4 or 5), the multitask wav2vec ASR side achieved:
  - `10.63%` WER.
  - `4.10%` CER.
- Practical properties:
  - Reported inference time for the 300M wav2vec model was roughly `30-50 ms` for a 3-second audio clip on one GPU.
  - Additional seed runs gave similar ranges (`UAR 36-38`, `MAE 0.53-0.57`), which the authors interpret as reasonable robustness.
- Failure mode:
  - The wav2vec baseline never predicted the rarest `1-star` class.
- Shared-task outcome:
  - Two participant systems beat the baselines, reaching `42.52%` and `44.81%` UAR respectively.

## Limitations / Notes

- This is a challenge paper, so most of the value is in the benchmark and baselines, not in a theory of pronunciation learning.
- The dataset is small at the speaker level, especially on the test side.
- Label imbalance is severe and clearly hurts minority-class recall.
- The task is single-word Norwegian assessment, so it does not directly test sentence-level fluency or spontaneous speech.
- The local PDF does not show a DOI for this paper, so the frontmatter leaves `doi` as `null`.

## Relevance To Peacock

- Useful if Peacock needs child-speech or low-resource pronunciation scoring benchmarks.
- Reinforces that class imbalance handling is not optional for star-rating style pronunciation tasks.
- Shows the value of combining assessment quality with latency and explainability, especially for gamified products.
- Also suggests that prompt-aware models, which use the expected target word intelligently, may be an important design direction.
