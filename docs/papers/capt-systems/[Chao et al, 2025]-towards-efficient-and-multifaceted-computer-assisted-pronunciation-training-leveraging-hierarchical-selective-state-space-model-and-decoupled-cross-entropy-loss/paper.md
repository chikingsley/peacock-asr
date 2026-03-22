---
title: "Towards Efficient and Multifaceted Computer-assisted Pronunciation Training Leveraging Hierarchical Selective State Space Model and Decoupled Cross-entropy Loss"
authors:
  - "Fu-An Chao"
  - "Berlin Chen"
citation_author: "Chao et al"
year: 2025
doi: null
pages: 15
source_pdf: "paper.pdf"
extraction_method: "Manual rewrite from the local paper.pdf, using the nearby extracted markdown only to verify table values and section boundaries."
extracted_at: "2026-03-07T20:03:00-08:00"
llm_friendly: true
---

# Title

Towards Efficient and Multifaceted Computer-assisted Pronunciation Training Leveraging Hierarchical Selective State Space Model and Decoupled Cross-entropy Loss

## Metadata

- Authors: Fu-An Chao, Berlin Chen
- Citation author: Chao et al
- Year: 2025
- DOI: Not stated in the local PDF
- Pages: 15
- Source PDF: `paper.pdf`
- Venue/status: not stated in the local PDF; the document reads like a manuscript or preprint

## TL;DR

This paper argues that CAPT systems should not split pronunciation scoring (`APA`) and mispronunciation diagnosis (`MDD`) into separate pipelines. It proposes `HMamba`, a hierarchical bidirectional Mamba-based model that predicts phone-, word-, and utterance-level pronunciation scores while also diagnosing phone errors.

The companion loss, `deXent`, is designed to fix a recurring MDD problem in text-prompt-aware models: they tend to over-copy canonical phones, which yields high precision but poor recall. On `speechocean762`, the combined system is stronger than the paper's baselines on most pronunciation scores and raises MDD `F1` from `41.50%` to `63.85%`.

## Abstract

The paper targets full-featured computer-assisted pronunciation training rather than isolated scoring or diagnosis. The proposed system, `HMamba`, jointly handles automatic pronunciation assessment and mispronunciation detection/diagnosis in one hierarchy-aware architecture. It also introduces a decoupled cross-entropy loss (`deXent`) to improve supervised learning for phone-level error detection. The reported experiments on `speechocean762` show gains on multi-granular pronunciation assessment and a substantial improvement in MDD performance over a strong joint baseline.

## Research Question

Can one efficient model jointly provide multi-granular pronunciation assessment and phone-level mispronunciation diagnosis, and can a specialized loss improve the precision/recall balance for text-prompt-aware MDD?

## Method

- Input consists of learner speech plus a canonical phone sequence derived from the prompt, aligned to phone boundaries.
- Acoustic features combine GOP-style features, phone duration and energy statistics, and self-supervised speech representations from `wav2vec 2.0`, `HuBERT`, and `WavLM`.
- Phonological features include canonical phone embeddings, absolute position embeddings, and relative phone-position embeddings within each word, including silence categories.
- `HMamba` uses a bottom-up hierarchy with bidirectional Mamba blocks at phone, word, and utterance levels.
- The APA branch predicts phone-level accuracy, word-level accuracy/stress/total, and utterance-level accuracy/completeness/fluency/prosody/total.
- The MDD branch treats diagnosis as free phone recognition, then compares the predicted phones against the canonical sequence to infer mispronunciations.
- `deXent` rebalances the MDD objective so the model is less biased toward simply repeating the canonical phones, which improves recall and overall `F1`.

## Data

- Dataset: `speechocean762`.
- Size: `5,000` English read-aloud recordings from `250` Mandarin L2 learners, split evenly into train and test.
- APA labels: multi-granular pronunciation scores, each rated by five experts using standardized rubrics.
- MDD labels: phone-level mispronunciation annotations over `46` phones (`39` CMU phones, `6` L2-specific phones, and `[unk]`), plus a `[DEL]` token for deletions. The dataset does not contain insertion errors.
- Evaluation: five independent runs with different seeds; the paper reports average metrics.

## Results

- `HMamba` is the best APA model among the reported baselines on nearly every listed metric.
- Phone-level APA: `MSE = 0.062`, `PCC = 0.739`.
- Word-level APA `PCC`: accuracy `0.708`, stress `0.366`, total `0.718`.
- Utterance-level APA `PCC`: accuracy `0.807`, completeness `0.278`, fluency `0.848`, prosody `0.843`, total `0.829`.
- The closest strong multi-granular baseline in the table, `3MH`, reaches utterance total `0.811`, so `HMamba` improves that to `0.829`.
- Hierarchical structure matters: `HMamba` beats the non-hierarchical `LMamba` and `PMamba` variants across all reported APA aspects.
- MDD results improve sharply over `Joint-CAPT-L1`: `HMamba` reaches precision `64.35%`, recall `63.41%`, `F1 = 63.85%`, `PER = 2.72%`, while `Joint-CAPT-L1` reaches `26.70%`, `91.40%`, `41.50%`, and `9.93%`.
- `deXent` changes the precision/recall tradeoff relative to ordinary cross-entropy: `Xent` gives precision `77.07%`, recall `38.60%`, `F1 = 51.40%`, `PER = 2.53%`; `deXent` with `alpha = 0.7` gives `64.35%`, `63.41%`, `63.85%`, `2.72%`; and `alpha = 0.9` raises recall to `71.12%` with precision `57.74%` and `F1 = 63.73%`.

## Limitations / Notes

- The paper explicitly notes limited accent diversity: all learners in the dataset are Mandarin L2 speakers.
- The evaluation is confined to read-aloud CAPT, so generalization to spontaneous or open-ended speech is unproven.
- The authors also note limited interpretability: the model imitates expert annotations but does not ground its outputs in explicit rubrics or external knowledge.
- Completeness is the weakest major APA score for `HMamba`; the paper suggests the model may emphasize phone accuracy and MDD more than word omission/completeness.
- Venue and DOI could not be verified from the local PDF.

## Relevance To Peacock

This paper is directly relevant to Peacock's pronunciation-training work. The strongest idea is the unified treatment of `APA` and `MDD`: phone-, word-, and utterance-level feedback are modeled together rather than as separate products.

`deXent` is also practically relevant. If Peacock wants phone-level diagnostic feedback that is usable for learners, not just high-level scores, balancing precision and recall matters more than optimizing a single cross-entropy objective. The main caution is external validity: the evidence here is still limited to Mandarin-accented English read-aloud speech.
