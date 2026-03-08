---
title: "Preserving Phonemic Distinctions for Ordinal Regression: A Novel Loss Function for Automatic Pronunciation Assessment"
authors:
  - "Bi-Cheng Yan"
  - "Hsin-Wei Wang"
  - "Yi-Cheng Wang"
  - "Jiun-Ting Li"
  - "Chi-Han Lin"
  - "Berlin Chen"
citation_author: "Yan"
year: 2023
doi: "10.1109/ASRU57964.2023.10389781"
pages: "1-7"
source_pdf: "paper.pdf"
extraction_method: "Manually curated from the local PDF, using the nearby OCR text only to recover clean table values and section boundaries."
extracted_at: "2026-03-07T20:02:23-08:00"
llm_friendly: true
---

# Preserving Phonemic Distinctions for Ordinal Regression: A Novel Loss Function for Automatic Pronunciation Assessment

## Metadata
- Authors: Bi-Cheng Yan, Hsin-Wei Wang, Yi-Cheng Wang, Jiun-Ting Li, Chi-Han Lin, Berlin Chen
- Year: 2023
- Venue: IEEE ASRU 2023
- DOI: 10.1109/ASRU57964.2023.10389781
- Pages: 1-7
- Task: Automatic pronunciation assessment (APA)

## TL;DR
The paper argues that standard regression losses such as MSE preserve score ordinality but collapse different phoneme categories with the same score too tightly. To fix that, it adds a phoneme-aware regularizer to MSE, called phonemic contrast ordinal (PCO) loss.

When plugged into GOPT on Speechocean762, PCO improves phone-level PCC and most word- and utterance-level metrics, but word-level stress gets worse and some gains remain uneven across aspects.

## Abstract
This work targets regression-based APA systems that predict continuous pronunciation scores. The authors claim that plain MSE encourages representations to cluster by score while discarding phonemic distinctions. Their proposed PCO loss adds a phoneme-distinct regularizer with two effects: pushing different phoneme categories apart and tightening same-phoneme representations in an ordinally aware way.

## Research Question
Can an ordinal regression loss that explicitly preserves phonemic distinctions improve pronunciation assessment beyond a vanilla MSE-trained regression model?

## Method
- Base model: GOPT, a Transformer-based multi-aspect, multi-granular pronunciation assessment model.
- Proposed loss: PCO loss = MSE plus a phoneme-distinct regularizer.
- The regularizer has two pieces:
- Phonemic distinction: separates embeddings from different phoneme categories.
- Ordinal tightness: pulls same-phoneme embeddings together while respecting score order.
- The paper applies the new loss mainly through phone-level intermediate representations and associated phone-level scores.
- Training setup otherwise follows the original GOPT configuration.

## Data
- Dataset: Speechocean762.
- Contents: 5,000 English recordings from 250 Mandarin L2 learners.
- Split: 2,500 training utterances and 2,500 test utterances.
- Annotation: multi-granular pronunciation scores from five expert raters, with the median used as the final label.
- Features: 84-dimensional GOP features from a TDNN-F acoustic model trained with the Kaldi Librispeech recipe.
- Score normalization: word- and utterance-level scores are mapped to the same 0-2 scale as phone scores.

## Results
- Phone level:
- GOPT baseline: MSE 0.085, PCC 0.612.
- PCO loss: MSE 0.083, PCC 0.622.
- Word level PCC:
- GOPT baseline: accuracy 0.533, stress 0.291, total 0.549.
- PCO loss: accuracy 0.558, stress 0.250, total 0.573.
- Utterance level PCC:
- GOPT baseline: accuracy 0.714, completeness 0.155, fluency 0.753, prosody 0.760, total 0.742.
- PCO loss: accuracy 0.727, completeness 0.359, fluency 0.763, prosody 0.763, total 0.752.
- Relative to the compared baselines in the readable table:
- PCO beats the vanilla GOPT model across every reported metric except word-level stress.
- It also beats HiPAMA on phone PCC and on utterance fluency/prosody, but not on all metrics.
- The visualization section supports the main claim qualitatively: embeddings become more separable by phoneme category while still reflecting score order.

## Limitations / Notes
- The main loss design is centered on phone-level representations; the paper does not show a broader reformulation of the entire architecture.
- Word-level stress degrades from 0.291 to 0.250, which the authors attribute partly to label imbalance.
- Completeness remains lower than the strongest competing number in the comparison table.
- The evaluation is limited to one benchmark and one feature pipeline.

## Relevance To Peacock
- The paper is useful if Peacock needs regression losses that keep phoneme identity information from being washed out by scalar target fitting.
- The idea is especially relevant for pronunciation scoring, confidence regression, or any setting where ordinal labels coexist with categorical phone identity.
- It is a loss-level improvement, not a full end-to-end acoustic modeling recipe, so it is easiest to reuse inside an existing APA stack.
