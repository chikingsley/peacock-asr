---
title: "Evaluating Automatic Pronunciation Scoring with Crowd-sourced Speech Corpus Annotations"
authors:
  - "Nils Hjortnæs"
  - "Daniel Dakota"
  - "Sandra Kübler"
  - "Francis Tyers"
citation_author: "Hjortnæs et al"
year: 2024
doi: null
pages: "67-77"
source_pdf: "paper.pdf"
extraction_method: "Manual rewrite from the local paper.pdf; the existing local paper.md extraction was used only as a noisy cross-check."
extracted_at: "2026-03-07T20:03:41-08:00"
llm_friendly: true
---

# Title

Evaluating Automatic Pronunciation Scoring with Crowd-sourced Speech Corpus Annotations

## Metadata

- Authors: Nils Hjortnæs, Daniel Dakota, Sandra Kübler, Francis Tyers
- Venue: NLP4CALL 2024, Linköping Electronic Conference Proceedings 211
- Pages: 67-77
- DOI: not stated in the local PDF
- Task: testing whether Common Voice vote annotations can stand in for pronunciation labels

## TL;DR

This paper is mostly a negative result. The authors try to use Common Voice upvotes and downvotes as weak pronunciation labels, then test pronunciation scorers against that signal.

The weak labels do not hold up. Downvotes are driven much more by audio quality, reading mistakes, and corpus noise than by pronunciation quality, so Common Voice voting is a poor substitute for a real pronunciation-annotated corpus.

## Abstract

The paper asks whether crowdsourced validation signals from Common Voice can be repurposed for pronunciation-scoring research. It uses ASR softmax outputs and distribution-comparison scores to predict whether a clip received any downvotes. Both the quantitative and qualitative analyses suggest that these votes are not reliable pronunciation labels and that dedicated pronunciation annotation is still necessary.

## Research Question

Can Common Voice's crowd upvote/downvote annotations be used as a practical proxy for pronunciation quality when building or evaluating automatic pronunciation-scoring methods?

## Method

- Use Common Voice English clips where many speakers read the same sentence.
- Treat clips with only upvotes as "expert" or acceptable references.
- Treat the presence of any downvote as a weak signal for pronunciation problems.
- Run audio through Coqui STT, modified to preserve the per-time-slice softmax distributions instead of only the final transcript.
- Align time-slice character distributions to the elicitation phrase with a modified Needleman-Wunsch procedure.
- Compare aligned character-level distributions between a reference clip and another speaker's clip using:
- baseline target-character probability
- Hellinger distance
- Jensen-Shannon divergence
- cross entropy
- Train an MLP classifier to predict whether a clip has any downvotes.

## Data

- Corpus: Common Voice English
- Sampling rule: 1,000 sentence sets with at least 10 different speakers per sentence
- Initial utterances: `34,105`
- Clips removed because Coqui could not process them cleanly: `9,061`
- Final usable clip count: `25,044`
- Pairwise comparisons generated: `511,532`
- Reduced comparison set for classification: `20,000`
- Train/test split for downvote detection: `15,000 / 5,000`
- ASR quality on the sampled Common Voice subset: WER `0.252`, CER `0.153`

## Results

- Predicting downvotes is hard even with the strongest setup.
- Reported downvote-detection accuracies:
- baseline scorer: `81.4%`
- Hellinger: `64.2%`
- cross entropy: `60.9%`
- Jensen-Shannon: `60.6%`
- The authors explicitly argue that `81.4%` is too low to treat as a reliable upper bound for this proxy task.
- Score distributions for clips with and without downvotes are nearly identical in aggregate.
- Qualitative inspection shows many downvotes correspond to:
- poor audio quality
- clipping or background noise
- misreadings
- dialect variation
- Conversely, some poor-quality clips receive no downvotes at all, so the weak labels are noisy in both directions.

## Limitations / Notes

- The entire setup assumes that downvotes correlate meaningfully with pronunciation quality; the paper's own results undermine that assumption.
- The ASR model is a general speech recognizer, not a pronunciation model fine-tuned with character-level pronunciation supervision.
- The paper studies English Common Voice only; it does not establish whether some low-resource settings might still get limited value from such proxy labels.
- Because most clips have very few votes, vote ratios are not stable enough to use as a richer target.

## Relevance To Peacock

This paper is useful mainly as a warning. If Peacock is tempted to use Common Voice validation votes, or similar crowd acceptance signals, as pronunciation labels, this study is evidence against doing that without much stronger filtering and relabeling.

The reusable technical pieces are secondary: character-level ASR confidence extraction and alignment-based comparison are interesting, but the core message is that proxy labels from generic crowd validation can easily measure recording quality or reading compliance instead of pronunciation quality.
