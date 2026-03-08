---
title: "An Effective Hierarchical Graph Attention Network Modeling Approach for Pronunciation Assessment"
authors:
  - "Bi-Cheng Yan"
  - "Berlin Chen"
citation_author: "Yan"
year: 2024
doi: "10.1109/TASLP.2024.3449111"
pages: "3974-3985"
source_pdf: "paper.pdf"
extraction_method: "Manually curated from the local PDF, with nearby OCR text used selectively where needed to recover implementation details and high-confidence result summaries."
extracted_at: "2026-03-07T20:02:23-08:00"
llm_friendly: true
---

# An Effective Hierarchical Graph Attention Network Modeling Approach for Pronunciation Assessment

## Metadata
- Authors: Bi-Cheng Yan, Berlin Chen
- Year: 2024
- Venue: IEEE/ACM Transactions on Audio, Speech, and Language Processing
- DOI: 10.1109/TASLP.2024.3449111
- Pages: 3974-3985
- Task: Multi-aspect, multi-granular automatic pronunciation assessment

## TL;DR
This paper replaces flat sequence modeling for pronunciation assessment with a hierarchical heterogeneous graph over phones, words, and the utterance node. The core claim is that explicit graph message passing plus aspect attention better captures both linguistic hierarchy and correlations among scoring aspects.

On Speechocean762, HierGAT improves over strong parallel baselines and remains competitive when enhanced with SSL features. The paper reports especially strong gains at word and utterance levels.

## Abstract
The authors argue that common APA models treat phone-level features as a flat sequence and predict all scores in parallel, which underuses utterance hierarchy and ignores explicit relations among scoring aspects. HierGAT addresses this by representing each utterance as a hierarchical graph with phone, word, and utterance nodes, then learning representations through graph attention and dedicated aspect-attention modules.

## Research Question
Can a hierarchical graph attention model that explicitly encodes phone-word-utterance structure and aspect relatedness outperform existing parallel APA models?

## Method
- Input representation:
- GOP features derived from forced alignment and phone posteriors.
- Phone-level energy statistics.
- Phone duration.
- Phone-averaged log Mel-filterbank features.
- Optional concatenation with SSL-based features such as wav2vec 2.0, HuBERT, and WavLM.
- Graph construction:
- Phone nodes connect within the same word.
- Phone-to-word edges connect each phone to its parent word.
- Word nodes are fully connected to model inter-word relations.
- Word-to-utterance edges connect all words to one utterance supernode.
- Modeling:
- Stacked graph attention layers perform hierarchical message passing across these node types.
- Separate regressors predict phone-, word-, and utterance-level scores.
- An aspect-attention module models dependencies among scoring aspects.

## Data
- Dataset: Speechocean762.
- Annotation setting: five human raters per sample; utterance-, word-, and phone-level pronunciation scores.
- Scores are normalized to the 0-2 scale used by earlier APA work.
- The paper compares GOP-only systems and GOP+SSL systems.
- The local PDF also documents the feature recipe explicitly: 84-dimensional GOP, 7-dimensional energy statistics, 1-dimensional duration, and 80-dimensional log Mel-filterbank features at phone level.

## Results
- With GOP-based features only, the paper reports that HierGAT outperforms:
- LSTM by an average 9.94% across pronunciation assessment tasks.
- GOPT by an average 8.28%.
- HiPAMA by up to 5.47%, while keeping a top-tier phone-level accuracy score.
- Relative to GFR, the paper reports an average PCC improvement of 3.56% on most word- and utterance-level tasks while staying comparable at phone level.
- When SSL features are added on top of GOP features, HierGAT reports additional average gains of:
- 4.30% on phone-level tasks.
- 3.20% on word-level tasks.
- 3.26% on utterance-level tasks.
- Ablation findings:
- Removing the hierarchical graph layer degrades performance across linguistic levels.
- Removing aspect attention hurts most at higher linguistic levels, especially utterance scoring.
- A 3-layer, single-head GAT is reported as the best tradeoff; more heads do not help on this dataset.

## Limitations / Notes
- The local OCR did not recover every cell of the large main results table cleanly, so this note keeps to the high-confidence summary statistics stated in the paper text instead of inventing exact per-cell values.
- The method still depends on forced alignment and engineered pronunciation features; it is not a fully end-to-end raw-audio system.
- The evidence is from a single benchmark, Speechocean762, which is dominated by Mandarin L2 English.
- Some reported gains are expressed as percentage improvements in the paper text rather than simple absolute PCC deltas, so they should be read in that form.

## Relevance To Peacock
- This is highly relevant if Peacock needs a structured assessment model rather than a flat phone-sequence regressor.
- The hierarchical graph view is a strong template for combining phone-, word-, and utterance-level evidence in one scorer.
- The paper also suggests that explicit aspect interaction modeling is worth keeping, especially for utterance-level ratings such as fluency, prosody, and total score.
