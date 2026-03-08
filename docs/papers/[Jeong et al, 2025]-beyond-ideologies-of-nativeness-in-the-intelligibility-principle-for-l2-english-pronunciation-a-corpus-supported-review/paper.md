---
title: "Beyond Ideologies of Nativeness in the Intelligibility Principle for L2 English Pronunciation: A Corpus-Supported Review"
authors:
  - "Hyeseung Jeong"
  - "Stephanie Lindemann"
citation_author: "Jeong and Lindemann"
year: 2025
doi: "10.1016/j.system.2025.103599"
pages: 12
source_pdf: "paper.pdf"
extraction_method: "Manually curated from the local PDF, with section-level summarization and cleanup of extraction artifacts."
extracted_at: "2026-03-07"
llm_friendly: true
---

# Beyond Ideologies of Nativeness in the Intelligibility Principle for L2 English Pronunciation: A Corpus-Supported Review

## Metadata

- Type: Corpus-supported review / discourse analysis.
- Venue: *System*.
- Topic: Whether research framed around the "intelligibility principle" still reproduces native-speaker bias.

## TL;DR

The paper argues that a large part of L2 pronunciation research still treats native-like speech as the hidden standard even when it claims to prioritize intelligibility. Using corpus-based keyword analysis, the authors show that terms like `errors` and `comprehensibility` dominate the reviewed literature and often shift responsibility for communication problems onto L2 speakers. Their practical recommendation is to move away from prescribed native norms, emphasize mutual responsibility for intelligibility, and expose learners to a wider range of Englishes.

## Abstract

This article reviews published L2 English pronunciation research built around the intelligibility principle and argues that it remains shaped by ideologies of nativeness. The authors focus especially on how researchers discuss `errors` and `comprehensibility`, showing that both constructs can preserve native-speaker benchmarks and obscure the mutual, context-dependent nature of communication. The paper ends by proposing more equitable approaches to both research and teaching.

## Research Question

To what extent do ideologies of nativeness persist in intelligibility-principle research on L2 English pronunciation, and what alternatives would better support equitable research and teaching?

## Method

- Corpus-supported discourse analysis rather than meta-analysis of outcomes.
- Two corpora were compiled from 1995-2023 publications:
  - Target corpus: 49 articles aligned with the intelligibility principle and evaluating L2 speech relative to native speakers.
  - Reference corpus: 48 articles from ELF / EIL / World Englishes traditions that more explicitly challenge native-speaker benchmarking.
- Reported corpus sizes:
  - Target corpus: 339,298 tokens.
  - Reference corpus: 320,267 tokens.
- Tooling:
  - `AntConc 4.2.0`
  - Log-likelihood keyness analysis with a high threshold (`alpha = 0.0001`).
- Main analytic focus:
  - Top keywords and collocates.
  - Close reading of discourse around `errors` and `comprehensibility`.

## Data

- No speech dataset or learner intervention is introduced.
- The "data" are published research articles:
  - 49 target-corpus papers.
  - 48 reference-corpus papers.
- The target corpus starts with Munro and Derwing (1995) and tracks three decades of downstream work.

## Results

- In the target corpus, `errors` is the top keyword.
  - The authors argue that this reflects a persistent habit of labeling L2 pronunciation features as deficits relative to idealized native norms.
  - They note that such labeling appears even when the speakers being described are otherwise highly proficient.
- `Comprehensibility` is the second most prominent keyword.
  - The paper argues that many studies operationalize comprehensibility in place of intelligibility, even though the two are not equivalent.
  - In their collocation analysis, `accentedness` is the strongest collocate of `comprehensibility`, while `intelligibility` is only eighth.
- The review argues that this matters because comprehensibility ratings may track nativeness and listener ideology as much as communicative success.
  - The paper cites prior work showing weak or nonexistent correlations between intelligibility and comprehensibility in some listener populations.
  - It also cites several studies where comprehensibility correlates strongly with accentedness.
- Negative keywords in the reference corpus include `ELF`, `international`, `varieties`, `Englishes`, and `mutual`.
  - The authors interpret this as evidence that the comparison literature more often frames communication as plural, interactional, and jointly managed.
- Main recommendations:
  - Stop treating pronunciation differences as `errors` by default; use more neutral language such as `features`.
  - Do not treat listener ratings as transparent measures of speaker ability.
  - Study repair, accommodation, and interactional success rather than only speaker deviation from norms.
  - In teaching, avoid fixed GA/RP targets and build learners' perceptual repertoire across diverse English varieties.

## Limitations / Notes

- This is an interpretive review based on discourse patterns, not a new pronunciation experiment.
- Its claims about bias depend partly on how the authors defined the target and reference corpora.
- The paper is strongest as a critique of framing and terminology, not as a direct test of which teaching approach improves learner outcomes.
- That said, the critique is concrete enough to affect how downstream systems should define labels, targets, and evaluation metrics.

## Relevance To Peacock

- Very relevant to how pronunciation feedback is framed.
- Suggests Peacock should avoid equating "better pronunciation" with "more native-like pronunciation" unless that target is explicitly required.
- Supports designing evaluation around communicative effectiveness, listener adaptation, and repair strategies rather than accent reduction alone.
- Also matters for annotation language: labels such as `error` may encode a stronger normative claim than the data justify.
