---
title: "HiPPO: Exploring A Novel Hierarchical Pronunciation Assessment Approach for Spoken Languages"
authors:
  - "Bi-Cheng Yan"
  - "Hsin-Wei Wang"
  - "Fu-An Chao"
  - "Tien-Hong Lo"
  - "Yung-Chang Hsu"
  - "Berlin Chen"
citation_author: "Yan et al."
year: 2025
doi: null
pages: 14
source_pdf: "paper.pdf"
extraction_method: "manual-curated from local paper.pdf and nearby repository markdown"
extracted_at: "2026-03-07"
llm_friendly: true
---

# HiPPO: Exploring A Novel Hierarchical Pronunciation Assessment Approach for Spoken Languages

## Metadata

- Authors: Bi-Cheng Yan, Hsin-Wei Wang, Fu-An Chao, Tien-Hong Lo, Yung-Chang Hsu, Berlin Chen
- Year: 2025
- DOI: Not found in the local PDF
- Pages: 14
- Source PDF: `paper.pdf`
- Note: The paper evaluates "free-speaking" through a simulated setup built on Speechocean762, which is originally a read-aloud corpus.

## TL;DR

- Tries to move APA from scripted read-aloud input to unscripted spoken-language input.
- Uses ASR transcripts plus a hierarchical Conv-LLaMA model, a contrastive ordinal regularizer (CONO), and curriculum learning.
- On simulated free-speaking Speechocean762, HiPPO reaches phone PCC 0.480 and utterance-total PCC 0.754, and both CONO and curriculum learning materially help.

## Abstract

The paper argues that most APA systems assume access to a reference text, which makes them ill-suited for genuine free-speaking assessment. HiPPO reframes spoken-language pronunciation assessment by first transcribing learner speech with ASR, then applying a hierarchical scoring model over phone, word, and utterance levels. A contrastive ordinal regularizer is used to make features more score-discriminative and more robust to ASR noise, and curriculum learning is used to bridge from read-aloud training to harder free-speaking-style input.

## Research Question

Can a pronunciation assessment model score unscripted learner speech using only the learner’s spoken audio, while remaining robust to ASR errors and preserving useful multi-granular structure?

## Method

- HiPPO first uses an ASR system to generate transcripts from learner speech, then derives phone sequences via G2P conversion.
- Because the benchmark is read-aloud, the authors simulate free-speaking by withholding the reference text at inference time and relying on ASR output.
- Core model: a hierarchical phone/word/utterance architecture built from `Conv-LLaMA` blocks.
- Training adds `CONO`, a contrastive ordinal regularizer intended to produce score-discriminative features and reduce sensitivity to ASR errors.
- A curriculum-learning schedule gradually shifts training from easier read-aloud style supervision toward harder free-speaking-style supervision.

## Data

- Dataset: `Speechocean762`
- 5,000 English utterances from 250 Mandarin L2 learners
- Split: 2,500 train / 2,500 test
- Main reported setting: simulated free-speaking built from the read-aloud corpus
- Read-aloud results are also reported as a comparison setting
- The paper reports Whisper-large-v3 word error rates of 19.22% on train and 17.49% on test

## Results

- Simulated free-speaking:
- `HiPPO`: phone MSE 0.202, phone PCC 0.480
- Word-level: accuracy 0.520, total 0.521
- Utterance-level: accuracy 0.733, fluency 0.806, prosody 0.797, total 0.754
- In the same setting, the strongest listed structural baseline is weaker overall:
- `Parallel-LLaMA`: phone PCC 0.345, utterance-total 0.748
- `Hier-LLaMA`: phone PCC 0.328, utterance-total 0.724
- Ablations show the add-ons matter:
- `w/o CONO`: phone PCC 0.448, utterance-total 0.743
- `w/o CL`: phone PCC 0.331, utterance-total 0.728
- Read-aloud setting:
- `HiPPO*` (reported without curriculum learning and CONO) still performs strongly with phone PCC 0.657 and utterance-total PCC 0.816
- Azure Pronunciation Assessment is better on utterance-level prosody alone (0.842), according to the paper’s table

## Limitations / Notes

- The "free-speaking" evaluation is simulated from a read-aloud dataset, not a native open-response corpus.
- The learner population is Mandarin-only, so accent diversity is limited.
- ASR errors remain a real bottleneck; the paper explicitly reports noticeable degradation under worse WER conditions.
- The model focuses on pronunciation only, not the broader spoken-language construct including content or grammar.
- The local PDF does not expose a DOI in recoverable text.

## Relevance To Peacock

- Highly relevant if Peacock wants pronunciation assessment beyond scripted prompts.
- The paper is a practical blueprint for using ASR transcripts as a bridge when reference text is unavailable.
- It also highlights two likely Peacock pain points: ASR error robustness and the gap between simulated free-speaking benchmarks and real open-response data.
