---
arxiv: 2308.11596
title: "SeamlessM4T: Massively Multilingual & Multimodal Machine Translation"
authors: "Seamless Communication (Meta AI, INRIA, UC Berkeley)"
citation_author: "Seamless Communication et al"
year: 2023
venue: "arXiv preprint"
source_pdf: "paper.pdf"
extraction_method: "Manual summary from arXiv PDF (abstract + intro + table of contents; full technical content in §4 not read in detail)."
extracted_at: "2026-03-22"
llm_friendly: true
---

## Metadata

- Authors: Large team from Meta AI, INRIA, UC Berkeley (lead authors include Loïc Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Marta R. Costa-jussà, Juan Pino)
- arXiv: 2308.11596
- Venue: arXiv preprint (Oct 2023)
- Task: Unified speech-to-speech / speech-to-text / text-to-speech / text-to-text translation; ASR

## TL;DR

SeamlessM4T is a single unified model capable of five modality tasks (ASR, S2TT, S2ST, T2TT, T2ST) across up to 100 languages. Its speech encoder is W2V-BERT 2.0, pre-trained on 1 million hours of open audio. It outperforms strong cascaded systems on FLEURS by +20% BLEU for S2TT and +2.6 ASR-BLEU for S2ST. This paper is the canonical reference for W2V-BERT 2.0 v1 (1M-hour version).

## Abstract

SeamlessM4T (Massively Multilingual & Multimodal Machine Translation) is a single model that supports ASR, speech-to-text translation (S2TT), text-to-speech translation (T2ST), text-to-text translation (T2TT), and speech-to-speech translation (S2ST) across up to 100 languages. To build this, 1 million hours of open speech audio were used to learn self-supervised speech representations with **W2V-BERT 2.0**. A multimodal aligned corpus called **SeamlessAlign** (470,000 hours) was created automatically. Combined with human-labeled and pseudo-labeled data (406,000 hours total), the system achieves +20% BLEU improvement on FLEURS over the previous SOTA for S2TT into English, outperforms cascaded models by 1.3 BLEU in S2TT, and exceeds 2-stage cascaded models by 8.5 ASR-BLEU on S2ST.

## Task Coverage (Table 1)

| Task | Description |
|------|-------------|
| ASR | Automatic Speech Recognition |
| S2TT | Speech-to-Text Translation |
| S2ST | Speech-to-Speech Translation |
| T2TT | Text-to-Text Translation |
| T2ST | Text-to-Speech Translation |
| X2T | {Speech, Text}-to-Text Translation (multitasking) |

## Model Sizes and Language Coverage (Table 2)

| Model | Size | S2TT | S2ST | ASR | T2TT | T2ST |
|-------|------|------|------|-----|------|------|
| SeamlessM4T-Large | 2.3B | 100-eng / eng-95 | 100-eng / eng-35 | 96 | 95-eng / eng-95 | 95-eng / eng-95 |
| SeamlessM4T-Medium | 1.2B | same | same | 96 | same | same |
| SeamlessM4T-NLLB-1.3B | 1.3B | — | — | — | 95-eng / eng-95 | — |

For context, Whisper-Large-v2 supports 96 ASR languages (English-centric). NLLB-3.3B covers 202-202 T2TT languages.

## W2V-BERT 2.0 (Speech Encoder)

The speech encoder is **W2V-BERT 2.0**, an improved version of the original w2v-BERT (Chung et al. 2021, 2108.06209). Key facts from this paper:

- Pre-trained on **1 million hours** of open speech audio
- Multilingual (contrasted with the original w2v-BERT trained on English-only Libri-Light 60k)
- Architecture follows the same contrastive + MLM dual-module design as the original, but scaled
- Detailed architecture is described in §4.1 of this paper (not fully extracted; see Seamless paper 2312.05187 for a more complete description of v2)

## SeamlessAlign

A new multimodal aligned corpus:

- 470,000 hours of automatically aligned speech translations (raw)
- 406,000 hours filtered + combined with human-labeled and pseudo-labeled data for training
- Created using a pipeline: speech-language identification → raw data gathering → speech mining
- Open-sourced (metadata to recreate the unfiltered 470,000 hours)

## Key Results

On FLEURS benchmark:

- SeamlessM4T-Large beats previous SOTA (AudioPaLM-2-8B-AST, 8B params) by 4.2 BLEU on S2TT into English (20% relative)
- Outperforms cascaded (ASR → T2TT → TTS) 3-stage systems by 2.6 ASR-BLEU on S2ST
- On CVSS, outperforms Whisper-Large-v2 + YourTTS 2-stage cascade by 8.5 ASR-BLEU (50% improvement)
- ASR: 45% WER reduction over Whisper-Large-v2 on FLEURS (77 overlapping languages)

Human evaluation (S2TT from English): XSTS scores > 4/5 consistently for 24 evaluated languages.

## Responsible AI

- Gender bias: model overgeneralizes to masculine forms (~10% preference for masculine pronouns from neutral terms)
- Toxicity: 63% reduction in added toxicity compared to SOTA
- Evaluations open-sourced via FAIRSEQ2 toolkit

## Structure of the Paper

The full paper is 110+ pages (including appendices). Key sections:

- §2: Sociotechnical dimensions (why speech matters more than text for translation)
- §3: SeamlessAlign pipeline (speech-language ID, data gathering, mining)
- §4: SeamlessM4T models (§4.1: W2V-BERT 2.0 pre-training; §4.2: X2T training; §4.3: S2ST; §4.4: full model)
- §5: Evaluation (BLASER 2.0 metric, human evaluation)
- §6: Responsible AI (toxicity, bias)

## Relevance to Peacock

This paper matters for two reasons:

1. **W2V-BERT 2.0 as a potential CHConv feature source**: SeamlessM4T's W2V-BERT 2.0 is multilingual and trained on 1M hours. If the research goal expands beyond English-only pronunciation assessment, it is a natural candidate for adding to the SSL pool alongside wav2vec2/HuBERT/WavLM. All layers of W2V-BERT 2.0 can be extracted with `output_hidden_states=True`.

2. **Context for OmniASR (2511.09690)**: OmniASR covers 1,600+ languages and scales SSL to 7B params. SeamlessM4T is the direct predecessor in the "massively multilingual speech" lineage.

The improved version of W2V-BERT 2.0 (pre-trained on 4.5M hours) is described in the Seamless paper (2312.05187).
