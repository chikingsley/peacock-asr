---
arxiv: 2312.05187
title: "Seamless: Multilingual Expressive and Streaming Speech Translation"
authors: "Seamless Communication (FAIR at Meta, INRIA, UC Berkeley)"
citation_author: "Barrault et al"
year: 2023
venue: "arXiv preprint"
source_pdf: "paper.pdf"
extraction_method: "Manual summary from arXiv PDF (abstract, intro, table of contents; §3.2 pre-training details not fully extracted)."
extracted_at: "2026-03-22"
llm_friendly: true
---

# Seamless: Multilingual Expressive and Streaming Speech Translation

## Metadata

- Authors: Large team from FAIR at Meta, INRIA, UC Berkeley (lead: Loïc Barrault, Yu-An Chung, Marta R. Costa-jussà, Xutai Ma)
- arXiv: 2312.05187
- Venue: arXiv preprint (Nov 2023)
- Task: Expressive and streaming speech-to-speech translation; introduces SeamlessM4T v2

## TL;DR

This paper introduces three models: **SeamlessM4T v2** (improved base with W2V-BERT 2.0 pre-trained on 4.5M hours), **SeamlessExpressive** (preserves vocal style and prosody across translation), and **SeamlessStreaming** (simultaneous translation without waiting for full utterances, using EMMA attention). The final **Seamless** model combines Expressive and Streaming. This paper contains the canonical description of W2V-BERT 2.0 as used in production.

## Abstract

SeamlessM4T v2 is an improved version of SeamlessM4T (2308.11596), using a new W2V-BERT 2.0 speech encoder pre-trained on **4.5 million hours** of unlabeled audio data (vs 1M hours in v1). SeamlessExpressive extends it to preserve vocal style and prosody (rhythm, speech rate, pauses) across translations, supporting 5 languages. SeamlessStreaming uses Efficient Monotonic Multihead Attention (EMMA) for low-latency simultaneous translation without waiting for complete source utterances. All three models are open-sourced.

## Key Contributions

### SeamlessM4T v2

Improvements over SeamlessM4T (2308.11596):

1. **W2V-BERT 2.0 upgrade**: speech encoder pre-trained on 4.5M hours (vs 1M hours in v1). This is the canonical version of W2V-BERT 2.0.
2. **UnitY2**: non-auto-regressive unit decoder + hierarchical upsampling (more data-efficient than UnitY in v1)
3. **SeamlessAlign expanded**: adds 114,800 hours of automatically aligned data (total 76 languages now)
4. Trained with more supervision from automatically aligned pairs to improve low-resource language performance
5. Language coverage: ~100 input speech/text languages

### SeamlessExpressive

- Preserves vocal style and prosody (rhythm, tone, speech rate, pauses) across speech translation
- Currently 5 languages for both from-English and to-English directions
- First model to enable expressive S2ST from **and** into English
- Uses expressive audio-aligned data for training

### SeamlessStreaming

- Leverages Efficient Monotonic Multihead Attention (EMMA) for low-latency translation
- Generates target translations without waiting for complete source utterances
- Supports many-to-many translations simultaneously (same language coverage as SeamlessM4T v2 in ASR, S2TT, S2ST)
- Latency measured by Ending Offset and Average Lagging metrics

### Seamless (Combined)

The final unified system combining SeamlessExpressive + SeamlessStreaming. First publicly available system enabling expressive cross-lingual communication in real-time.

## W2V-BERT 2.0 (§3.2)

The upgraded speech encoder for SeamlessM4T v2:

- **Pre-training data**: 4.5 million hours of unlabeled audio (vs 60k in original W2V-BERT, vs 1M in SeamlessM4T v1)
- **Architecture**: same contrastive + MLM dual-module design as original W2V-BERT (Chung et al. 2021, 2108.06209)
- **Framework**: UnitY2 multitask framework used for downstream training
- Multilingual by design (large-scale diverse audio)

## Responsible AI

Four-pronged approach:

1. Red-teaming: first known red-teaming effort for multimodal machine translation
2. Added-toxicity detection and mitigation
3. Systematic gender bias evaluation
4. **SeamlessWM**: inaudible localized watermarking mechanism to dampen deepfakes

New concept: **metric card** that compiles evaluation + Responsible AI metrics for a model.

## Structure (Table of Contents)

- §2: Sociotechnical need for expressive/streaming translation (user interviews)
- §3: SeamlessM4T v2 (data, pre-training W2V-BERT 2.0, UnitY2, S2ST training, results)
- §4: SeamlessExpressive (data, modeling, results, ablations)
- §5: SeamlessStreaming (EMMA, setup, results)
- §6: Seamless combined (architecture, results)
- §7: Automatic and Human Evaluation (AutoPCP expressivity metric, XSTS, MOS)
- §8: Responsible AI (red-teaming, toxicity, gender bias, watermarking)
- Appendices (101+)

## Relevance to Peacock

This paper is the canonical reference for **W2V-BERT 2.0** — the specific version that is deployed in production and available on HuggingFace. If extending the CHConv pool beyond the current three SSL models, W2V-BERT 2.0 is the most natural candidate:

- 4.5M hours of training data vs 60k (HuBERT) or 94k (WavLM)
- Multilingual, so captures phonetic diversity beyond English
- Already used in SeamlessM4T, available at `facebook/w2v-bert-2.0`

The fact that W2V-BERT 2.0 combines contrastive learning + MLM means different layers likely encode different information (lower layers: acoustic/phonetic via contrastive; upper layers: linguistic/contextual via MLM) — exactly the kind of layer-diversity that CHConv is designed to exploit.

**Relation to other papers in vault**: W2V-BERT (2108.06209) → W2V-BERT 2.0 here → OmniASR (2511.09690) continues the massively multilingual SSL trajectory.
