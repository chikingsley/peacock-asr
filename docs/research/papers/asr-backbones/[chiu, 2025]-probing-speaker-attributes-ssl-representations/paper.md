---
arxiv: 2501.05310
title: "A Large-Scale Probing Analysis of Speaker-Specific Attributes in Self-Supervised Speech Representations"
authors: "Aemon Yat Fei Chiu, Kei Ching Fung, Roger Tsz Yeung Li, Jingyu Li, Tan Lee"
year: 2025
venue: "Interspeech 2025"
category: asr-backbones
tags: [ssl, probing, layer-analysis, speaker-attributes, wav2vec2, hubert, wavlm]
---

Chiu et al. probe 11 speech SSL models (Wav2vec 2.0, HuBERT, UniSpeech-SAT, WavLM at multiple scales) across six speaker-specific attributes — speaker identity, gender (timbre proxy), pitch, tempo, energy, and emotion — using a simple MLP classifier on frame-averaged layer representations. They confirm an initial-to-middle hierarchy where acoustic/timbre attributes (gender) peak in early-to-middle layers while prosodic attributes (pitch, tempo, energy) are best captured in intermediate layers, but crucially challenge the consensus that final layers carry only linguistic content: larger models unexpectedly recover speaker identity in their deep layers. For pronunciation assessment, this layered picture is directly actionable — phonetic content is concentrated in upper layers while speaker-discriminative signal persists across early and deep layers in large models, so weighted or selective layer aggregation (rather than using only the final layer) can trade off speaker-normalisation against phonetic sensitivity when building GOP or scoring features.
