# Track 09 Paper Collection

All papers relevant to the ConPCO / hierarchical pronunciation assessment line of research.

## Yan & Chen Lab (NTNU) — Core Chain

These papers build on each other. Read in chronological order to understand the evolution.

| # | File | Paper | Venue | Year | Key Contribution |
|---|------|-------|-------|------|-----------------|
| 1 | `[Yan et al, 2023]-pco-preserving-phonemic-distinctions-ordinal-regression.pdf` | PCO: Preserving Phonemic Distinctions for Ordinal Regression | ASRU 2023 | 2023 | Ordinal entropy loss for pronunciation scoring |
| 2 | `[Yan et al, 2024]-hierarchical-transformer-pre-training-strategies-apa.pdf` | HierTFR: Hierarchical Transformer with Pre-training Strategies | ACL 2024 | 2024 | HierCB architecture (BlockCNN + phone→word→utterance hierarchy) |
| 3 | `[Yan et al, 2024]-conpco-contrastive-phonemic-ordinal-regularization.pdf` | ConPCO: Contrastive Phonemic Ordinal Regularization | ICASSP 2025 | 2025 | Adds CLAP contrastive alignment to PCO. **SOTA PCC 0.701** (phone-level, Table II) on SpeechOcean762 |
| 4 | `[Li et al, 2025]-multi-task-pretraining-interpretable-apa.pdf` | Multi-task Pretraining for Interpretable L2 Pronunciation Assessment | Interspeech 2025 | 2025 | Masked pretraining of phonetic/prosodic features on HierCB encoder |
| 5 | `[Yan et al, 2025]-muffin-multifaceted-apa-mdd-joint.pdf` | MuFFIN: Multifaceted Pronunciation Feedback | IEEE/ACM TASLP | 2025 | Unifies MDD + APA, reuses ConPCO regularizer |
| 6 | `[Yan et al, 2025]-hippo-hierarchical-apa-unscripted-speech.pdf` | HiPPO: Hierarchical Pronunciation Assessment | IJCNLP-AACL 2025 | 2025 | Hierarchical assessment with interpretability |

### Dependency chain

```text
Ordinal Entropy (ICLR 2023, Zhang et al.) ─── math foundation
  └── PCO (ASRU 2023) ─── adapts ordinal entropy for pronunciation
       └── HierTFR/HierCB (ACL 2024) ─── hierarchical architecture
            ├── ConPCO (ICASSP 2025) ─── + CLAP contrastive term → SOTA
            ├── Multi-task Pretraining (Interspeech 2025) ─── + masked pretraining
            ├── MuFFIN (TASLP 2025) ─── + joint MDD/APA
            └── HiPPO (AACL 2025) ─── + interpretability
```

## Math Foundation

| File | Paper | Authors | Venue | Year | Relevance |
|------|-------|---------|-------|------|-----------|
| `[Zhang et al, 2023]-improving-deep-regression-with-ordinal-entropy.pdf` | Improving Deep Regression with Ordinal Entropy | Zhang, Yang, Mi, Zheng, Yao (NUS / Huawei) | ICLR 2023 | 2023 | **The math behind PCO/ConPCO.** Shows regression learns low-entropy features vs classification. Proposes ordinal entropy regularizer that Yan & Chen adapted for pronunciation scoring. |

## Other Groups

| File | Paper | Authors | Venue | Year | Relevance |
|------|-------|---------|-------|------|-----------|
| `[Chen et al, 2025]-textpa-zero-shot-pronunciation-llm.pdf` | Read to Hear: Zero-Shot Pronunciation Assessment via Text+LLMs | Chen, Ma, Hirschberg (Columbia) | EMNLP 2025 | 2025 | Alternative paradigm: zero-shot LLM scoring, no training on scored audio |

## Acquired (IEEE Official — purchased)

| File | Paper | Venue | Year | Notes |
|------|-------|-------|------|-------|
| `[Yan et al, 2024]-hierarchical-graph-attention-network-apa.pdf` | HierGAT: Hierarchical Graph Attention Network for APA | IEEE/ACM TASLP 2024 | 2024 | 12 pages, full journal paper. Graph-based architecture preceding HierCB. |
| `[Yan et al, 2025]-conpco-contrastive-phonemic-ordinal-regularization.pdf` | ConPCO (IEEE official version) | ICASSP 2025 | 2025 | Official IEEE version (also have arxiv preprint as `[Yan et al, 2024]-conpco-contrastive-phonemic-ordinal-regularization.pdf`) |

## Still Missing

| Paper | Venue | Why We Need It | How to Get |
|-------|-------|---------------|-----------|
| MASA: Multimodal Foundation Model for L2 Speaking Assessment | LREC 2026 (not yet published) | Newest Yan & Chen work — multimodal LLM for picture-description assessment. | Wait for May 2026 conference proceedings, or check arxiv closer to date. |

## Also In Repo

The ConPCO reference implementation is cloned at `projects/P002-conpco-scoring/third_party/ConPCO/` (includes code + poster PNG, no paper PDF).
