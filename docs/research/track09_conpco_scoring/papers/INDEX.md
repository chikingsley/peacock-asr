# Track 09 Paper Collection

All papers relevant to the ConPCO / hierarchical pronunciation assessment line of research.

## Yan & Chen Lab (NTNU) — Core Chain

These papers build on each other. Read in chronological order to understand the evolution.

| # | File | Paper | Venue | Year | Key Contribution |
|---|------|-------|-------|------|-----------------|
| 1 | `pco_predecessor_2310.01839.pdf` | PCO: Preserving Phonemic Distinctions for Ordinal Regression | ASRU 2023 | 2023 | Ordinal entropy loss for pronunciation scoring |
| 2 | `hiertfr_acl2024.pdf` | HierTFR: Hierarchical Transformer with Pre-training Strategies | ACL 2024 | 2024 | HierCB architecture (BlockCNN + phone→word→utterance hierarchy) |
| 3 | `conpco_2406.02859.pdf` | ConPCO: Contrastive Phonemic Ordinal Regularization | ICASSP 2025 | 2025 | Adds CLAP contrastive alignment to PCO. **SOTA PCC 0.743** on SpeechOcean762 |
| 4 | `multitask_pretraining_2509.16876.pdf` | Multi-task Pretraining for Interpretable L2 Pronunciation Assessment | Interspeech 2025 | 2025 | Masked pretraining of phonetic/prosodic features on HierCB encoder |
| 5 | `muffin_2510.04956.pdf` | MuFFIN: Multifaceted Pronunciation Feedback | IEEE/ACM TASLP | 2025 | Unifies MDD + APA, reuses ConPCO regularizer |
| 6 | `hippo_2512.04964.pdf` | HiPPO: Hierarchical Pronunciation Assessment | IJCNLP-AACL 2025 | 2025 | Hierarchical assessment with interpretability |

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
| `ordinal_entropy_2301.08915.pdf` | Improving Deep Regression with Ordinal Entropy | Zhang, Yang, Mi, Zheng, Yao (NUS / Huawei) | ICLR 2023 | 2023 | **The math behind PCO/ConPCO.** Shows regression learns low-entropy features vs classification. Proposes ordinal entropy regularizer that Yan & Chen adapted for pronunciation scoring. |

## Other Groups

| File | Paper | Authors | Venue | Year | Relevance |
|------|-------|---------|-------|------|-----------|
| `read_to_hear_emnlp.pdf` | Read to Hear: Zero-Shot Pronunciation Assessment via Text+LLMs | Chen, Ma, Hirschberg (Columbia) | EMNLP 2025 | 2025 | Alternative paradigm: zero-shot LLM scoring, no training on scored audio |

## Missing (Need to Acquire)

| Paper | Venue | Why We Need It | How to Get |
|-------|-------|---------------|-----------|
| HierGAT: Hierarchical Graph Attention Network for APA | IEEE/ACM TASLP 2024 (DOI: 10.1109/TASLP.2024.3449111) | Introduces the graph-based architecture that preceded HierCB. Needed to understand full lineage. | IEEE paywalled. Try: email <bicheng@ntnu.edu.tw>, or check ResearchGate for full-text request. |
| MASA: Multimodal Foundation Model for L2 Speaking Assessment | LREC 2026 (not yet published) | Newest Yan & Chen work — multimodal LLM for picture-description assessment. | Wait for May 2026 conference proceedings, or check arxiv closer to date. |

## Also In Repo

The ConPCO reference implementation is cloned at `references/ConPCO/` (includes code + poster PNG, no paper PDF).
