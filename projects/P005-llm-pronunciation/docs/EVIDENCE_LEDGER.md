# Track 06 Evidence Ledger: Multimodal LLM Pronunciation Assessment

Scope:

- LoRA fine-tuning of multimodal LLMs for pronunciation scoring
- Comparison of LLM-based vs CTC+GOP pipeline approaches
- Inference cost analysis

Citation policy:

- Use numbered citations in text: `[1]`, `[2]`, ...
- Use `./refs.bib` as canonical bib source.

---

## 1. Claim Map

| ID | Claim | Evidence Status | Primary Citations |
|---|---|---|---|
| C1 | LoRA fine-tuning of Phi-4-multimodal on SpeechOcean762 achieves sentence-level Accuracy PCC ~0.656-0.675 | Supported by paper (not yet reproduced internally) | [1] |
| C2 | Unfreezing audio encoder layers during fine-tuning can push Accuracy PCC to 0.743 at sentence level | Supported by paper (not reproduced internally) | [1] |
| C3 | Qwen2-Audio-7B fine-tuning achieves sentence-level Accuracy PCC ~0.77 on SpeechOcean762 | Supported by paper (not reproduced internally) | [5] |
| C4 | Zero-shot GPT-4o performs poorly for pronunciation scoring (PCC near 0 without fine-tuning) | Supported by [1] Table 3 (no-training baseline: Accuracy PCC ~0) | [1], [3] |
| C5 | Zero-shot Qwen2-Audio-7B shows strong agreement with human ratings within tolerance, but poor PCC | Partially supported — [6] reports qualitative agreement; PCC not primary metric | [6] |
| C6 | HiPPO (Mamba-based, non-LLM) achieves ~0.83 PCC at utterance-level on SpeechOcean762 | Supported by paper | [9] |
| C7 | Our GOPT+GOP-SF achieves phone-level PCC 0.677 on SpeechOcean762 | Supported (internal: Track 05 Phase 1 A3, 5 seeds) | Internal |
| C8 | LLM-based systems produce sentence-level scores; phone-level LLM scoring is rarely evaluated | Supported by gap analysis across [1]-[6] | [1]-[6] |
| C9 | Phi-4 fine-tuning requires ~80GB VRAM; 12GB GPU requires quantization | Supported by [1] Section 4.2 (A100 80GB used), [1] Section 5.3 (5.8B params) | [1] |
| C10 | LLM pronunciation fine-tuning may overfit on SpeechOcean762 (~5h of data, 2500 utterances) | Hypothesis — needs experiment; raised in [1] Section 5.3 as limitation | [1] |

Notes:

- C1-C3 are paper-reported, not internally reproduced. Mark all as "external claim, unverified" until Phase 1-2 complete.
- C8 is the key structural argument: LLM papers and our pipeline report PCC at different granularities.
  This makes naive comparison misleading and motivates Phase 3.
- C9 is a hard constraint for our hardware. Resolve before starting Phase 1.

---

## 2. External Reference Papers

| # | Paper | Key Result | Status |
|---|---|---|---|
| [1] | Ryu et al. (2025) "English Pronunciation Evaluation without Complex Joint Training" | Phi-4 LoRA: Accuracy PCC 0.656-0.675 (LoRA), 0.743 (Unfreeze, epoch 4), sentence-level | PDF local: `papers/[Ahn et al, 2025]-lora-mllm-apa-mdd-joint.pdf` |
| [2] | Fu et al. (2024) "Pronunciation Assessment with Multi-modal Large Language Models" | Data2vec2+Qwen MLLM: Fluency PCC 0.777, Accuracy PCC 0.713, sentence-level | PDF local: `papers/[Fu et al, 2024]-pronunciation-assessment-multimodal-llm.pdf` |
| [3] | Wang et al. (2025) "Exploring the Potential of LMMs for Pronunciation Assessment" | GPT-4o zero-shot: near-zero PCC; integration with traditional methods recommended | PDF local: `papers/[Wang et al, 2025]-lmm-pronunciation-assessment-gpt4o.pdf` |
| [4] | Fang et al. (2025) "Beyond Modality Limitations: Unified MLLM for Speaking Assessment" | SFMT curriculum: PCC 0.783 -> 0.846 holistic with MLLM | PDF local: `papers/[Fang et al, 2025]-mllm-automated-speaking-assessment-sfmt.pdf` |
| [5] | Yang et al. (2025) "Fine-Tuning LMMs for Automatic Pronunciation Assessment" | Qwen2-Audio-7B fine-tuned: sentence-level Accuracy PCC ~0.77 | PDF local: `papers/[Wang et al, 2025]-fine-tuning-lmm-automatic-pronunciation-assessment.pdf` |
| [6] | Parikh et al. (2026) "Zero-Shot Speech LLMs for Multi-Aspect Evaluation of L2 Speech" | Qwen2-Audio-7B zero-shot: strong ±2 tolerance agreement, poor PCC | PDF local: `papers/[Parikh et al, 2026]-zero-shot-speech-llms-l2-multi-aspect-evaluation.pdf` |
| [7] | Chen et al. (2024) "MultiPA: Multi-task Pronunciation Assessment for Open Response" | MultiPA multi-task: Fluency 0.772, Accuracy 0.705 | PDF local: `papers/[Chen et al, 2023]-multipa-multitask-open-response-pronunciation.pdf` |
| [8] | Chen et al. (2025) "Read to Hear: Zero-Shot Pronunciation Assessment via TextPA" | TextPA zero-shot via text descriptors: competitive PCC at lower cost than audio LLMs | PDF local: `papers/[Chen et al, 2025]-textpa-zero-shot-pronunciation-llm.pdf` |
| [9] | Yan et al. (2025) "HiPPO: Hierarchical Pronunciation Assessment" | HiPPO (Mamba): ~0.83 utterance-level PCC; non-LLM SOTA | PDF local: `papers/[Yan et al, 2025]-hippo-hierarchical-apa-unscripted-speech.pdf` |
| [10] | Kim et al. (2022) "Automatic Pronunciation Assessment using SSL" | SSL (wav2vec2 + HuBERT): fine-tuned CTC + BLSTM for pronunciation scoring | PDF local: `papers/[Kim et al, 2022]-ssl-pronunciation-assessment-wav2vec-hubert.pdf` |
| [11] | Gong et al. (2022) "GOPT: Transformer-Based Pronunciation Assessment" | GOPT: phone-level PCC 0.612 (original Kaldi features); our adaptation: 0.677 | See Track 05 workspace |

---

## 3. Key Technical Details

### Phi-4-Multimodal Architecture (from [1])

- Total params: ~5.8B
- Components: Audio Encoder + Audio Projector + LoRA adapter + LLM
- Audio encoder and projector: pretrained, claimed to not need additional training
- LoRA fine-tuning: trains only audio adapter layers; LLM weights frozen
- Unfreeze variant: also trains Audio Encoder and Audio Projector weights
- Training hardware used in paper: 1x NVIDIA A100 SXM 80GB
- Training config: batch size 8, gradient accumulation 8, Adam lr=2e-5

### Phi-4 LoRA Results (from [1], Table 3)

| Strategy | Epoch | Accuracy PCC | Fluency PCC | Prosodic PCC | Total PCC |
|---|---|---|---|---|---|
| No training | - | ~0 | -0.041 | -0.017 | -0.104 |
| LoRA only | 1 | 0.547 | 0.585 | 0.567 | 0.544 |
| LoRA only | 2 | 0.637 | 0.726 | 0.709 | 0.662 |
| LoRA only | 3 | 0.656 | 0.727 | 0.711 | 0.675 |
| LoRA only | 4 | 0.645 | 0.733 | 0.714 | 0.668 |
| Unfreeze | 4 | 0.743 | 0.717 | 0.704 | 0.666 |

All of the above are **sentence-level** PCC. Our GOPT baseline 0.677 is **phone-level**.

### Data2vec2+Qwen MLLM Results (from [2], Table 2)

| Method | Type | Fluency PCC | Accuracy PCC |
|---|---|---|---|
| GOPT | Align-based | 0.753 | 0.714 |
| MultiPA | Align-based | 0.772 | 0.705 |
| Fu et al. (proposed) | Align-free MLLM | 0.777 | 0.713 |

These are also **sentence-level** PCC.

### Granularity Gap (Critical for Comparison)

Our GOP pipeline natively produces phone-level scores. LLM papers score at sentence level.
To compare fairly, we must either:
  (a) Aggregate our phone-level scores to sentence level (mean pool)
  (b) Prompt the LLM to produce phone-level scores (novel, difficult)
  (c) Report both and be explicit about the granularity difference

The granularity difference is the single most important confound in this track.

### VRAM Requirements (for hardware planning)

| Model | Params | FP16 VRAM | 4-bit VRAM |
|---|---|---|---|
| Phi-4-multimodal | 5.8B | ~12GB (inf only) | ~4-6GB |
| Qwen2-Audio-7B | 7B | ~14GB | ~5-7GB |
| Our CTC backbone (xlsr-espeak) | ~300M | ~2-3GB | N/A |
| Our GOPT scorer | ~1M | <100MB | N/A |

RTX 5070 has 12GB VRAM. Phi-4 FP16 inference is borderline; training is not feasible without
quantization. Qwen2-Audio-7B in 4-bit is the more accessible option for our hardware.

---

## 4. Internal Evidence Anchors

- GOPT baseline runs: Track 05 Phase 1 (`runs/2026-03-03_001037_track05_phase1_baseline/`)
- GOPT model: `/home/simon/github/peacock-asr/projects/P001-gop-baselines/code/p001_gop/gopt_model.py`
- GOP feature extraction: `/home/simon/github/peacock-asr/projects/P001-gop-baselines/code/p001_gop/gop.py`
- Track 06 narrative: `/home/simon/github/peacock-asr/docs/research/archived/06_LLM_PRONUNCIATION.md`

---

## 5. Open Questions (from archived 06_LLM_PRONUNCIATION.md)

- Does LoRA fine-tuning on SpeechOcean762 overfit? The dataset has only ~5h of audio.
- Can an LLM give per-phone scores or only word/utterance level?
- What is the inference latency? Is real-time feedback feasible?
- Is the LLM learning pronunciation quality or just ASR confidence?
- How does phone-level PCC from our GOP pipeline compare to sentence-level PCC from LLMs
  when both are evaluated at the same granularity?

---

## 6. Recommendation for Paper Draft

Use this ledger as the single source of truth for claims:

- `Introduction/Motivation`: cite [1]-[4] to establish LLM approach; cite [11] for GOP baseline.
- `Methods`: cite [1] for Phi-4 architecture; [5] for Qwen2-Audio; [10] for SSL bridge.
- `Results`: all LLM numbers from [1]-[6] until we have internal reproductions.
  Mark external numbers explicitly until reproduced.
- `Discussion/Cost Analysis`: cite [1] Section 5.3 for hardware constraints.
- `Related Work`: [7], [8], [9] for non-LLM context and zero-shot alternatives.
