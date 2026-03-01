# Track 2: Real-Time Pronunciation Scoring via RL-Aligned Streaming Models

A second development track exploring streaming speech models + RL alignment techniques for real-time, open-ended pronunciation assessment. This runs parallel to the classical GOP-SF pipeline (Tracks covered in 01-05).

**Last audited:** 2026-03-01

**Core vision:** A language learner speaks freely (no script), and gets per-word pronunciation quality feedback in real-time, as they're speaking.

**Why this is different from the classical track:** The GOP-SF pipeline requires knowing the target transcript upfront. This track solves the open-ended case where we don't know what the speaker will say.

---

## 1. The Research Gap

Nobody is doing this yet. As of 2026-03-01:
- All pronunciation scoring systems are batch/offline
- All require a reference transcript (what the speaker was supposed to say)
- MultiPA (Interspeech 2024) is closest -- multi-task open-response scoring, but not streaming
- LLM-based MDD is emerging, but mostly detection/diagnosis instead of continuous per-word scoring

**The opportunity:** First streaming, open-ended pronunciation scoring system.

---

## 2. Foundation Model: Voxtral Realtime

**Why this model specifically:**

Voxtral Realtime (Mistral, Feb 2026) is a 4.4B natively streaming ASR model with architecture that maps perfectly to this problem.

| Property | Value |
|----------|-------|
| Parameters | 4.4B (970M encoder + 25M adapter + 3.4B decoder) |
| Streaming | True causal, 80ms granularity |
| Latency | Configurable 80ms - 2400ms (sweet spot: 480ms) |
| License | Apache 2.0 |
| HuggingFace | mistralai/Voxtral-Mini-4B-Realtime-2602 |
| LoRA | Supported via PEFT |
| vLLM | Full production support with WebSocket bidirectional streaming |

**Architecture details that matter:**
- **Causal audio encoder** (970M params): Trained from scratch. Cannot look ahead. Log-Mel → causal conv → 32 Transformer layers with sliding window (15s). Output at 50Hz.
- **Adapter**: Downsamples 4x → 12.5Hz (80ms per decoder step)
- **Decoder** (Ministral 3B): Emits `[P]` (padding/wait) and `[W]` (word-boundary) tokens
- **Ada RMS-Norm**: Conditions decoder on target delay via sinusoidal embeddings -- single model works at any latency

**Why [W] tokens are key:** The model already segments words in real-time. At each `[W]` token, the decoder hidden state encodes everything the model knows about that word acoustically. We tap those hidden states for pronunciation scoring.

Paper: arXiv:2602.11298 (downloaded to docs/papers/rl_alignment_speech/)

---

## 3. Proposed Architecture

### Primary: Auxiliary Scoring Head on Voxtral Realtime

```
Audio Stream → Causal Encoder → Adapter → LM Decoder → Transcription tokens
                                              ↓
                              Decoder hidden states at [W] boundaries
                                              ↓
                                 [Pronunciation Score MLP]
                                     per-word quality (0-5)
```

**Implementation:**
1. Freeze Voxtral Realtime weights
2. Add a lightweight MLP head (2-3 layers) on decoder hidden states at `[W]` token positions
3. LoRA-adapt the decoder layers + train the MLP head
4. Training data: speechocean762 word-level scores + L2-ARCTIC
5. Multi-task loss: original ASR loss + pronunciation regression loss
6. Deploy via vLLM WebSocket API

**Added latency:** Near zero (MLP inference is negligible)
**Added parameters:** ~1-5M scoring head + ~10-50M LoRA adapters
**Training compute:** Single A100 should suffice with LoRA

### Alternative: Token-level Scoring (simpler, less precise)

Fine-tune decoder to emit score tokens inline:
```
[W] word1 [SCORE:4] [W] word2 [SCORE:3] [W] word3 [SCORE:5]
```

Pro: No architectural changes. Con: Scores compete with text tokens for capacity.

---

## 4. RL Alignment Techniques for Speech (The Toolbox)

The LLM alignment stack has been ported to speech in 2025-2026. Here's what's available:

### 4.1 DPO (Direct Preference Optimization)

| Paper | ID | Date | Key Contribution |
|-------|----|------|------------------|
| DPO for Speech Enhancement | 2507.09929 | Jul 2025 (ICASSP 2026) | Uses UTMOS as reward proxy. 56% improvement. Replace UTMOS with GOP scorer → DPO for pronunciation |
| ARDM-DPO | 2509.18928 | Sep 2025 | DPO for speech diffusion models. Shows DPO works on continuous speech tokens |
| INTP + DPO for Intelligibility | 2505.04113 | May 2025 (ACL 2025) | 250K preference pairs for pronunciation-like intelligibility scoring. Dataset construction methodology directly applicable |
| DPO for TTS without labels | ICNLSP 2025 | 2025 | Automated preference pair construction using WER + speaker similarity ranking. Code: github.com/andrii-zhuravlov/xtts-dpo |
| Emo-DPO | 2409.10157 | ICASSP 2025 | DPO controls fine-grained speech attributes -- applicable to stress, intonation |

### 4.2 GRPO / PPO

| Paper | ID | Date | Key Contribution |
|-------|----|------|------------------|
| GRPO for ASR | 2509.01939 | Sep 2025 (ASRU 2025) | Rule-based rewards → 18.4% WER improvement. GOP scores as rewards = GRPO for pronunciation |
| GRPO for TTS | 2509.18798 | Sep 2025 | Uses ASR model as reward source (no dedicated reward model needed). CER as pronunciation metric |
| PPO for Pronunciation | DOI:10.1016/j.rineng.2025.103943 | 2025 | 97.9% phoneme accuracy, 87.7% word accuracy on CMU Sphinx. Exactly our use case |
| DiffRO | 2507.05911 | Jul 2025 (Interspeech 2025) | Differentiable rewards on codec tokens. Multi-Task Reward model includes pronunciation accuracy component |

### 4.3 LoRA for Pronunciation Assessment

| Paper | ID | Date | Key Contribution |
|-------|----|------|------------------|
| **LoRA on Phi-4-multimodal** | **2509.02915** | **Sep 2025** | **PCC > 0.7 on speechocean762. Single model does scoring + error detection. THE key paper** |
| Fine-tuning LMMs | 2509.15701 | Sep 2025 | PCC 0.9 at sentence level. Phoneme level still hard (SCC ~0.6). Need rank-aware training |
| Unified MLLM (ASRU 2025) | 2508.12591 | Aug 2025 | Phi-4 + curriculum learning → PCC 0.846 holistic. Speech-First Multimodal Training |
| GPT-4o baseline | 2503.11229 | Mar 2025 | Establishes commercial LLM baseline for pronunciation scoring |

### 4.4 Speech Reward Models

| Paper | ID | Date | Key Contribution |
|-------|----|------|------------------|
| **GSRM** | **2602.13891** | **Feb 2026** | **Chain-of-Thought reasoning reward model. Decomposes speech eval into acoustic features + reasoning. Near human-level** |
| SpeechJudge-GRM | 2511.07931 | Nov 2025 (ICLR 2026) | 99K preference pairs. SFT+GRPO trained on Qwen2.5-Omni. 77% human agreement |
| MOS-RMBench | 2510.00743 | Oct 2025 (ICLR 2026) | MOS → preference conversion methodology. Applicable to speechocean762 scores |
| Vox-Evaluator | 2510.20210 | Oct 2025 | Temporal error detection (WHERE errors occur). Detect-Mask-Regenerate workflow |

### 4.5 Synthetic Data for Pronunciation

| Paper | ID | Date | Key Contribution |
|-------|----|------|------------------|
| Zero-shot LLMs for L2 (SLaTE 2025) | 2601.16230 | Jan 2026 | Qwen2-Audio zero-shot on speechocean762. Overpredicts low-quality → RL alignment could fix |
| Reinforced Behavior Alignment | 2509.03526 | Aug 2025 | Self-synthesis + RL. Generate synthetic training data from model's own outputs |
| GoP2Vec (Interspeech 2025) | ISCA archive | Aug 2025 | GOP → i-vector representation. Few-shot pronunciation assessment matching supervised SOTA |

---

## 5. Atropos (Nous Research RL Framework)

GitHub: github.com/NousResearch/atropos

Viable for speech extensions via custom environments:
- **Custom Environment:** Audio samples in, pronunciation scores + feedback out, correlation with human scores as reward
- **LoRA via Tinker:** Tinker-atropos integration enables LoRA fine-tuning with RL objectives
- **DPO Data Generation:** `atropos-dpo-gen` for creating pronunciation preference pairs
- **Async Training:** Collect pronunciation rollouts in parallel while training proceeds

No existing speech environments yet. This would be novel.

---

## 6. Model Comparison (Streaming Candidates)

| Model | Params | Streaming | Latency | License | LoRA | Pronunciation Fit |
|-------|--------|-----------|---------|---------|------|-------------------|
| **Voxtral Realtime** | **4.4B** | **True causal** | **80ms-2.4s** | **Apache 2.0** | **Yes** | **★★★★★** |
| Canary-Qwen 2.5B | 2.5B | Cache-aware | Low | CC-BY-4.0 | Yes | ★★★★☆ |
| Qwen2.5-Omni-7B (2503.20215) | 7B | Block-wise (2s) | ~2s | Apache 2.0 | Yes (mixed) | ★★★☆☆ |
| Granite Speech 8B (2505.08699) | ~9B | Block (4s) | ~4s | Apache 2.0 | Yes | ★★★☆☆ |
| Parakeet TDT 1.1B (2509.14128) | 1.1B | True (RNN-T) | Ultra-low | CC-BY-4.0 | Yes | ★★★☆☆ |
| Moonshine v2 | 27M-300M | True causal | Ultra-low | MIT | Yes | ★★☆☆☆ |

Model-source note:
- Canary-Qwen row currently anchored to model card/source docs (not yet pinned to a single canonical paper in this track file).
- Current reference: huggingface.co/nvidia/canary-qwen-2.5b

---

## 7. Proposed Implementation Pattern

Combining the most promising techniques from the research:

### Phase 4A: LoRA + Scoring Head (Foundation)
1. Download Voxtral Realtime weights
2. Add pronunciation scoring MLP head at `[W]` token positions
3. LoRA fine-tune + train head on speechocean762 (word-level scores)
4. Evaluate: does it simultaneously transcribe + score?
5. Compare PCC with classical GOP-SF pipeline

### Phase 4B: DPO Alignment (Quality Boost)
1. Use Phase 4A model to generate pronunciation assessments
2. Build preference pairs: (good_score, bad_score) for same utterance
3. Use GOP model as reward proxy for pair construction
4. Apply DPO to align scoring with human preferences
5. Inspired by: 2507.09929 (DPO for speech), 2505.04113 (INTP preference pairs)

### Phase 4C: GRPO with Reward Model (Full RL Loop)
1. Train a pronunciation reward model inspired by GSRM (2602.13891)
2. Use GRPO (no critic needed) with composite reward: PCC + calibration + consistency
3. Optional: integrate with Atropos framework for async RL
4. Inspired by: 2509.01939 (GRPO for ASR), SpeechJudge-GRM

### Phase 4D: Real-Time Deployment
1. vLLM WebSocket API for bidirectional streaming
2. Audio in, transcription + per-word scores out, in real-time
3. Integration with language learning game frontend
4. Latency target: <500ms from word completion to score delivery

---

## 8. Key Papers Downloaded

All in `docs/papers/rl_alignment_speech/`:
- 2602.11298 — Voxtral Realtime architecture
- 2507.09929 — DPO for speech enhancement (ICASSP 2026)
- 2602.13891 — GSRM generative speech reward model
- 2507.05911 — DiffRO differentiable rewards
- 2509.02915 — LoRA pronunciation assessment (the key paper)
- 2308.12490 — MultiPA open-response pronunciation assessment
- 2511.07931 — SpeechJudge (ICLR 2026)
- 2510.00743 — MOS-RMBench / score-to-preference reward benchmarking
- 2025 Interspeech GoP2Vec paper (`sirigiaju25_interspeech.pdf`)
- 2512.04964 — HiPPO hierarchical pronunciation assessment
- 2509.16876 — Multi-task pretraining for interpretable L2 pronunciation assessment
- 2509.14187 — Read to Hear zero-shot pronunciation assessment

All in `docs/papers/streaming_realtime/`:
- 2602.12241 — Moonshine v2 streaming ASR
- 2601.22779 — Streaming ASR with decoder-only LLMs
- 2509.03256 — End-to-end word-level pronunciation assessment
- 2503.20215 — Qwen2.5-Omni Technical Report
- 2505.08699 — Granite-speech
- 2509.14128 — Canary-1B-v2 and Parakeet-TDT-v3
- 2509.08753 — Delayed Streams Modeling (Kyutai)
- 2511.23404 — LFM2 Technical Report (Liquid AI)
- 2602.11072 — Hibiki-Zero real-time speech translation
- 2502.03382 — Hibiki streaming speech translation
- 2410.00037 — Moshi speech-text full duplex architecture

### Download Status (2026-03-01)
- Previously listed "still to download" arXiv papers are now all present on disk.
- Several additional relevant CAPT/GOP papers are already present in other local folders:
  - `docs/papers/capt_systems/` (e.g., 2506.19315, 2510.04956)
  - `docs/papers/gop_methods/` (e.g., 2506.12067)
  - `docs/papers/core_segmentation_free/` (e.g., 2507.14346, 2507.16838)
- Items still lacking a clean primary paper ID in this track doc:
  - "DPO for TTS without labels (ICNLSP 2025)" (currently tracked via code repo reference)
  - Specific "Qwen2-Audio fine-tuning for MDD" citation (mentioned conceptually, not pinned to one paper)

---

## 9. Implementation Reality Check (Current Codebase)

What exists right now in `references/ADVANCED-transcription/`:
- Voxtral path has modal training/serving scripts, but serving is still transcription-oriented (no explicit pronunciation scoring head, no confidence/logprob API output, no true word-level score stream).
- Kyutai path has the strongest alignment primitives today for this track:
  - forced alignment tooling
  - timestamp alignment utilities
  - 80ms step / delayed-stream style setup
  - train/eval code where logits are accessible
- Neither path currently has an end-to-end CAPT/GOP-style loop:
  - `reference text -> phone alignment -> per-phone/word scoring -> learner feedback`

Practical near-term build order:
1. Add a reusable scoring module that consumes model logits + alignment and outputs calibrated per-word scores.
2. Wire that module first into Kyutai fine-tuning/eval pipeline for fast iteration and metric grounding.
3. Add online/chunked inference mode (80ms-160ms stride) with score smoothing and delayed commit.
4. Port the scoring head idea to Voxtral Realtime hidden states at `[W]` boundaries for open-ended streaming.
5. Only then add RL alignment (DPO/GRPO) once supervised scoring is stable and calibrated.

---

## 10. The Big Insight

No one has combined all these pieces yet:

**LoRA + DPO/GRPO + pronunciation reward model + streaming architecture + synthetic data = unified RL-trained real-time pronunciation assessment**

The papers exist in silos (TTS alignment, ASR GRPO, LoRA pronunciation, speech reward models, streaming ASR). Peacock-ASR could be the first to unify them.
