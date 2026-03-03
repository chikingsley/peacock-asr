# Track 06 Ablation Plan: Multimodal LLM Pronunciation Assessment

## Research Question

Can LoRA-fine-tuned multimodal LLMs match or exceed our CTC+GOP+GOPT pipeline (PCC 0.677)
on SpeechOcean762 phone-level accuracy scoring? And at what inference cost?

## Context: What LLM Papers Report

The Phi-4 LoRA paper (2509.02915) reports sentence-level scores:

| Model | Accuracy PCC | Fluency PCC | Prosodic PCC | Total PCC |
|---|---|---|---|---|
| Phi-4 LoRA (epoch 3) | 0.656 | 0.727 | 0.711 | 0.675 |
| Phi-4 Unfreeze (epoch 4) | 0.743 | 0.717 | 0.704 | 0.666 |

These are sentence-level scores, not phone-level. Our GOPT baseline (0.677) is phone-level.
Comparison requires care: different granularity means different metrics.

The Qwen2-Audio fine-tuning paper (2509.15701) reports Accuracy PCC ~0.77 at sentence level.
HiPPO (2512.04964) reports ~0.83 PCC at utterance level (non-LLM Mamba approach).

## Frozen Setup (Eval Contract)

- Dataset: SpeechOcean762 (standard 2,500 train / 2,500 test split)
- Primary eval: PCC between predicted and human scores
- Granularity note: must be explicit — phone-level vs sentence-level are not directly comparable
- Baseline: our GOPT+GOP-SF (Track 05 Phase 1 A3): phone-level PCC 0.677 +/- 0.013 (5 seeds)

## Phase 1: Reproduce Phi-4-Multimodal LoRA Result

Goal: reproduce the Ryu et al. (2509.02915) result using their training setup on SpeechOcean762.

| Run ID | Model | Fine-tuning | Output granularity | Target metric |
|---|---|---|---|---|
| P1-A | Phi-4-multimodal-instruct | LoRA only (audio adapter) | Sentence-level accuracy PCC | Reproduce 0.656 (epoch 3) |
| P1-B | Phi-4-multimodal-instruct | LoRA + unfreeze audio layers | Sentence-level accuracy PCC | Reproduce 0.743 (epoch 4) |

Implementation requirements:

- `microsoft/Phi-4-multimodal-instruct` from HuggingFace
- LoRA on audio adapter layers; LLM layers frozen
- SFT with APA prompts (see 2509.02915 Appendix 7.1)
- Training: batch size 8, gradient accumulation 8, Adam lr=2e-5
- Hardware: requires ~80GB VRAM (paper used A100 80GB); our RTX 5070 (12GB) is not sufficient
  for full Phi-4 (5.8B). Quantized inference (int4/int8) may be needed.
- Record: training loss curves, per-epoch PCC on test split

Expected effort: 3-5 days (including quantization/compute planning)

Hardware note: The 5070 has 12GB VRAM. Phi-4 multimodal at fp16 requires ~12GB for inference
alone, making training infeasible without quantization or offloading. Options:
  (a) Use Unsloth's 4-bit quantized Phi-4 for fine-tuning on 12GB
  (b) Use cloud GPU (A100/H100) for training, 5070 only for inference
  (c) Use Qwen2-Audio-7B-Instruct which may be more 12GB-friendly with quantization

## Phase 2: Qwen2-Audio-7B as Alternative Backbone

Goal: evaluate Qwen2-Audio-7B-Instruct as a potentially more hardware-accessible LLM backbone.

| Run ID | Model | Fine-tuning | Granularity | Target |
|---|---|---|---|---|
| P2-A | Qwen2-Audio-7B-Instruct | Zero-shot (no fine-tuning) | Sentence-level | Establish zero-shot baseline |
| P2-B | Qwen2-Audio-7B-Instruct | LoRA fine-tuning | Sentence-level accuracy PCC | Approach reported 0.77 |
| P2-C | Qwen2-Audio-7B-Instruct | LoRA fine-tuning | Phone-level accuracy PCC | Novel: can LLMs score phones? |

Implementation requirements:

- `Qwen/Qwen2-Audio-7B-Instruct` from HuggingFace
- Zero-shot: design rubric-aligned prompt (see 2601.16230 for prompt design approach)
- LoRA fine-tuning: ~7B params, 4-bit quantized should fit 12GB for inference
- P2-C is novel: no prior paper reports phone-level PCC from Qwen2-Audio on SpeechOcean762

Expected effort: 3-5 days (after Phase 1 infrastructure is in place)
Expected P2-B result: PCC 0.70-0.77 sentence-level (based on 2509.15701)

## Phase 3: Fair Comparison — LLM Scores vs GOP PCC on Same Eval Set

Goal: determine whether LLMs actually beat our pipeline on a comparable task.

Key challenge: LLM papers report sentence-level PCC; our pipeline produces phone-level PCC.
These are different tasks. This phase establishes an apples-to-apples comparison.

| Run ID | System | Granularity | PCC | Notes |
|---|---|---|---|---|
| P3-A | GOPT+GOP-SF (ours) | Phone-level accuracy | 0.677 | From Track 05 Phase 1 |
| P3-B | GOPT+GOP-SF (ours) | Sentence-level accuracy (mean pooled) | TBD | Aggregate phone scores to sentence |
| P3-C | Best LLM from Phase 1-2 | Sentence-level accuracy | TBD | Best LLM on same sentence-level task |
| P3-D | Best LLM from Phase 1-2 | Phone-level accuracy | TBD | Can the LLM score at phone level? |

Analysis:

- P3-B vs P3-C: sentence-level comparison (our GOP aggregated vs LLM native)
- P3-A vs P3-D: phone-level comparison (our native vs LLM attempting phone scoring)
- If LLM wins on sentence but loses on phone: LLMs are better holistic scorers
- If LLM loses on sentence but GOPT wins on phone: CTC+GOP pipeline more phone-precise

Expected effort: 1-2 days (once Phase 1-2 models are trained)

## Phase 4: Inference Cost Analysis

Goal: quantify the practical cost of LLM-based vs CTC+GOP assessment.

Metrics to measure:

- Wall-clock inference time per utterance (mean, p95)
- GPU memory required at inference time
- Parameters: LLM total params vs our pipeline (CTC backbone + GOPT)
- VRAM required for serving at batch size 1 (real-time setting) and batch size 32

| System | Params | VRAM (inf) | Time/utterance (est) |
|---|---|---|---|
| GOPT+GOP-SF (ours) | ~1M scorer + 300M CTC backbone | ~6GB (backbone) | ~50ms (CTC) + ~2ms (GOPT) |
| Phi-4 LoRA | ~5.8B | ~12-24GB | TBD |
| Qwen2-Audio-7B LoRA | ~7B | ~14-28GB | TBD |
| Qwen2-Audio-7B 4-bit | ~7B (quant) | ~8-10GB | TBD |

Expected effort: 1 day (systematic profiling after Phase 1-2 inference is working)

## Decision Rules

- `Phase 1 -> Phase 2`: Proceed regardless; Qwen2-Audio provides hardware diversity.
- `Phase 2 -> Phase 3`: If either LLM achieves >0.65 sentence-level PCC, Phase 3 comparison is meaningful.
- `Phase 3 -> Phase 4`: Always proceed. Cost analysis is always publishable.
- `Paper claim: LLMs win`: Requires LLM PCC >= 0.70 at the same granularity as our best GOP result.
- `Paper claim: GOP wins`: Our GOP PCC >= LLM PCC at matched granularity, with lower inference cost.
- `Paper claim: trade-off`: If LLM wins sentence-level but GOP wins phone-level (or vice versa).

## Deliverables per Run

- Config snapshot (model ID, LoRA rank/alpha, training epochs, lr, batch size)
- Exact training command + environment (Python version, CUDA, library versions)
- Checkpoint path and HuggingFace model ID if applicable
- Per-epoch PCC on test split
- Final PCC with granularity explicitly noted (phone vs sentence)
- Inference profiling: time/utterance, VRAM
- Random seeds used (minimum 3 for LoRA fine-tuning)

## Paper Structure Preview

- **Table 1**: Phase 1-2 LLM fine-tuning results (sentence-level PCC)
- **Table 2**: Phase 3 head-to-head comparison at matched granularity
- **Table 3**: Phase 4 inference cost comparison
- **Discussion**: When does CTC+GOP beat LLMs? When do LLMs win? What does this mean for CAPT?
- **Practical takeaway**: For real-time phone-level CAPT, which approach is more deployable?
