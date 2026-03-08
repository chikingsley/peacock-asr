# Multimodal LLM Pronunciation Assessment: LoRA Fine-Tuning vs CTC+GOP Pipelines on SpeechOcean762

## Abstract

*TODO: Write after Phase 3 experiments complete.*

We compare LoRA-fine-tuned multimodal large language models (Phi-4-multimodal, Qwen2-Audio-7B)
against our established CTC+GOP+GOPT pipeline for pronunciation scoring on SpeechOcean762.
Our CTC+GOP+GOPT baseline achieves PCC 0.677 at phone level. Prior work reports sentence-level
Accuracy PCC of 0.656-0.675 for Phi-4 LoRA and ~0.77 for Qwen2-Audio-7B. We conduct a
controlled comparison at matched granularity, and analyze inference cost. We find that
[results pending].

## 1. Introduction

Automatic pronunciation assessment (APA) is a key component of computer-assisted pronunciation
training (CAPT) systems, which help second-language (L2) learners develop fluency by providing
automated, scalable feedback [@zhang2021speechocean762]. Two broad families of approaches exist:
pipeline-based systems that extract acoustic features (GOP, CTC posteriors) and pass them to a
downstream scorer [@cao2026segmentation_free_gop; @gong2022gopt_transformer_pronunciation_assessment],
and end-to-end systems that directly consume raw audio and produce scores without explicit
phoneme modeling [@kim2022ssl_pronunciation_assessment].

The emergence of large multimodal models (LMMs) capable of processing audio and text jointly
has introduced a third paradigm: feeding learner speech and a canonical text prompt directly
to a pretrained LLM, which predicts pronunciation scores as free text
[@fu2024mllm_pronunciation; @ryu2025phi4_lora_pronunciation]. This approach bypasses CTC
training, GOP computation, and explicit phoneme modeling entirely. It is appealing in its
simplicity: LoRA fine-tuning on SpeechOcean762 alone appears to produce competitive results
[@ryu2025phi4_lora_pronunciation].

However, a careful reading of the literature reveals an important confound: LLM-based papers
report sentence-level PCC, while our CTC+GOP+GOPT pipeline is optimized for and evaluated at
phone level. These metrics are not directly comparable. The question of which approach is
superior depends critically on the evaluation granularity required by the downstream CAPT task.

This paper addresses that gap. We (1) reproduce the Phi-4 LoRA and Qwen2-Audio-7B results
on SpeechOcean762, (2) conduct a head-to-head comparison at matched granularity, and (3)
analyze the inference cost trade-off between a 5.8-7B parameter LLM and our lightweight
CTC+GOPT pipeline.

Contributions:

- A reproducible reproduction of Phi-4 LoRA and Qwen2-Audio-7B fine-tuning on SpeechOcean762.
- A granularity-matched comparison of LLM-based and CTC+GOP-based pronunciation scoring.
- A systematic inference cost analysis (VRAM, latency) across both system families.
- Evidence for [the better approach] in phone-level and sentence-level CAPT settings.

## 2. Related Work

### 2.1 CTC+GOP Pipeline Approaches

Goodness of Pronunciation (GOP) scores computed from CTC phone posteriors are a well-established
basis for APA systems. Segmentation-free GOP [@cao2026segmentation_free_gop] avoids forced
alignment by marginalizing over all CTC paths, and is the feature extraction method used in
our pipeline. Downstream modeling via the GOPT transformer [@gong2022gopt_transformer_pronunciation_assessment]
adds contextual phone-sequence modeling on top of GOP features. Our Track 05 ablation
establishes PCC 0.677 (phone-level) on SpeechOcean762 using this stack.

Self-supervised learning approaches [@kim2022ssl_pronunciation_assessment] bridge the CTC and
end-to-end families by fine-tuning wav2vec 2.0 or HuBERT with CTC, then extracting layer-wise
representations for scoring.

### 2.2 Multimodal LLM Approaches

Fu et al. [-@fu2024mllm_pronunciation] introduced the first MLLM-based pronunciation scoring
system, using Data2vec2 as the audio encoder and Qwen-7B as the LLM backbone. Their two-stage
training (ASR pretraining, then pronunciation fine-tuning) achieves Fluency PCC 0.777 and
Accuracy PCC 0.713 at sentence level.

Ryu et al. [-@ryu2025phi4_lora_pronunciation] demonstrate that LoRA-only fine-tuning of
Phi-4-multimodal-instruct on SpeechOcean762 achieves Accuracy PCC 0.656-0.675 at sentence
level, while unfreezing the audio encoder pushes Accuracy PCC to 0.743 at the cost of full
audio layer training.

Yang et al. [-@yang2025qwen2audio_finetuning] fine-tune Qwen2-Audio-7B-Instruct on
SpeechOcean762, achieving sentence-level Accuracy PCC ~0.77, and demonstrate that
phone-level assessment remains challenging for LMMs.

### 2.3 Zero-Shot and Text-Based Approaches

Without any fine-tuning, GPT-4o achieves near-zero PCC for pronunciation scoring
[@wang2025gpt4o_pronunciation], confirming that task-specific adaptation is necessary.
Parikh et al. [-@parikh2026zeroshot_speech_llm] evaluate zero-shot Qwen2-Audio-7B,
finding strong agreement within ±2 tolerance but poor PCC correlation.

Chen et al. [-@chen2025read_to_hear] propose TextPA, a zero-shot approach that converts
audio to text-based acoustic descriptors (IPA transcripts, pause durations) and feeds them
to a text-only LLM. This avoids the high inference cost of audio LLMs while remaining
competitive for sentence-level scoring.

### 2.4 Non-LLM SOTA

HiPPO [@yan2025hippo] achieves ~0.83 utterance-level PCC using a hierarchical Mamba-based
architecture. This represents the non-LLM SOTA and establishes an upper bound for comparison
without requiring multi-billion-parameter inference.

## 3. Methods

### 3.1 CTC+GOP+GOPT Baseline (Track 05 Inherited)

*Inherit from Track 05 manuscript, Section 3.*

Pipeline:

- Audio -> CTC phone posterior matrix P(t, v) via wav2vec2-xlsr-53 with espeak phonemizer
- Segmentation-free GOP feature extraction [@cao2026segmentation_free_gop]
- Phone-level transformer scoring (GOPT) [@gong2022gopt_transformer_pronunciation_assessment]

Implementation: `projects/P001-gop-baselines/code/p001_gop/gop.py`, `projects/P001-gop-baselines/code/p001_gop/gopt_model.py`.

Baseline result: phone-level PCC 0.677 +/- 0.013 (5 seeds), MSE 0.073 (Track 05 Phase 1 A3).

### 3.2 Phi-4-Multimodal LoRA Fine-Tuning

*TODO: Complete after Phase 1 experiments.*

Model: `microsoft/Phi-4-multimodal-instruct` (5.8B parameters).

Architecture:

- Audio Encoder: pretrained on multilingual speech
- Audio Projector: aligns audio encoder output to LLM token space
- LoRA adapter: appended to language model layers
- LLM: phi-4 backbone (frozen during LoRA training)

Training configuration (following [@ryu2025phi4_lora_pronunciation]):

- Dataset: SpeechOcean762 (2500 train utterances)
- Batch size: 8, gradient accumulation: 8 (effective batch 64)
- Optimizer: Adam, lr=2e-5
- Output format: JSON with APA scores (accuracy, fluency, prosodic, total)
- Hardware: [TODO: document actual hardware used]
- LoRA rank: [TODO]
- Training epochs: 3-4 (evaluate per-epoch on test split)

### 3.3 Qwen2-Audio-7B Fine-Tuning

*TODO: Complete after Phase 2 experiments.*

Model: `Qwen/Qwen2-Audio-7B-Instruct` (7B parameters).

Training configuration:

- Dataset: SpeechOcean762 (2500 train utterances)
- LoRA fine-tuning (PEFT)
- 4-bit quantization for 12GB VRAM compatibility
- Output: sentence-level scores + optional phone-level scoring

### 3.4 Granularity Normalization for Fair Comparison

LLM systems natively produce sentence-level scores. Our GOPT pipeline produces phone-level
scores. To enable fair comparison, we adopt two strategies:

1. Aggregate phone scores to sentence level: take the mean of phone-level GOPT predictions
   over the utterance, then compute sentence-level PCC. This yields P3-B in the ablation.

2. Prompt the LLM for phone-level scores: using a structured prompt that requests
   per-phone accuracy assessments. This is novel and may be unreliable.

All results tables will explicitly label granularity (phone vs sentence).

## 4. Experimental Setup

### 4.1 Dataset

SpeechOcean762 [@speechocean762]: 5000 English utterances from 250 Mandarin-L1 speakers,
standard 2500/2500 train/test split. Phone-level accuracy scores (0-2) and sentence-level
scores (0-10) for accuracy, fluency, prosody, completeness, and total.

We exclude `completeness` from sentence-level evaluation (all test values are 10, per [1]).

### 4.2 Metrics

- Primary: Pearson correlation coefficient (PCC) between predicted and human scores.
- Secondary: MSE (phone-level only).
- Granularity must be explicit in all tables: phone-level PCC vs sentence-level PCC.
- 95% CI reported for all PCC values.

### 4.3 Reproducibility

- Minimum 3 seeds for all LoRA fine-tuning runs.
- Log: per-epoch test PCC, training loss, VRAM peak, wall-clock time.
- Config snapshots and model checkpoints archived.
- All runs logged to MLflow at `mlflow.peacockery.studio`.

## 5. Results

*TODO: Complete after all phases.*

### 5.1 Phase 1: Phi-4 LoRA Reproduction

| Run ID | Model | Strategy | Epoch | Accuracy PCC | Fluency PCC | Total PCC |
|---|---|---|---|---|---|---|
| P1-A | Phi-4 | LoRA only | best | TBD | TBD | TBD |
| P1-B | Phi-4 | Unfreeze | best | TBD | TBD | TBD |
| Paper [1] | Phi-4 | LoRA only | 3 | 0.656 | 0.727 | 0.675 |
| Paper [1] | Phi-4 | Unfreeze | 4 | 0.743 | 0.717 | 0.666 |

All above: sentence-level PCC.

### 5.2 Phase 2: Qwen2-Audio-7B Results

| Run ID | Model | Strategy | Accuracy PCC | Fluency PCC |
|---|---|---|---|---|
| P2-A | Qwen2-Audio-7B | Zero-shot | TBD | TBD |
| P2-B | Qwen2-Audio-7B | LoRA | TBD | TBD |
| Paper [5] | Qwen2-Audio-7B | LoRA | ~0.77 | TBD |

All above: sentence-level PCC.

### 5.3 Phase 3: Granularity-Matched Comparison

| Run ID | System | Granularity | Accuracy PCC |
|---|---|---|---|
| P3-A | GOPT+GOP-SF (ours) | Phone-level | 0.677 |
| P3-B | GOPT+GOP-SF (ours) | Sentence-level (mean pooled) | TBD |
| P3-C | Best LLM (Phase 1-2) | Sentence-level | TBD |
| P3-D | Best LLM (Phase 1-2) | Phone-level | TBD |

### 5.4 Phase 4: Inference Cost

| System | Params | VRAM (inf) | Latency/utterance |
|---|---|---|---|
| GOPT+GOP-SF (ours) | ~301M | TBD | TBD |
| Phi-4 LoRA | 5.8B | TBD | TBD |
| Qwen2-Audio-7B LoRA | 7B | TBD | TBD |
| Qwen2-Audio-7B 4-bit | 7B (quant) | TBD | TBD |

## 6. Discussion

*TODO: Complete after results.*

Key questions to address:

- Is the LLM Accuracy PCC advantage real, or is it an artifact of granularity mismatch?
- When both systems are evaluated at sentence level, which wins?
- At phone level, which is more precise?
- What is the practical deployment trade-off? A 7B LLM vs a 300M CTC model.
- Is the LLM learning pronunciation quality, or is it learning ASR-level content recognition?
  (The Phi-4 paper notes negative correlation between PER and accuracy score: -0.446 for GT,
  -0.354 for predicted. This suggests the model is partly tracking phoneme recognition accuracy.)

## 7. Limitations and Threats to Validity

- SpeechOcean762 is a single benchmark (Mandarin-L1 speakers). Generalization is unknown.
- LoRA fine-tuning on ~5h of audio risks overfitting; we mitigate with per-epoch PCC tracking.
- Granularity normalization (phone -> sentence mean pool) may not reflect human sentence scores.
- LLM phone-level scoring (P3-D) is novel and may be unreliable without specific training.
- We do not evaluate open-response (non-scripted) scenarios; all results are read-aloud.
- Hardware constraints may limit our reproduction to quantized models, which may differ
  from paper results on full-precision A100.

## 8. Conclusion

*TODO: Complete after results.*

## Reproducibility Appendix (Draft)

Run folders: `runs/track06_*/` (to be created per phase).

Primary artifacts (per run):

- `config.yaml` (model, LoRA settings, training hyperparameters)
- `per_epoch_pcc.tsv` (granularity labeled)
- `inference_profile.json` (VRAM, latency)
- Checkpoint path

## Bibliography

Use `./refs.bib`.
