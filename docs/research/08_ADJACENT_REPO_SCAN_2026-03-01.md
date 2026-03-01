# 08: Adjacent Repo Scan (2026-03-01)

## Scope

Reviewed:
- `references/voxtral-finetune`
- `references/Finetune-Voxtral-ASR`
- `https://github.com/kyutai-labs`
- `https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B`
- `https://github.com/yazinsai/offline-tarteel`

Goal:
- Identify what is directly reusable for real-time pronunciation scoring (open-ended and/or near-real-time), and what is only adjacent.

## Quick Verdict

1. **Kyutai is currently the strongest systems reference** for true streaming speech architecture and production serving patterns.
2. **Voxtral community fine-tune repos are useful scaffolds** for prompt-format fine-tuning, but are not yet robust production training pipelines.
3. **Liquid LFM2.5-Audio is strong for compact real-time speech-text-audio** and architecture ideas (FastConformer + lightweight audio decoder), but no public fine-tune recipe in the referenced repo/model card.
4. **offline-tarteel is highly useful as an engineering playbook** for practical ASR adaptation loops (data manifests, Modal training, benchmark discipline), even though the task is Quran verse retrieval.

## Repo-by-Repo Findings

### A) `references/voxtral-finetune`

What it provides:
- Structured package (`src/voxtral_finetune/*`) with config-driven training.
- Custom collator for Voxtral transcription prompt/completion formatting.
- Custom trainer logic to evaluate generation from prompt-only during eval.
- Dataset adapters (currently narrow support: LibriSpeech clean + MultiMed German).

What to watch:
- Positioning is explicitly experimental/research code.
- Full fine-tuning focus (README notes high VRAM requirement).
- Not a streaming inference/training stack; mainly batch transcription fine-tuning.
- Some hardcoded token IDs (separator/start-of-transcription) that may be brittle across tokenizer/model variants.

Usefulness for this project:
- Good reference for Voxtral prompt construction and label masking.
- Useful starting point for adding score-head supervision in training loops.
- Not enough alone for low-latency pronunciation scoring deployment.

### B) `references/Finetune-Voxtral-ASR`

What it provides:
- Very small, readable scripts (`train.py`, `train_lora.py`) for Voxtral ASR fine-tuning.
- Includes LoRA variant and audio tower freezing example.

What to watch:
- Minimal baseline scripts (small sample slices in code, few hardcoded params).
- No robust experiment management, metrics suite, or streaming integration.

Usefulness for this project:
- Fast prototyping scaffold only.
- Good for validating prompt/label masking mechanics and quick LoRA tests.

### C) Kyutai (`kyutai-labs/*`)

Important repos:
- `delayed-streams-modeling`: Kyutai STT/TTS streaming usage and server configs.
- `moshi`: full duplex speech-text architecture and production stacks.
- `moshi-finetune`: actual LoRA/full training pipeline for Moshi-family models.
- `hibiki`, `hibiki-zero`: streaming speech translation systems with realtime serving patterns.

Key takeaway on your question "do they provide training scripts?":
- **Yes, but not in every repo.**
- `delayed-streams-modeling` is mostly inference/evaluation/serving oriented.
- `moshi-finetune` is the main public fine-tuning/training entry point.

Usefulness for this project:
- Excellent reference for delayed-stream and chunked generation behavior.
- Strong operational patterns for streaming servers and batching.
- Most relevant external blueprint for moving from offline scoring to near-real-time scoring.

### D) Liquid AI (`LFM2.5-Audio-1.5B` + `liquid-audio`)

What is clear from model card/repo:
- FastConformer-based audio encoder.
- End-to-end audio model with interleaved and sequential generation routines.
- Focus on low-latency speech-to-speech and ASR/TTS tasks.

What appears missing (publicly, in scanned sources):
- No obvious end-to-end fine-tuning cookbook equivalent to Kyutai `moshi-finetune`.
- Inference and usage APIs are clear; training adaptation path is not equally surfaced.

Usefulness for this project:
- Strong architectural reference for compact low-latency multimodal design.
- Useful comparative baseline vs Voxtral/Kyutai for latency-size tradeoffs.

### E) `yazinsai/offline-tarteel`

Why it is relevant despite different task:
- Strong engineering rigor around experiment tracking and benchmark reproducibility.
- Concrete FastConformer adaptation scripts (including Modal GPU workflows).
- Practical lessons about what did and did not move accuracy under tight size/latency constraints.

Usefulness for this project:
- High value as a process template (data prep, benchmark harness, trial structure).
- Less direct on open-ended pronunciation scoring model heads.

## Immediate Integration Suggestions

1. Reuse Voxtral collator/training patterns from `voxtral-finetune`, but avoid hardcoded token IDs by deriving IDs from tokenizer special tokens.
2. Use Kyutai delayed-stream assumptions (chunk cadence, delayed commit patterns) for your near-real-time scoring event protocol.
3. Borrow offline-tarteel’s experiment discipline:
   - fixed corpus slices
   - timestamped benchmark JSON outputs
   - strict size/latency reporting per run
4. Keep Liquid AI as a compact architecture benchmark reference, not as your primary fine-tuning base unless/until training guidance is exposed.

## Downloaded Papers Added During This Scan

Added to `docs/papers/streaming_realtime/`:
- `2509.08753_delayed_streams_modeling.pdf`
- `2511.23404_lfm2_technical_report.pdf`
- `2602.11072_hibiki_zero.pdf`
- `2502.03382_hibiki_streaming_translation.pdf`
- `2410.00037_moshi.pdf`

These complement the existing Voxtral/Moonshine/decoder-LLM streaming papers.
