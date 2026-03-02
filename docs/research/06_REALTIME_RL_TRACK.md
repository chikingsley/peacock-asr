# 06: Real-Time Pronunciation Research Track

This file is the canonical research status for track 2.
It focuses on evidence, decisions, and open questions.

Last updated: 2026-03-01

## Objective

Build a pronunciation system that supports both use cases:

- Near-real-time per-word scoring with a known transcript.
- Open-ended real-time scoring when no reference transcript is provided.

This track runs in parallel with the classical GOP-focused flow in
`01-05`.

## Decision Snapshot

### Primary near-term bet

Use LoRA-based pronunciation adaptation on multimodal LLMs as the
practical baseline, anchored on `2509.02915`.

Rationale:

- It directly targets simultaneous pronunciation assessment and
  mispronunciation detection.
- It reports strong score correlation with human ratings.
- It avoids full-model retraining and supports fast iteration.

### Primary streaming backbone

Use Voxtral Realtime (`2602.11298`) as the open-ended streaming
candidate.

Rationale:

- Native streaming design with explicit word-boundary behavior.
- Production serving path is available (`vLLM` streaming support).
- Architecture is suitable for attaching a lightweight scoring head.

### Primary implementation substrate

Use the Kyutai path first for scoring experiments.

Rationale:

- Existing alignment and timestamp tooling.
- Readable fine-tuning hooks and logit access.
- Lower integration risk for first scoring loop.

## What Is New In This Iteration

- Added three papers from Mac `Downloads` into
  `docs/papers/rl_alignment_speech/`:
  - `2005.11902v1.pdf`
  - `2407.09209v2.pdf`
  - `2509.02915v1.pdf`
- Folded adjacent repo findings into this file and the execution
  blueprint.
- Reduced overlap between `06`, `07`, and `08`.

## Priority Evidence Stack

### A. Pronunciation scoring baselines

- `2509.02915v1`:
  LoRA fine-tuned Phi-4 multimodal model for pronunciation evaluation.
  This is the most directly actionable baseline.
- `2407.09209v2`:
  Multimodal LLM pronunciation assessment design and prompt strategy.
- `2005.11902v1`:
  ASR-free pronunciation assessment baseline; useful as a
  non-ASR-confidence control.

### B. Streaming architecture references

- `2602.11298_voxtral_realtime.pdf`
- `2509.08753_delayed_streams_modeling.pdf`
- `2410.00037_moshi.pdf`
- `2602.11072_hibiki_zero.pdf`

These define chunk cadence, delayed commit behavior, and serving
patterns for low-latency speech systems.

### C. Alignment and reward modeling

- `2507.09929_dpo_speech_enhancement.pdf`
- `2509.01939.pdf` (GRPO for ASR)
- `2602.13891_gsrm_speech_reward_model.pdf`
- `2511.07931_speechjudge_grm.pdf`
- `2510.00743_mos_rmbench.pdf`

These define practical ways to move from supervised scoring to
preference and RL-style alignment.

## Repo Scan Findings (Folded From 08)

### Most useful now

- `references/ADVANCED-transcription/speech-to-text/kyutai`:
  best current place to add a first scoring module.
- `references/voxtral-finetune`:
  useful collator/training pattern for Voxtral prompt formatting.
- `references/Finetune-Voxtral-ASR`:
  minimal LoRA fine-tuning scripts for quick checks.

### Useful but secondary

- Kyutai public repos (`moshi`, `moshi-finetune`, `hibiki*`):
  strongest reference for true streaming operational patterns.
- `offline-tarteel`:
  strong experiment discipline template.
- Liquid `LFM2.5-Audio-1.5B` references:
  valuable architecture comparison, less direct for fine-tuning flow.

## Current Gaps

- No end-to-end pipeline yet that emits calibrated per-word
  pronunciation scores under streaming updates.
- No shared evaluation harness that tracks quality plus latency in one
  report.
- Risk of learning ASR confidence instead of pronunciation quality.

## Guardrails

- Always evaluate with pronunciation labels and correlation metrics,
  not only WER.
- Track calibration error and score stability during streaming updates.
- Keep transcription quality and pronunciation quality as separate
  outputs.

## Next Actions

- Execute the implementation plan in
  `07_REALTIME_PRONUNCIATION_BLUEPRINT.md`.
- Keep this file as the source of research decisions and paper
  priority.
