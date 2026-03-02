# 07: Real-Time Pronunciation Blueprint

This file is the execution plan for track 2.
It translates research decisions from `06` into concrete delivery work.

Last updated: 2026-03-01

## Outcome Targets

Primary targets:

- Score latency below 500 ms after word completion.
- Strong correlation with human pronunciation scores.
- Stable streaming updates with low score jitter.

Measurement targets:

- Word and utterance PCC/SCC.
- MAE and calibration error.
- Time-to-first-score and time-to-final-score.

## Workstream Split

### Workstream 1: Baseline Reproduction

Goal:

- Reproduce a LoRA pronunciation baseline from `2509.02915v1`.

Deliverables:

- Repeatable training config.
- Baseline metrics report.
- Clear gap summary versus our current stack.

### Workstream 2: Scoring Core In Kyutai Path

Goal:

- Build a reusable scoring module using current alignment primitives.

Inputs:

- Token logits.
- Alignment data.
- Transcript hypothesis.

Outputs:

- Per-word score in `0-5`.
- Uncertainty signal.
- Calibration metadata.

### Workstream 3: Near-Real-Time Inference

Goal:

- Emit provisional and finalized scores during chunked streaming.

Requirements:

- 80-160 ms update cadence.
- Delayed score commit to reduce instability.
- Score smoothing to avoid UI flicker.

### Workstream 4: Voxtral Open-Ended Head

Goal:

- Attach a lightweight scoring head to Voxtral word-boundary states.

Approach:

- Freeze most base weights.
- Train selective LoRA plus scoring head.
- Keep overhead minimal for low-latency serving.

### Workstream 5: Alignment Loop

Goal:

- Improve score behavior with preference and reward alignment.

Sequence:

- DPO first.
- GRPO second.

Reward signals:

- Human score correlation.
- Calibration quality.
- Consistency constraints.

## Milestones

### Milestone A

- Baseline reproduction complete.
- First offline scoring core integrated in Kyutai path.
- Report includes PCC/SCC, MAE, and calibration.

### Milestone B

- Streaming harness emits provisional plus final score events.
- Latency and stability metrics are logged per run.
- Initial frontend-friendly event schema is locked.

### Milestone C

- Voxtral scoring head prototype runs end-to-end.
- A/B comparison against Kyutai path is available.
- Decision made on primary production path.

### Milestone D

- DPO/GRPO alignment runs complete.
- Quality uplift is measured against no-RL baseline.
- Go or no-go decision for RL in production.

## Immediate Task List

1. Add `scoring/` interface and adapters in Kyutai training and eval code.
2. Build a benchmark script that outputs JSON plus plots per run.
3. Define streaming score event schema for provisional and final updates.
4. Add Voxtral inference hook for word-boundary hidden state extraction.
5. Prepare preference pair generation pipeline from supervised labels.

## Risks And Mitigations

Risk:

- Model learns ASR confidence instead of pronunciation quality.

Mitigation:

- Train directly on pronunciation labels.
- Evaluate on disagreement subsets where transcript is correct but
  pronunciation is weak.
- Keep calibration and uncertainty as first-class outputs.

Risk:

- Streaming updates become unstable for the UI.

Mitigation:

- Use delayed commit windows.
- Smooth provisional scores.
- Log revision counts per word as a stability metric.

## Dependencies

Core papers:

- `docs/papers/rl_alignment_speech/2509.02915v1.pdf`
- `docs/papers/rl_alignment_speech/2602.11298_voxtral_realtime.pdf`
- `docs/papers/rl_alignment_speech/2602.13891_gsrm_speech_reward_model.pdf`
- `docs/papers/streaming_realtime/2509.08753_delayed_streams_modeling.pdf`

Code references:

- `references/ADVANCED-transcription/speech-to-text/kyutai`
- `references/ADVANCED-transcription/speech-to-text/voxtral`
- `references/voxtral-finetune`
