# 07: Real-Time Pronunciation Scoring Blueprint (Voxtral + Kyutai)

## Goal

Deliver accurate per-word pronunciation scores with near-real-time latency, then evolve to open-ended streaming scoring without requiring a reference transcript.

Target constraints:
- Score latency: <500 ms after word completion
- Accuracy: approach human ratings on speechocean762/L2-ARCTIC (PCC/SCC and calibration)
- Streaming cadence: 80-160 ms backend update steps

## What We Have Today

### Voxtral path (`references/ADVANCED-transcription/speech-to-text/voxtral`)
- Modal fine-tuning and serving scripts are present.
- Current serving path is transcription-focused and does not emit pronunciation scores or calibrated confidences.
- Useful for production-grade streaming infrastructure experiments and throughput baselines.

### Kyutai path (`references/ADVANCED-transcription/speech-to-text/kyutai`)
- Real-time/streaming-oriented setup with delayed-stream style assumptions.
- Existing forced alignment + timestamp alignment utilities.
- Fine-tuning/eval code already exposes logits and training hooks.
- Best current substrate for a first scoring implementation.

## Recommended Technical Path

### Phase A: Supervised Scoring Core (fastest path)

Objective:
- Build a model-agnostic scoring module:
  - inputs: token logits + alignment + transcript hypothesis
  - outputs: per-word score (0-5), uncertainty, calibration metadata

Implementation notes:
- Start in Kyutai fine-tuning code to reduce integration risk.
- Use a small MLP/regressor over aligned word spans and confidence features.
- Add calibration layer (temperature/isotonic) post-training.

Why first:
- Gives a measurable pronunciation signal before RL complexity.

### Phase B: Near-Real-Time Inference

Objective:
- Stream chunked audio and emit provisional then finalized per-word scores.

Implementation notes:
- 80-160 ms frame updates.
- Delay commit until sufficient right-context (e.g., 320-640 ms).
- Add score smoothing to avoid UI flicker.

Why second:
- Real-time UX without waiting for full open-ended RL stack.

### Phase C: Voxtral Open-Ended Scoring Head

Objective:
- Attach pronunciation head to Voxtral Realtime hidden states at `[W]` boundaries.

Implementation notes:
- Freeze most base weights, train lightweight head + selective LoRA.
- Multi-task objective: ASR token loss + pronunciation score loss.
- Keep scoring head lightweight to preserve latency.

Why third:
- Moves from reference-driven scoring toward open-ended free speech scoring.

### Phase D: RL Alignment (quality refinement)

Objective:
- Align model scoring behavior toward human judgment consistency.

Implementation notes:
- DPO first (more stable), then GRPO.
- Reward sources:
  - supervised human scores (speechocean762 etc.)
  - consistency and calibration terms
  - optional speech reward model (GSRM/SpeechJudge-style)

Why fourth:
- RL works best after baseline score signal is already strong and calibrated.

## Evaluation Stack

Primary:
- PCC/SCC at word and utterance levels
- MAE for absolute score error
- Calibration error (ECE / reliability curves)

Operational:
- Time-to-first-score
- Time-to-final-score
- Real-time factor (RTF)
- Score stability under streaming revisions

Ablations:
- Kyutai-only vs Voxtral-head
- No-RL vs DPO vs GRPO
- With/without calibration stage

## Paper Priorities for This Track

Must-use (already local):
- `2602.11298` Voxtral Realtime
- `2602.13891` GSRM
- `2509.02915` LoRA pronunciation assessment
- `2509.01939` GRPO for ASR
- `2505.04113` preference-pair data methodology
- `2511.07931` SpeechJudge
- `2510.00743` MOS-RMBench
- `2308.12490` MultiPA

Useful adjacent (already local across `docs/papers/*`):
- `2506.12067` logit-based GOP analysis
- `2507.16838` segmentation-free GOP
- `2506.19315` JCAPT

## Immediate Build Tasks (next 1-2 weeks)

1. Add pronunciation scoring head interface in Kyutai train/eval path.
2. Implement offline benchmark script producing PCC/SCC + calibration plots.
3. Add streaming inference harness with provisional/final score events.
4. Mirror scoring interface in Voxtral serving path for A/B latency and quality comparison.
5. Prepare DPO pair-construction pipeline from supervised scores.

## Main Risk

Open-ended scoring can drift toward ASR confidence rather than pronunciation quality.

Mitigation:
- Explicitly train score head against pronunciation labels, not transcription confidence.
- Keep separate calibration for score quality.
- Track disagreement cases where transcript is correct but pronunciation is weak.
