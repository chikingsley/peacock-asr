# Track 08 Ablation Plan: Real-Time Streaming Pronunciation Scoring

## Research Question

Can pronunciation scoring be made real-time (word-level feedback within ~500 ms of word
completion) without redesigning the entire pipeline, and if not, what is the minimum
architectural change that gets us there?

## Goal

Characterize the latency-accuracy tradeoff for pronunciation scoring under incremental
(streaming) conditions, starting from the cheapest possible change (chunked GOP-SF) and
escalating only if needed.

## Frozen Setup (inherited from Track 05)

- Dataset: SpeechOcean762 (2500 train / 2500 test, pinned revision)
- Evaluation: PCC with 95% CI, minimum 3 seeds; additionally wall-clock latency per word
- Feature extraction baseline: GOP-SF from CTC posteriors (Track 05 A3)
- Baseline accuracy: PCC 0.677 (phone-level, GOPT scorer, Track 05 Phase 1 A3)

## Phase 1: Chunked GOP-SF (Known-Transcript, Word-by-Word)

**Hypothesis**: GOP-SF can be applied to each word independently as it completes,
because phoneme posteriors are locally computable from a CTC model.

**No new model required.** Reuse the existing wav2vec2-bert backbone and GOP-SF
implementation from Track 05.

### Experiment Design

| Run ID | Input | Scope | Purpose |
|--------|-------|-------|---------|
| P1-A | Full utterance (baseline) | All phones | Reproduce Track 05 A3 accuracy |
| P1-B | Word-chunked (word boundaries from forced align) | Per-word phones | Accuracy delta: chunked vs full |
| P1-C | Word-chunked (simulated streaming: left-context only) | Per-word phones | Accuracy delta: no right-context |
| P1-D | Word-chunked + 200ms right lookahead | Per-word phones | Cost of small lookahead window |

**Latency measurement protocol** (establish before running):

- Metric: time from last sample of word audio to score output (ms)
- Hardware: gmk-server (RTX 5070, 12 GB VRAM), single-stream
- Measure: feature extraction time + GOP-SF time + scorer forward pass
- Report: mean and p95 over all words in SpeechOcean762 test set

**Key question for P1**: Does removing right-context hurt PCC significantly?
If PCC drops < 0.01, chunked GOP-SF is viable for known-transcript scoring
and we have a paper with minimal engineering.

**Implementation needed**:

- Forced-alignment word boundary extraction (already have CTC occupancy, derive boundaries)
- Chunked feature extraction path: run GOP-SF per word segment, not full utterance
- Latency instrumentation wrapper around the existing pipeline

Expected effort: 3-5 days
Expected accuracy drop (hypothesis): < 0.01 PCC (CTC posteriors are locally well-defined)

### Decision Rule

- If P1-B PCC drop vs P1-A < 0.01: chunked GOP-SF works, paper is viable from Phase 1 alone.
- If P1-B PCC drop 0.01-0.03: acceptable with lookahead (P1-D), continue to Phase 2.
- If P1-B PCC drop > 0.03: chunked approach is insufficient, must go to Phase 2/3.

---

## Phase 2: Streaming ASR Models as CTC Posterior Sources

**Hypothesis**: Streaming ASR models (Voxtral, Moonshine v2, Moshi) produce CTC-quality
phoneme posteriors in real time, enabling GOP-SF to run on their outputs.

**Context**: Our current GOP-SF uses wav2vec2-bert (offline). Streaming models use
different architectures (causal attention, sliding window). We need to verify that their
posteriors are usable for GOP.

### Experiment Design

| Run ID | Backbone | Streaming Type | Purpose |
|--------|----------|---------------|---------|
| P2-A | Moonshine v2 | Ergodic sliding window | Low-latency edge candidate |
| P2-B | Voxtral | Chunked CTC decode | Production-grade streaming ASR |
| P2-C | Moshi | Full-duplex token stream | Full real-time dialogue substrate |

**Evaluation**:

- Phoneme posterior quality: compare posterior distributions vs offline wav2vec2-bert
- GOP-SF accuracy: PCC on SpeechOcean762 using streaming-model posteriors
- Latency: end-to-end from audio chunk to score

**Key question for P2**: Which streaming backbone produces posteriors closest in quality
to offline wav2vec2-bert? This determines Phase 3 substrate choice.

**Implementation needed**:

- Adapter layer: map each streaming model's output format to GOP-SF's expected input
- May require CTC head fine-tuning if model does not have phoneme-level posterior output
- Latency benchmarking harness (reuse from Phase 1)

Expected effort: 1-2 weeks per backbone (P2-A first, cheapest)

### Decision Rule

- If any P2 model achieves PCC within 0.02 of offline baseline AND latency < 500 ms: viable.
- If no streaming model reaches threshold: full streaming architecture (Phase 3) is required.

---

## Phase 3: Pronunciation Scoring Head on Streaming Backbone

**Hypothesis**: Attaching a lightweight scoring head to a streaming model (Moshi or Voxtral)
enables end-to-end real-time pronunciation feedback without a separate GOP-SF stage.

**This phase only runs if Phase 2 shows a viable backbone.**

### Experiment Design

| Run ID | Backbone | Scoring Head | Training | Purpose |
|--------|----------|-------------|----------|---------|
| P3-A | Best from P2 | Linear probe (frozen backbone) | SpeechOcean762 | Minimal adaptation |
| P3-B | Best from P2 | GOPT transformer (frozen backbone) | SpeechOcean762 | Full scorer on top |
| P3-C | Best from P2 | GOPT transformer (LoRA backbone) | SpeechOcean762 | Joint fine-tune |

**Training setup**:

- Freeze backbone, train scorer head only (P3-A/B)
- LoRA fine-tune backbone + scorer (P3-C, only if P3-B insufficient)
- Dataset: SpeechOcean762 train split (same as Track 05)
- Evaluation: PCC + latency

Expected effort: 2-3 weeks

### Decision Rule

- If P3-B PCC >= 0.65 at latency < 500 ms: sufficient for paper.
- If P3-C required: assess whether LoRA gain justifies compute cost.

---

## Phase 4: Latency Benchmarking and Real-Time Feasibility

**Run regardless of which path was taken (P1-only or P1+P2+P3).**

Goal: characterize what "real-time" means for pronunciation feedback in practice.

### Measurements

| Metric | Target | Method |
|--------|--------|--------|
| Word-level score latency | < 500 ms from word end | Instrumented pipeline |
| Score stability | < 0.1 score swing on successive partial utterances | Run on held-out examples |
| Throughput | >= 1x real-time on single GPU | gmk-server RTX 5070 |
| On-device feasibility | Inference on CPU or small GPU | Moonshine v2 only |

### Stress Tests

- Long utterances (> 10 words): does latency accumulate?
- Fast speech: does chunking degrade accuracy more at high word rate?
- Non-native L2 accents: does streaming accuracy drop more than offline?

Expected effort: 2-3 days

---

## Paper Structure Preview

This ablation generates the paper naturally:

- **Section 3**: Chunked GOP-SF (Phase 1 results)
- **Section 4**: Streaming backbone evaluation (Phase 2, if run)
- **Section 5**: End-to-end streaming system (Phase 3, if run)
- **Section 6**: Latency analysis (Phase 4)
- **Discussion**: What is the minimum change needed for real-time scoring?

Even a null result from Phase 1 (chunked GOP-SF does not degrade accuracy) is a clean
publishable contribution: it establishes that known-transcript real-time scoring is
feasible without a streaming architecture.

## Deliverables per Run

- Config snapshot (model, chunking strategy, latency target)
- MLflow run ID + metrics JSON (PCC, latency mean/p95)
- Wall-clock time, GPU hours
- One-line summary for results table
- Random seeds used (minimum 3 for accuracy; latency measured deterministically)
