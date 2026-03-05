# Chunked GOP-SF for Low-Latency Pronunciation Scoring: Toward Real-Time Feedback

## Abstract

*TODO: Write after Phase 1 experiments complete.*

We investigate whether goodness-of-pronunciation scoring based on CTC phoneme posteriors
(GOP-SF, [@cao2026segmentation_free_gop]) can be applied word-by-word as utterances
arrive, enabling real-time pronunciation feedback without a streaming ASR architecture.
Our existing pipeline achieves PCC 0.677 on SpeechOcean762 with full-utterance processing.
We evaluate word-chunked GOP-SF with and without right-context lookahead, measure end-to-end
latency per word, and characterize score stability under incremental input. If chunked
processing degrades accuracy by less than 0.01 PCC, we establish that real-time known-transcript
pronunciation scoring is achievable with no change to model architecture.

## 1. Introduction

Automatic pronunciation assessment (APA) systems traditionally process complete utterances:
the learner speaks, the system waits for the utterance to end, and then produces scores.
For interactive language learning applications — where feedback is most effective when
immediate — this delay is a practical barrier. The question is whether pronunciation
scores can be delivered within ~500 ms of each word completing, while the learner is still
speaking.

Most APA research, including the current SOTA [@yan2025conpco] and our own prior work
(Track 05), assumes offline processing. The only prior work directly addressing streaming
pronunciation scoring is CoCA-MDD [@shi2021coca_mdd], which targets binary mispronunciation
detection rather than continuous scores.

We take a different starting point: rather than designing a new streaming architecture,
we ask whether our existing GOP-SF pipeline can be applied word-by-word without meaningful
accuracy loss. GOP-SF computes phoneme posteriors from frame-level CTC outputs, which are
in principle locally defined. If phoneme posteriors at a given position are not strongly
contaminated by distant right-context, chunked processing should preserve accuracy.

This paper:

1. Quantifies the accuracy cost of word-chunked versus full-utterance GOP-SF (Phase 1).
2. Characterizes end-to-end word-score latency on current hardware (Phase 4).
3. If chunked accuracy is insufficient, evaluates streaming ASR models as backbone
   replacements (Phase 2) and attachment points for a scoring head (Phase 3).

## 2. Related Work

### 2.1 Pronunciation Scoring and GOP

*Inherit from Track 05 manuscript, Section 2.1.*

### 2.2 Streaming Speech Models

Real-time speech processing has advanced rapidly. Voxtral [@nachmani2025voxtral]
achieves sub-second ASR latency matching offline quality. Moshi [@defossez2024moshi]
operates in full-duplex mode, processing speech token streams in real time. Moonshine v2
[@rybakov2025moonshine_v2] uses an ergodic sliding-window encoder suited for
edge deployment. Hibiki [@kyutai2024hibiki; @kyutai2025hibiki_zero] demonstrates
simultaneous streaming translation. These models were designed for ASR or translation;
attaching pronunciation scoring requires access to phoneme-level posteriors, which
varies by architecture.

### 2.3 Streaming Pronunciation Scoring

CoCA-MDD [@shi2021coca_mdd] uses coupled cross-attention to detect mispronunciations in
a streaming fashion. It processes audio chunk-by-chunk and maintains cross-attention state
across chunks. However, CoCA-MDD targets binary MDD (correct/incorrect per phoneme), not
continuous scores (0-2 range), and does not produce the per-phoneme PCC scores used to
compare against SOTA. The architecture is not directly transferable to our regression
objective.

### 2.4 CTC-Based GOP and Context Sensitivity

*TODO: discuss CTC context sensitivity; is phoneme posterior at position i influenced
by frames at position i+k for large k? What does the architecture's attention window size
imply?*

## 3. Method

### 3.1 Baseline Pipeline (Full-Utterance)

*Describe our GOP-SF + GOPT pipeline. Inherited from Track 05.*

Backbone: wav2vec2-bert (offline). GOP-SF computes phoneme log-posterior ratios from
CTC frame posteriors over the full utterance. GOPT transformer scorer operates on the
resulting 42-dimensional feature vectors.

### 3.2 Chunked GOP-SF

*TODO after implementation.*

For each word in the transcript, extract the corresponding audio segment using CTC-derived
word boundaries. Run GOP-SF on the word segment only. Feed resulting phone-level features
directly to the GOPT scorer.

Variants:

- **No lookahead** (P1-C): strict causal chunking; only audio up to word end is used.
- **200 ms lookahead** (P1-D): include a small right-context window per word.

### 3.3 Streaming Backbone Adaptation (Phase 2)

*TODO after Phase 1 results.*

### 3.4 Scoring Head on Streaming Backbone (Phase 3)

*TODO after Phase 2 results.*

## 4. Experimental Setup

### 4.1 Dataset and Protocol

SpeechOcean762 [@speechocean762]. 2500 train / 2500 test. Phone-level PCC as primary
accuracy metric (95% CI, minimum 3 seeds). Latency measured as time from last audio
sample of each word to score output (ms), reported as mean and p95 over full test set.

### 4.2 Implementation Details

*TODO after implementation.*

Hardware: gmk-server, RTX 5070 (12 GB VRAM), single-stream evaluation.

## 5. Results

### 5.1 Phase 1: Chunked GOP-SF Accuracy

*TODO: Table 1 from ABLATION_PLAN.md Phase 1 (P1-A through P1-D).*

| Run | Input Scope | Right Context | PCC | Latency (ms) |
|-----|------------|--------------|-----|-------------|
| P1-A | Full utterance | Yes | — | — |
| P1-B | Word-chunked (forced align) | No | — | — |
| P1-C | Word-chunked (causal) | No | — | — |
| P1-D | Word-chunked + 200 ms | 200 ms | — | — |

### 5.2 Phase 2: Streaming Backbone Evaluation

*TODO after Phase 2 runs.*

### 5.3 Phase 3: End-to-End Streaming System

*TODO if applicable.*

### 5.4 Phase 4: Latency Profile

*TODO: latency breakdown by component (feature extraction, GOP-SF, scorer).*

## 6. Discussion

*What is the minimum change needed for real-time pronunciation scoring?*
*Does chunked GOP-SF work, or do we need a streaming architecture?*
*What is the practical latency bound for useful pronunciation feedback?*

## 7. Conclusion

*TODO after experiments.*

## References
