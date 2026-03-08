# Track 08 Evidence Ledger: Real-Time Streaming Pronunciation Scoring

Scope:

- Chunked GOP-SF (word-by-word processing as utterances arrive)
- Streaming ASR models as GOP posterior sources
- End-to-end latency characterization

Citation policy:

- Use numbered citations in text: `[1]`, `[2]`, ...
- Use `./refs.bib` as canonical bib source.

---

## 1. Claim Map

| ID | Claim | Evidence Status | Primary Citations |
|----|-------|-----------------|-------------------|
| C1 | CTC phoneme posteriors are locally well-defined per word, enabling word-by-word GOP | Plausible from CTC theory; needs Phase 1 experiment | [5] |
| C2 | Chunked GOP-SF achieves PCC within 0.01 of full-utterance baseline | Unverified; is the main Phase 1 hypothesis | [5], internal |
| C3 | Streaming ASR models (Voxtral, Moonshine v2, Moshi) produce CTC-compatible phoneme posteriors | Unverified; depends on whether they have phoneme CTC heads | [1], [2], [3] |
| C4 | Word-level pronunciation scores can be delivered within 500 ms of word completion | Unverified; target latency from practical requirements | [6] |
| C5 | CoCA-MDD is the only prior system addressing streaming + pronunciation together | Supported by literature survey in 08_REALTIME_STREAMING.md | [6] |
| C6 | Full-utterance processing at ~1-2 s latency may be sufficient for MVP | Open question; no user study data available | — |

---

## 2. External Reference Papers

| # | Paper | Key Result | Status |
|---|-------|------------|--------|
| [1] | Nachmani et al. (2025) "Voxtral" (2602.11298) | Streaming ASR, sub-second latency, matches offline WER, word boundaries available | PDF in papers/ |
| [2] | Defossez et al. (2024) "Moshi" (2410.00037) | Full-duplex speech model; real-time by design; logit access; alignment tooling; fine-tuning hooks | PDF in papers/ |
| [3] | Rybakov et al. (2025) "Moonshine v2" (2602.12241) | Ergodic sliding-window encoder; low-latency edge deployment; on-device feasible | PDF in papers/ |
| [4] | Kyutai (2025) "Hibiki Zero" (2602.11072) | GRPO-trained simultaneous streaming translation; no aligned training data needed | PDF in papers/ |
| [5] | Cao et al. (2026) "Segmentation-Free GOP" (2507.16838) | GOP-SF algorithm used as our feature extractor; phoneme posteriors from CTC | Not in papers/ (see Track 05) |
| [6] | Shi et al. (2021) "CoCA-MDD" (2111.08191) | Streaming MDD via coupled cross-attention; only prior work on streaming + pronunciation | PDF in papers/ |
| [7] | Tang et al. (2025) "Streaming ASR with Decoder-Only LLMs" (2601.22779) | LLM-based streaming ASR with latency optimization | PDF in papers/ |
| [8] | Jain et al. (2025) "Canary 1B v2 / Parakeet-TDT" (2509.14128) | Strong CTC-based streaming ASR; TDT decoding | PDF in papers/ |
| [9] | Timkey et al. (2025) "Delayed Streams" (2509.08753) | Formal streaming seq2seq formulation; chunk cadence and delayed commit | PDF in papers/ |
| [10] | Kyutai (2024) "Hibiki" (2502.03382) | Decoder-only simultaneous translation; strong streaming baseline | PDF in papers/ |

---

## 3. Internal Evidence Anchors

- GOP-SF implementation: `/home/simon/github/peacock-asr/projects/P001-gop-baselines/code/p001_gop/gop.py`
- GOPT scorer: `/home/simon/github/peacock-asr/projects/P001-gop-baselines/code/p001_gop/gopt_model.py`
- Track 05 final results: `/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/final/results/aggregate_summary.tsv`
- Chunked GOP-SF (to be implemented): new module, will link here once created
- Latency harness (to be implemented): will link here once created

---

## 4. Key Technical Details

### GOP-SF and Chunking

GOP-SF (Segmentation-Free Goodness of Pronunciation) computes phoneme posteriors from
CTC log-probabilities without forced alignment. The key question for streaming is whether
the posteriors for a given word's phonemes are contaminated by right-context.

In standard CTC decoding, each frame's posterior depends on the entire sequence (through
softmax normalization and the trained model's receptive field). However:

- **Local receptive field**: transformer models have a finite attention window; in practice
  phoneme posteriors are dominated by nearby frames.
- **Word-level GOP**: if we run GOP-SF on a word-length segment (rather than the full
  utterance), we lose only the long-range context, not local phoneme information.

This needs empirical validation (Phase 1), not just theory.

### Streaming Model Compatibility with GOP

To attach GOP-SF to a streaming ASR backbone, the backbone must expose per-frame
phoneme posterior probabilities. This requires:

1. A CTC head over the phoneme vocabulary (not just word/subword tokens).
2. Frame-level outputs (not just token sequences).

- **Voxtral**: Uses Whisper-style encoder-decoder; streaming via chunked encoder. Probably
  does not natively expose phoneme posteriors. May need a CTC phoneme head fine-tuned on top.
- **Moshi**: Operates on audio tokens + text tokens in parallel streams. Logit access
  documented. Alignment tooling exists. Most likely to support phoneme posterior extraction.
- **Moonshine v2**: Ergodic encoder with sliding window; CTC decoder. Most compatible with
  GOP-SF because it already uses a CTC objective.

### Latency Budget Analysis (Target: 500 ms from word end to score)

Typical word duration: 300-500 ms. Components:

| Component | Estimated time | Notes |
|-----------|---------------|-------|
| Audio buffer flush | 0-50 ms | Depends on chunk size |
| Feature extraction (wav2vec2-bert, GPU) | 30-80 ms | GPU-bound, measured in Track 05 |
| GOP-SF CTC forward-backward | 20-100 ms | CPU-bound, single-threaded |
| GOPT scorer forward pass | 5-20 ms | GPU, small model |
| **Total** | **55-250 ms** | Well under 500 ms budget (hypothesis) |

These are estimates. Phase 4 will provide measured values.

### CoCA-MDD Architecture (Only Prior Streaming + Pronunciation Work)

From 2111.08191 (Shi et al., 2021):

- End-to-end streaming MDD (mispronunciation detection and diagnosis)
- Coupled cross-attention (CoCA) between expected and spoken phoneme sequences
- Streaming via chunk-by-chunk processing with carried state
- Does NOT output continuous pronunciation scores (0-2 range); outputs binary MDD labels
- Not directly comparable to our GOP-SF + GOPT pipeline

Key difference: CoCA-MDD targets MDD (correct/incorrect per phoneme), not APA (continuous
score). Our target is continuous scoring, which requires a regression head, not a
classification head. This gap means CoCA-MDD is a reference architecture, not a baseline
to beat directly.
