---
title: "Speculative Speculative Decoding"
authors:
  - "Tanishq Kumar"
  - "Tri Dao"
  - "Avner May"
citation_author: "Kumar et al"
year: 2026
doi: null
pages: 33
source_pdf: "paper.pdf"
extraction_method: "Manual rewrite from the local paper.pdf, using the nearby extracted markdown only to confirm definitions, tables, and section structure."
extracted_at: "2026-03-07T20:03:00-08:00"
llm_friendly: true
---

# Title

Speculative Speculative Decoding

## Metadata

- Authors: Tanishq Kumar, Tri Dao, Avner May
- Citation author: Kumar et al
- Year: 2026
- DOI: Not stated in the local PDF
- Pages: 33
- Source PDF: `paper.pdf`
- Venue/status: arXiv preprint (`arXiv:2603.03251v1`, `cs.LG`)

## TL;DR

Standard speculative decoding still has a sequential bottleneck: the draft model must wait for verification to finish before preparing the next speculation. This paper removes that dependency by having the draft model predict likely verification outcomes and precompute speculations for them on separate hardware.

The concrete algorithm, `Saguaro`, is lossless and combines three ideas: geometric cache fan-out over likely verification outcomes, cache-aware sampling to make bonus tokens easier to predict, and a fallback strategy that changes with batch size. On the paper's main `Llama-3.1-70B` setup, SSD averages `1.58x` the throughput of their speculative-decoding baseline and `4.68x` over autoregressive decoding.

## Abstract

The paper asks whether speculative decoding can itself be parallelized. Instead of waiting for the verifier to finish before the draft model begins the next round, the draft predicts several likely verification outcomes and speculates for those outcomes in advance. If the actual outcome is one of the cached cases, the next draft is available immediately. The authors formalize this asynchronous framework, identify the main optimization problems it creates, and instantiate the approach as `Saguaro`, which improves end-to-end decoding speed over both ordinary speculative decoding and autoregressive baselines.

## Research Question

Can speculative decoding be made asynchronous and still remain lossless by preparing next-round speculations before the current verification completes?

## Method

- The paper introduces speculative speculative decoding (`SSD`), where a verifier process runs the target model while a speculator process runs a draft model on separate hardware.
- While round `T` is being verified, the speculator predicts likely verification outcomes for that round and pre-speculates round `T+1` for each predicted outcome.
- If the actual verification outcome is already in the speculation cache, the next speculation is returned immediately. Otherwise the system falls back to a backup speculator.
- Verification outcomes are defined by both the number of accepted drafted tokens and the bonus token sampled from the residual distribution.
- `Saguaro`, the paper's optimized SSD instantiation, adds three main optimizations: geometric fan-out over likely acceptance depths, cache-aware sampling that trades some acceptance rate for a higher cache-hit rate, and a batch-aware fallback that switches to a lower-latency backup at larger batch sizes.
- The framework is lossless because cache misses revert to ordinary speculative decoding behavior.

## Data

- Main model family: `Llama-3.1-70B-Instruct` target with `Llama-3.2-1B-Instruct` draft.
- Secondary model family in the appendix: `Qwen-3-32B` target with `Qwen-3-0.6B` draft.
- Hardware: target on `4 x H100 80GB`; SSD draft on a separate `1 x H100`.
- Datasets: `HumanEval`, `UltraFeedback`, `Alpaca`, and `GSM8K`.
- Appendix numerical setup: `512` prompts sampled evenly across those datasets, `512` generated tokens per prompt, greedy decoding, mainly batch size `1`.

## Results

- The paper reports up to `90%` accuracy when predicting bonus tokens for cache construction.
- Geometric fan-out improves both cache-hit rate and end-to-end speed over uniform fan-out, especially at higher temperatures.
- On the main `Llama-3.1-70B / Llama-3.2-1B` setup, end-to-end decoding speed is `283 tok/s` on `HumanEval`, `215` on `UltraFeedback`, `224` on `Alpaca`, and `301` on `GSM8K`, versus `176`, `138`, `145`, and `188` for ordinary speculative decoding and `54.7` for autoregressive decoding in all four cases.
- The same setup averages `255.8 tok/s` for SSD vs `161.8` for SD vs `54.7` for autoregressive decoding, which is `1.58x` over SD and `4.68x` over autoregressive decoding.
- On the `Qwen-3-32B / 0.6B` setup, the average is `203.8 tok/s` for SSD vs `136.8` for SD vs `88.8` for autoregressive, or `1.49x` over SD and `2.29x` over autoregressive.
- The paper-level summary claim is stronger than the appendix table because it also compares against optimized open-source inference baselines: up to `2x` over speculative-decoding baselines and up to `5x` over autoregressive decoding, while improving the throughput-latency Pareto frontier.

## Limitations / Notes

- SSD requires extra draft-side hardware and compute. The latency gains are not free.
- The paper explicitly notes that speculative decoding is generally a poor fit for throughput-bound workloads such as large offline generation jobs; the main win is latency reduction.
- Performance depends heavily on cache-hit rate. At higher temperatures and larger batch sizes, fallback behavior becomes much more important.
- Most main-text experiments are on greedy decoding and batch size `1`; broader serving conditions are less fully explored.
- The paper presents several obvious extensions, including combining SSD with `EAGLE` or token-tree speculation, but does not fully evaluate those hybrids.

## Relevance To Peacock

This is highly relevant if Peacock has any large autoregressive inference stack, especially one where latency matters more than raw offline throughput. The key reusable idea is not a new model architecture, but a scheduling change: overlap draft and verification work across devices and optimize the cache-hit/acceptance tradeoff explicitly.

It is less relevant to speech-dataset curation directly. For Peacock's speech or pronunciation work, the value is mainly infrastructural: faster LLM serving, faster text generation components, or possible inspiration for asynchronous decoding in other sequential generation systems.
