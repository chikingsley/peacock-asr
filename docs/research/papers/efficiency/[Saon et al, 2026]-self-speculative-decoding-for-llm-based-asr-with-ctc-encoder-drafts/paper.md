---
title: "Self-Speculative Decoding for LLM-based ASR with CTC Encoder Drafts"
authors:
  - "George Saon"
  - "Samuel Thomas"
  - "Takashi Fukuda"
  - "Tohru Nagano"
  - "Avihu Dekel"
  - "Luis Lastras"
citation_author: "Saon et al"
year: 2026
doi: null
pages: 6
source_pdf: "paper.pdf"
extraction_method: "Manual rewrite from the local paper.pdf."
extracted_at: "2026-03-15"
llm_friendly: true
---

## Metadata

- Authors: George Saon, Samuel Thomas, Takashi Fukuda, Tohru Nagano, Avihu Dekel, Luis Lastras
- Citation author: Saon et al
- Year: 2026
- DOI: Not stated in the local PDF
- Pages: 6
- Source PDF: `paper.pdf`
- Venue/status: arXiv preprint (`arXiv:2603.11243v1`, `eess.AS`)

## TL;DR

This paper turns the CTC encoder inside a speech-aware LLM into a draft decoder
for speculative inference. The core idea is simple: accept confident CTC output
directly when entropy is low, otherwise verify the CTC hypothesis in one LLM
pass, and only fall back to full autoregressive decoding when verification
fails.

The headline result is useful because it is not just a latency trick. On the
reported `1B` LLM + `440M` CTC-encoder setup, the method can both accelerate
inference and improve WER relative to plain autoregressive decoding.

## Abstract

The paper asks whether a speech-aware LLM can use its own CTC encoder as a
cheap draft model for speculative decoding. The proposed three-stage pipeline
first tries direct CTC acceptance using frame-level entropy, then uses one LLM
forward pass to verify the CTC hypothesis under a relaxed likelihood criterion,
and finally falls back to ordinary autoregressive decoding only when needed.
Across nine corpora and five languages, the authors report that this can reduce
WER while also speeding inference.

## Research Question

Can a speech-aware LLM reuse its CTC encoder as a draft path so that ASR
decoding becomes faster than full autoregressive inference without giving up
accuracy?

## Method

Three-step decoding procedure:

1. Run greedy CTC decoding and compute frame-level entropy.
2. If the entropy is below a threshold, accept the CTC hypothesis directly.
3. Otherwise, verify the CTC hypothesis with one LLM forward pass using a
   relaxed token-likelihood criterion.
4. If verification fails, resume full autoregressive decoding from the accepted
   CTC prefix.

Important design points from the paper:

- The draft model is not an extra small network; it is the existing CTC encoder
  already present in the speech-aware LLM stack
- The relaxed LLM verification criterion matters because exact-match
  verification would reject too many plausible CTC drafts
- The two verification stages play different roles:
  - CTC entropy gating gives the biggest speed gains
  - LLM verification recovers accuracy by accepting plausible drafts that would
    otherwise require full AR fallback

## Data

- Evaluation scope: `9` corpora, `5` languages
- Main benchmark emphasis in the paper: Hugging Face Open ASR / English test
  sets plus multilingual FLEURS
- Main reported model setup:
  - `1B` parameter LLM
  - `440M` parameter CTC encoder
- The paper also compares against existing Granite speech models and a newly
  trained `1B` speech-aware model built on top of `granite-4.0-1b-base`

## Results

Headline claims in the abstract:

- `5.58%` WER on the Hugging Face Open ASR benchmark
- `4.4x` improvement in inverse real-time factor in the high-throughput setting
- only `12%` relative WER increase over plain autoregressive search in that
  high-RTFx regime

Paper-level qualitative findings:

- In the high-accuracy regime, the method can improve WER over plain AR because
  CTC and SLM make complementary errors
- Removing the LLM verification stage keeps some speed advantages but cannot
  reach the best WER
- Removing the initial CTC-acceptance stage loses much of the speed advantage
- The method yields a controllable WER / RTFx tradeoff through the CTC and LLM
  verification thresholds

## Limitations / Notes

- The method depends on a speech-aware LLM that already has a frozen
  CTC-trained encoder. It is not a generic drop-in for arbitrary ASR models.
- The paper explicitly limits the method to ASR; it does not claim the same
  approach works for speech translation or spoken QA.
- Verification is utterance-based, so when verification fails the system may
  still have to resume expensive autoregressive decoding for the remaining
  suffix.
- The new value here is decoding strategy, not a new acoustic encoder family.

## Relevance To Peacock

- Directly relevant to Granite-4.0-1b-speech because the model card explicitly
  markets speculative decoding as one of the main new improvements
- Relevant to Peacock wherever transcript quality and serving speed both matter,
  especially for ASR sidecars or low-latency unscripted CAPT
- Not a direct answer to the `P004` question of training a new phoneme / ASR
  backbone from scratch; it is a strong serving-time optimization layered on top
  of an existing speech-aware LLM stack
