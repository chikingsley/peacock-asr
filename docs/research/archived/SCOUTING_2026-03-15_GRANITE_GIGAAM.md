# Granite / GigaAM Scouting

Date: 2026-03-15

## Why this note exists

Quick checkpoint after the paper-asset cleanup pass. The goal is to place the
new ASR model information in the repo without turning `P004` into a general
model-zoo workspace.

## Snapshot

- `ibm-granite/granite-4.0-1b-speech`
  - model card: <https://huggingface.co/ibm-granite/granite-4.0-1b-speech>
  - released March 6, 2026 per the model card
  - compact `1B` speech-language model for multilingual ASR and bidirectional
    AST
  - supported speech languages listed on the model card: English, French,
    German, Spanish, Portuguese, Japanese
  - notable product-facing hooks: keyword list biasing and faster inference via
    speculative decoding

- `ai-sage/GigaAM-v3`
  - model card: <https://huggingface.co/ai-sage/GigaAM-v3>
  - paper: <https://arxiv.org/abs/2506.01192>
  - Russian-first Conformer family with `ssl`, `ctc`, `rnnt`, `e2e_ctc`, and
    `e2e_rnnt` variants
  - model card states `220-240M` parameters and `700,000` hours of Russian
    speech pretraining for the SSL encoder
  - strongest current open Russian-specific ASR lead visible in this repo

- `Self-Speculative Decoding for LLM-based ASR with CTC Encoder Drafts`
  - paper: <https://arxiv.org/abs/2603.11243>
  - this matters because the "new Granite" story is not only a new checkpoint;
    it is also a decoding strategy that uses the CTC encoder as a draft path to
    cut autoregressive cost

## Where these fit

### `P004`

Keep `P004` narrow. The repo docs already define it as the from-scratch phoneme
/ ASR training lane, not the place to collect every promising released model.

Granite is not a direct `P004` replacement. It is a ready-made speech-language
system built around an LLM plus a speech encoder, useful as a benchmark or
sidecar, not as the canonical "train our own phoneme model from scratch" path.

GigaAM is closer to the `P004` question in spirit because it is a strong
speech-modeling result with open weights, but dropping it into `P004` would
still blur two separate jobs:

1. training and validating our own canonical stack
2. benchmarking against released external systems

### `P009`

`P009` is the right first home for the Russian thread.

The repo already uses `P009` to map Russian data provenance, public hours, and
Parakeet / FastConformer-era evidence. GigaAM belongs there first because it is
the most concrete Russian-specific open baseline in view. It sharpens the
question "what public or reproducible Russian data path would be needed if we
wanted a Russian pronunciation or ASR sidecar of our own?"

Granite is relevant to `P009` only as a multilingual comparison point. It is
not the Russian data anchor.

### Pronunciation work

Neither Granite nor GigaAM is a drop-in replacement for the current
`phoneme-model -> GOP-SF -> GOPT` stack. They are transcript / intelligibility
systems first.

That makes them more natural for:

- `P006`-style ASR-conditioned or unscripted CAPT
- sidecar transcription for future pronunciation products
- external ASR baselines for multilingual or Russian experiments

They are less natural as direct substitutes for the phoneme-posterior backbone
work in `P003` and `P004`.

## Immediate practical takeaways

- If the next question is multilingual transcript quality with low serving cost,
  Granite-4.0-1b-speech is worth benchmarking against Canary-Qwen.
- If the next question is Russian ASR or Russian pronunciation-adjacent
  grounding, GigaAM-v3 is the more relevant immediate baseline.
- If the next question is still "can we train a stronger phoneme / ASR backbone
  ourselves," keep that bounded inside `P004` and do not let released-model
  scouting redefine the project.

## Paper asset status

The repo now has structured local paper folders for all three relevant papers
under `docs/papers/`:

- `2505.08699-[Saon et al, 2025]-granite-speech-open-source-speech-aware-llms-with-strong-english-asr`
- `2506.01192-[Kutsakov et al, 2025]-gigaam-efficient-self-supervised-learner-for-speech-recognition`
- `2603.11243-[Saon et al, 2026]-self-speculative-decoding-for-llm-based-asr-with-ctc-encoder-drafts`

The arXiv source bundles were successfully imported for:

- `2505.08699`
- `2506.01192`
- `2603.11243`

Each paper folder now has:

- the local `paper.pdf`
- the arXiv LaTeX source bundle when available
- a manually curated `paper.md` note in the repo's standard research-note shape

The main remaining work is not asset ingestion. It is deciding where, at the
project level, Granite and GigaAM should actually be benchmarked without
turning `P004` into a generic model-scouting bucket.
