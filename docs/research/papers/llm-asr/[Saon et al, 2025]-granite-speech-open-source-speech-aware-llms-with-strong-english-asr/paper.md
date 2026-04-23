---
title: "Granite-speech: open-source speech-aware LLMs with strong English ASR capabilities"
authors:
  - "George Saon"
  - "Avihu Dekel"
  - "Alexander Brooks"
  - "Tohru Nagano"
  - "Abraham Daniels"
  - "Aharon Satt"
  - "Ashish Mittal"
  - "Brian Kingsbury"
  - "David Haws"
  - "Edmilson Morais"
  - "Gakuto Kurata"
  - "Hagai Aronowitz"
  - "Ibrahim Ibrahim"
  - "Jeff Kuo"
  - "Kate Soule"
  - "Luis Lastras"
  - "Masayuki Suzuki"
  - "Ron Hoory"
  - "Samuel Thomas"
  - "Sashi Novitasari"
  - "Takashi Fukuda"
  - "Vishal Sunder"
  - "Xiaodong Cui"
  - "Zvi Kons"
citation_author: "Saon et al"
year: 2025
doi: null
pages: 7
source_pdf: "paper.pdf"
extraction_method: "Manual rewrite from the local paper.pdf and local source files."
extracted_at: "2026-03-15"
llm_friendly: true
---

## Metadata

- Authors: George Saon, Avihu Dekel, Alexander Brooks, Tohru Nagano, Abraham Daniels, Aharon Satt, Ashish Mittal, Brian Kingsbury, David Haws, Edmilson Morais, Gakuto Kurata, Hagai Aronowitz, Ibrahim Ibrahim, Jeff Kuo, Kate Soule, Luis Lastras, Masayuki Suzuki, Ron Hoory, Samuel Thomas, Sashi Novitasari, Takashi Fukuda, Vishal Sunder, Xiaodong Cui, Zvi Kons
- Citation author: Saon et al
- Year: 2025
- DOI: Not stated in the local PDF
- Pages: 7
- Source PDF: `paper.pdf`
- Venue/status: arXiv preprint (`arXiv:2505.08699v2`, `eess.AS`)

## TL;DR

Granite-speech is IBM's speech-aware LLM stack for English ASR and speech
translation. The design is a Conformer CTC encoder plus a windowed Q-former
projector feeding a Granite text LLM with LoRA adapters.

The paper's framing is important: this is not "ASR plus a separate chat model."
It is an LLM-centered speech stack that keeps the text model largely intact in
text-only mode while activating the speech encoder, projector, and LoRA path in
speech mode.

## Abstract

The paper presents compact speech-aware LLMs built by modality-aligning
`granite-3.3-instruct` to speech using public ASR corpora and synthetic speech
translation data. The core claim is that these models are competitive on
English ASR despite relying on public data rather than massive proprietary
collections, while also supporting English-to-X AST for several major languages.

## Research Question

Can a speech-aware LLM built from a strong text model plus a speech encoder and
lightweight adapters achieve competitive English ASR and AST without losing the
underlying text model's normal text capabilities and safety behavior?

## Method

System components in the paper:

- Acoustic encoder: Conformer stack trained with character-level CTC
- Attention pattern: block attention inside the encoder
- Speech modality adapter: window-level Q-former used both to downsample in
  time and project speech embeddings into the LLM embedding space
- Text model: Granite text LLM
- LLM adaptation: LoRA adapters on the attention query/value projections

Operationally, the paper describes two modes:

- speech mode: encoder + projector + LoRA path active for ASR / AST
- text-only mode: underlying Granite text LLM runs without LoRA, preserving the
  normal text stack

## Data

The paper trains on public English ASR corpora plus synthetic translation data.
The listed ASR corpora include:

- MLS English
- GigaSpeech
- YODAS
- SPGI Speech
- CommonVoice 17
- Fisher
- LibriSpeech
- VoxPopuli
- Switchboard
- TED-LIUM
- AMI
- Voicemail
- CallHome

The AST side uses synthetic translations of CommonVoice English into major
European languages plus Japanese and Chinese.

## Results

Paper-level conclusions:

- Granite-speech is strongest on English ASR, which the paper explicitly treats
  as the main target task
- The `8B` model is generally strongest, but the `2B` model remains competitive
  on several corpora
- The window-level Q-former projector beats the simpler projector alternatives
  the paper compares against
- The system stays competitive on English-to-X AST while remaining centered on
  ASR rather than pure translation

Why this matters for the later `granite-4.0-1b-speech` release:

- the 2026 `1B` model card keeps the same overall speech-aware-LLM framing
- the new release adds more multilingual support, smaller size, keyword biasing,
  and the speculative-decoding path described in `2603.11243`

## Limitations / Notes

- This paper is about `granite-speech-3.3` (`2B` and `8B`), not the newer
  `granite-4.0-1b-speech` checkpoint directly
- The paper is primarily English ASR plus AST; it is not a multilingual phone
  recognition paper
- The architecture is LLM-centric, so it is not a natural drop-in replacement
  for a phoneme-posterior backbone in the GOP pipeline

## Relevance To Peacock

- Useful as an ASR / transcript sidecar candidate, especially if the product
  direction needs strong speech-to-text plus later text reasoning in one stack
- Less useful as a direct replacement for `P003` / `P004` phoneme-backbone work
- Most relevant immediate roles:
  - multilingual or semi-multilingual ASR baseline
  - reference point for `P006`-style unscripted CAPT
  - architectural comparison against Canary-Qwen and other speech-aware LLMs
