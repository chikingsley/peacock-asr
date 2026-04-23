---
title: "TADA: A Generative Framework for Speech Modeling via Text-Acoustic Dual Alignment"
authors:
  - "Trung Dang"
  - "Sharath Rao"
  - "Ananya Gupta"
  - "Christopher Gagne"
  - "Panagiotis Tzirakis"
  - "Alice Baird"
  - "Jakub Piotr Cłapa"
  - "Peter Chin"
  - "Alan Cowen"
citation_author: "Dang et al."
year: 2026
doi: "10.48550/arXiv.2602.23068"
pages: 10
source_pdf: "paper.pdf"
extraction_method: "Manually curated from the local PDF and arXiv source bundle, with section-level summarization and cleanup of extraction artifacts."
extracted_at: "2026-03-10"
llm_friendly: true
---

## Metadata

- Type: arXiv preprint / generative speech modeling paper.
- Venue status: arXiv preprint; the source bundle uses the `Interspeech` 2026 camera-ready class.
- Core idea: compress each text token into one aligned acoustic representation so a speech model can run at text-like step rates instead of fixed audio-frame rates.

## TL;DR

TADA proposes a 1:1 text-acoustic tokenization scheme for speech generation: one autoregressive step per text token, with a continuous acoustic vector covering that token's full duration. The main payoff is efficiency and reliability rather than pronunciation diagnosis: the paper reports competitive voice cloning, zero catastrophic hallucinations under its CER-based criterion, and much lower effective frame rates than typical codec-token systems. For Peacock, this is more relevant to personalized reference audio, corrective playback, and learner-facing speech generation than to replacing GOP-style scoring.

## Abstract

The paper introduces a speech-language modeling framework that synchronizes acoustic representations with text tokens one-to-one. Instead of generating many fixed-rate acoustic tokens per second, TADA aligns the speech signal to text, compresses each text token into a single acoustic latent, and models the result with a Llama-based autoregressive decoder plus a flow-matching head. The reported result is a faster, more stable TTS/voice-cloning system that reduces hallucinations while preserving high-fidelity audio generation.

## Research Question

- Can speech be tokenized so that acoustic generation proceeds at one step per text token without losing too much fidelity?
- Does this synchronous representation reduce hallucination and improve efficiency relative to fixed-frame-rate speech token systems?
- Can a single speech-text model retain useful language-model behavior while also generating high-quality speech?

## Method

- Tokenization pipeline:
  - A `Wav2Vec2-large` CTC aligner maps audio frames to the LLM's subword tokens.
  - An encoder compresses each aligned text span into one continuous acoustic vector.
  - A decoder reconstructs waveform-level detail from those token-aligned vectors.
- Main generative model:
  - Uses `Llama 3.2` 1B and 3B backbones for autoregressive speech-text modeling.
  - Predicts acoustic features plus duration information per text token rather than per fixed frame.
- Guidance / control:
  - `Speech Free Guidance (SFG)` blends text-only and text-plus-speech logits to reduce the language-modality gap.
  - Online rejection sampling uses a speaker-embedding head to reject bad candidates and improve speaker consistency in long-form generation.

## Data

- Training data:
  - `270k` hours of English speech.
  - `635k` hours of non-English speech.
  - Non-English training languages in the paper: Chinese, French, Italian, Japanese, Portuguese, Polish, and German.
- Training transcripts:
  - `Parakeet-TDT-0.6B-v2` for English and European-language transcription.
  - `Whisper-v3` for Chinese and Japanese.
- Evaluation datasets:
  - `SeedTTS-Eval`
  - `LibriTTS-clean` subset for voice cloning
  - `EARS` for long-form expressive generation
  - `Seamless Interaction`, Spoken StoryCloze, and Spoken TopicStoryCloze for speech-text language modeling

## Results

- Token efficiency:
  - `TADA-Codec` operates at roughly `2-3 fps`, versus much higher frame rates for common codec-token baselines.
  - Reconstruction quality stayed competitive despite the much lower token rate.
- Voice cloning:
  - `TADA-1B`: `CER 0.73 / 0.55`, `SIM 77.9 / 80.2`, `oMOS 2.79 / 3.11` on `SeedTTS-Eval / LibriTTSR-Eval`.
  - `TADA-3B-ML`: `CER 0.76 / 0.40`, `SIM 75.1 / 79.9`, `oMOS 2.85 / 3.17`.
- Hallucination claim:
  - Using `CER > 0.15` as the paper's catastrophic-failure threshold, TADA produced `0` hallucinated samples in the reported benchmark, versus `41` for `FireRedTTS-2`, `24` for `Higgs Audio V2`, and `17` for `VibeVoice 1.5B`.
- Long-form generation:
  - Base `TADA-3B` showed some speaker drift.
  - `Text-Free Guidance` and especially `Online Rejection Sampling` improved speaker similarity for long-form expressive speech.
- Spoken language modeling:
  - `SFG` recovered much of the text-only performance drop and brought the text-speech model closer to the base Llama behavior on the spoken reasoning benchmarks.

## Limitations / Notes

- The training data are mostly proprietary, so the paper is harder to reproduce than a pure open-data recipe.
- The paper is about generative speech modeling, not phone-level pronunciation assessment or mispronunciation diagnosis.
- Long-form generation still showed speaker drift before the extra guidance / rejection-sampling tricks.
- The paper's own numbers suggest TADA is especially strong on reliability and efficiency, while perceptual naturalness (`oMOS`) still leaves room for improvement.
- The model-card multilingual support and the paper's multilingual training setup are related but not identical, so language-coverage claims should be checked against the release artifact, not only the paper.

## Relevance To Peacock

- Strongly relevant to personalized CAPT reference audio and corrective playback.
- Plausible fit for `P008` feedback generation, especially if you want learner-specific or style-controlled model speech after scoring.
- Also relevant to `P007` annotation and ABX workflows, where synthetic reference audio can make comparison tasks easier.
- Not a replacement for `P001` / `P003` pronunciation scoring, because it does not directly solve phone-level error detection, GOP extraction, or interpretable CAPT scoring.
- The multilingual angle is interesting for L2 work, but the paper alone does not remove the need for your canonicalizer, aligner, and scorer split.
