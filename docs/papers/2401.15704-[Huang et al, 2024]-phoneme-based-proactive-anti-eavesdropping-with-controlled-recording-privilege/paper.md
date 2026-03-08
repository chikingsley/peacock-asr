---
title: "Phoneme-Based Proactive Anti-Eavesdropping with Controlled Recording Privilege"
authors:
  - "Peng Huang"
  - "Yao Wei"
  - "Peng Cheng"
  - "Zhongjie Ba"
  - "Li Lu"
  - "Feng Lin"
  - "Yang Wang"
  - "Kui Ren"
citation_author: "Huang et al"
year: 2024
doi: null
pages: 14
source_pdf: "paper.pdf"
extraction_method: "Manual rewrite from the local paper.pdf, using the nearby extracted markdown only to cross-check section boundaries and reported metrics."
extracted_at: "2026-03-07T20:03:00-08:00"
llm_friendly: true
---

# Title

Phoneme-Based Proactive Anti-Eavesdropping with Controlled Recording Privilege

## Metadata

- Authors: Peng Huang, Yao Wei, Peng Cheng, Zhongjie Ba, Li Lu, Feng Lin, Yang Wang, Kui Ren
- Citation author: Huang et al
- Year: 2024
- DOI: Not stated in the local PDF
- Pages: 14
- Source PDF: `paper.pdf`
- Venue/status: arXiv preprint (`arXiv:2401.15704v1`, `cs.CR`)

## TL;DR

InfoMasker is an ultrasonic anti-eavesdropping system that replaces simple jamming noise with phoneme-structured interference. The idea is informational masking: if the injected signal looks enough like speech at the phoneme level, it is harder for both people and ASR systems to recover the protected speech, and harder for denoisers to remove.

The system also supports authorized recording by storing the transmitted noise and using a recovery network to remove it later from privileged recordings. In the paper's tests, this approach consistently beats white-noise-style baselines and remains stronger under denoising and specialized attacks.

## Abstract

The paper argues that prior ultrasonic anti-eavesdropping systems depend too much on energetic masking, which forces higher power, makes the jamming easier to denoise, and usually prevents any legitimate recording. The proposed alternative is a phoneme-based jamming signal that aims to obscure speech structure rather than only overwhelm it with power. The authors combine this signal design with transmission optimization and a hardware prototype, and report substantially stronger ASR degradation than earlier methods while preserving a path for authorized recovery.

## Research Question

Can microphone jamming based on speech-like phoneme structure, rather than simple white or random noise, provide stronger and more denoising-resistant anti-eavesdropping while still allowing authorized users to recover recordings?

## Method

- The system goal is fourfold: jam human and ASR understanding, resist enhancement attacks, stay low-interference and inaudible, and support controlled recording privilege.
- The jamming signal combines three phoneme streams: `S1` contiguous vowels with target-like speaking rate, `S2` vowels with random gaps and speed scaling for temporal variability, and `S3` consonant sequences from LibriSpeech for additional diversity.
- User registration either collects the target user's speech directly or, if only a short sample is available, matches similar voiceprints from a corpus and uses those phoneme materials instead. A multi-user mode averages voiceprints across people in the room.
- The phoneme inventory is augmented by modifying speech rate, `F0` mean, `F0` contour, energy, and per-phoneme time reversal to reduce repetition while preserving phonetic similarity.
- Before transmission, the signal is pre-compensated for the transmitter/receiver frequency response and modulated with `40 kHz` lower-sideband amplitude modulation (`LSB-AM`) to reduce audible self-demodulation.
- Hardware uses a multi-transmitter ultrasonic array with separate carrier and modulated-noise groups to increase coverage.
- For authorized users, a transformer-based recovery network takes the noisy recording plus the stored clean noise reference and predicts a cleaned recording.

## Data

- Main ASR effectiveness tests use a `27,000`-word test set derived from LibriSpeech.
- Registration and voice-matching experiments use LibriSpeech subsets plus short user speech for voiceprint extraction.
- Cross-language tests use English, Mandarin (`AISHELL-1`), Portuguese (Multilingual LibriSpeech), and Japanese (Japanese Versatile Speech).
- Human intelligibility tests use Harvard Sentences.
- Specialized-ASR attack experiments use `TIMIT`.
- Real-world end-to-end tests use two volunteers, multiple smartphones, and an office-room deployment with four transmitter arrays.

## Results

- The paper's headline claim is that the system drives recognition accuracy below `50%` for all tested ASR systems and beats earlier ultrasonic jammers.
- Across six ASR systems (`Amazon`, `Tencent`, `Xunfei`, `Google`, `DeepSpeech`, `WeNet`), the phoneme-based noise outperforms band-limited white noise when `SNR <= 4`, with the gap widening as `SNR` decreases.
- In the real-world smartphone experiment, average `WER` across `SNR` bins `<-4`, `[-4,-2]`, `[-2,0]`, `[0,2]`, `[2,4]`, `>4` is `85.8`, `81.6`, `77.6`, `70.2`, `56.4`, and `42.3`, versus `11.5` for clear speech. The corresponding digital-domain `WER`s are `88.6`, `85.4`, `68.8`, `48.67`, `28.9`, and `17.0`, versus `4.1` for clear speech.
- In the human-perception study, human `WER` rises from `26.69%` on clear audio to `64.28%` with white noise and `75.89%` with the matched phoneme-based noise. Mean opinion score falls from `4.88` to `1.54`. In the over-the-air matched-speaker condition, human `WER` reaches `99.9%` and MOS drops to `0.18`.
- The authorized content-recovery model improves recognition and `SI-SNR` when `SNR < 1`, while attacker-side enhancement without the clean noise reference does not recover the content and often worsens recognition.
- A specialized ASR trained directly on jammed speech still struggles when `SNR <= 1`; below `SNR = 0`, the paper reports unstable convergence for ASRs trained against the proposed noise.
- The proposed noise also outperforms speech-like noise baselines, prior ultrasonic jammers, and a commercial jammer in the paper's over-the-air comparisons. In the office case study, blind source separation does not restore usable recognition.

## Limitations / Notes

- The paper does not present a dedicated limitations section.
- Based on the experiments, effectiveness still depends on `SNR`, recorder placement, room geometry, and how well the registration data matches the protected speaker or speakers.
- Authorized recovery is privileged, not universal: it requires the stored jamming-noise reference and a recovery model.
- Registration is more convenient if the user can provide either balanced speech or at least a short voiceprint sample. The voice-matching fallback is practical, but it is still a dependency.
- The office deployment keeps ultrasound below the WHO-cited `110 dB SPL` limit at `40 kHz`, but the system still relies on broadcasting ultrasound into the environment.
- Inference from the evaluation setup: most tests use read speech or controlled sentences, so robustness on unconstrained spontaneous multi-party speech is plausible but not fully established in this paper.

## Relevance To Peacock

This is not a CAPT paper, but it is relevant if Peacock cares about privacy-preserving speech systems or robustness of ASR to structured interference. The main reusable idea is that phoneme-like interference is much harder to remove than simple white noise, which suggests a useful stress-test direction for speech models.

It is also relevant as an adversarial-audio design pattern: speaker-matched perturbations, denoising-aware evaluation, and privileged recovery with a clean reference. The direct product overlap with Peacock is limited unless Peacock is building on-device speech privacy or wants to benchmark ASR failure modes under speech-shaped masking.
