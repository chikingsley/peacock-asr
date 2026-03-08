---
title: "Speech Synthesis and Pronunciation Teaching"
authors:
  - "Waris Quamer"
  - "Anurag Das"
  - "Ricardo Gutierrez-Osuna"
citation_author: "Quamer et al."
year: 2024
doi: null
pages: 7
source_pdf: "paper.pdf"
extraction_method: "Manually curated from the local PDF, with section-level summarization and cleanup of extraction artifacts."
extracted_at: "2026-03-07"
llm_friendly: true
---

# Speech Synthesis and Pronunciation Teaching

## Metadata

- Type: Encyclopedia chapter / narrative review.
- Venue: *The Encyclopedia of Applied Linguistics* (2nd ed., listed as in press in the local PDF).
- Focus: How modern ASR and TTS can support explicit and implicit feedback in computer-assisted pronunciation training (CAPT).

## TL;DR

Modern pronunciation training can use speech recognition for explicit corrective feedback and speech synthesis for personalized imitation models. The chapter argues that recent deep-learning systems make both directions more practical, but it also makes clear that labeled L2 pronunciation data remain scarce and that many self-imitation findings still come from small studies without strong controls.

## Abstract

This chapter reviews how speech recognition and speech synthesis have changed pronunciation teaching. It contrasts explicit feedback based on mispronunciation detection with implicit feedback based on synthetic or converted model voices, then highlights recent deep-learning systems that combine these ideas in more capable CAPT pipelines.

## Research Question

How can recent advances in speech-to-text and text-to-speech be used to improve pronunciation training, and what do current systems suggest about the next generation of CAPT feedback?

## Method

- Narrative literature review rather than a new experiment.
- Organizes the field into two feedback modes:
  - Explicit feedback via ASR / mispronunciation detection (MPD).
  - Implicit feedback via speech synthesis, accent conversion, and self-imitation.
- Uses two case studies to illustrate current system design:
  - Accent conversion that disentangles segmental, prosodic, and voice-identity information.
  - Joint MPD + TTS training, where speech reconstruction is used as an auxiliary objective.

## Data

- No original dataset is introduced.
- The chapter uses prior datasets and studies as examples, including:
  - LibriSpeech for large-scale ASR training, described as about 1,000 hours from 1,000+ speakers.
  - L2-ARCTIC for MPD, described as 24 hours from 24 non-native English speakers.
- It also discusses prior learner studies involving Chinese learners of English, Japanese learners of Italian, and Korean learners of English.

## Results

- CAPT feedback has historically been constrained by weak ASR robustness on L2 speech and by the small size of phonetically labeled non-native corpora.
- Classical GOP-style scoring is useful but limited by forced-alignment quality and by its weaker handling of insertions.
- The review cites evidence that corrective feedback does not need to be perfect to help learners:
  - In a Wizard-of-Oz study, 66% accurate feedback led to pronunciation gains comparable to 100% accurate feedback, and both outperformed 33% accurate feedback.
- For implicit feedback, the chapter argues that a "golden speaker" can be personalized to the learner, potentially reducing cognitive load during imitation.
- Prior self-imitation studies are summarized as showing gains in comprehensibility, fluency, and prosodic convergence, but the chapter treats that evidence as promising rather than settled.
- In the case studies:
  - Accent-conversion systems can recombine segmental content, prosody, and voice identity without requiring strictly parallel source/target recordings.
  - A multitask MPD + TTS system is reported to outperform a baseline MPD-only system on both seen and unseen utterances, suggesting reconstruction helps the decoder learn better phonetic representations.

## Limitations / Notes

- This is a review chapter, so it does not provide a unified experimental setup or directly comparable metrics across all cited work.
- Several self-imitation studies discussed here lacked control groups, so causal claims are still limited.
- The local PDF does not show a DOI.
- The folder name uses `Ding et al`, but the local PDF lists the chapter authors as Waris Quamer, Anurag Das, and Ricardo Gutierrez-Osuna; the frontmatter follows the PDF.

## Relevance To Peacock

- Strongly relevant to any pronunciation product that combines assessment with personalized audio feedback.
- Supports exploring joint modeling of pronunciation assessment and speech reconstruction rather than treating them as separate systems.
- Reinforces that personalized target voices may be useful for learner engagement and imitation quality.
- Also highlights a core product risk: high-quality MPD still depends on scarce, expensive L2 pronunciation annotations.
