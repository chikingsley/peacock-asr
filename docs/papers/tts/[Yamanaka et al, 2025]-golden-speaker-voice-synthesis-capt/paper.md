---
title: "Synthesizing True Golden Voices to Enhance Pronunciation Training for Individual Language Learners"
authors:
  - "Ryoga Yamanaka"
  - "Kento Osa"
  - "Akari Fujiwara"
  - "Geng Haopeng"
  - "Daisuke Saito"
  - "Nobuaki Minematsu"
  - "Yusuke Inoue"
citation_author: "Yamanaka et al."
year: 2025
doi: "10.21437/SLaTE.2025-42"
pages: 5
source_pdf: "paper.pdf"
extraction_method: "Manually curated from the local PDF, with section-level summarization and cleanup of extraction artifacts."
extracted_at: "2026-03-07"
llm_friendly: true
---

# Synthesizing True Golden Voices to Enhance Pronunciation Training for Individual Language Learners

## Metadata

- Type: Short conference paper / proof-of-concept study.
- Venue: 10th Workshop on Speech and Language Technology in Education (SLaTE 2025).
- Core idea: Convert native model speech to match the learner's self-perceived voice (SPV), not just their recorded voice (ACV).

## TL;DR

The paper argues that the best pronunciation model may be a native utterance transformed to sound like the learner's internally perceived voice. In a 15-person study, learners generally preferred this SPV-based "true golden voice" over other-speaker voices and often over their own recorded-voice model, with subjective and objective evidence of better imitation. The result is interesting but early: the sample is small, the learners are relatively advanced, and the SPV conversion still introduced audible artifacts.

## Abstract

This study revisits the "golden speaker" idea in pronunciation training by accounting for voice confrontation, the mismatch between a person's recorded voice and the voice they feel they hear internally. The authors synthesize native model speech that first matches the learner's recorded voice and then is further converted toward the learner's self-perceived voice. They report that this SPV-based model is more preferred and often more effective for imitation than recorded-voice or other-speaker alternatives.

## Research Question

Can a pronunciation model that matches a learner's self-perceived voice improve imitation more than models based on the learner's recorded voice or on other speakers' voices?

## Method

- Participants:
  - 15 native Japanese speakers learning English.
  - 9 female, 6 male.
  - Ages ranged from teens to sixties.
  - Included 4 English teachers, 4 English drama club members, and 7 adult learners from other backgrounds.
- Voice modeling:
  - Each participant recorded about 30 minutes of read speech to build a personalized speech-to-speech model using `ElevenLabs`.
  - Each participant also recorded about 10 minutes with multiple sensors to estimate body-conducted components and emulate self-perceived voice.
  - Native General American (GA) speech from one assistant language teacher was converted into each learner's voice space.
- Training comparison:
  - Four model-voice conditions were tested:
    - Participant ACV (`P-ACV`)
    - Participant SPV (`P-SPV`)
    - Same-gender other speaker ACV (`SG-ACV`)
    - Different-gender other speaker ACV (`DG-ACV`)
- Evaluation:
  - Subjective ranking of ease of imitation, ease of comparison, and preferred future training voice.
  - Objective articulatory similarity via PPG-DTW distance.
  - Objective prosodic similarity via pitch correlation.

## Data

- 60 short texts per participant were used to train the speech-to-speech models.
- A 10-minute parallel corpus per participant was collected for ACV-to-SPV conversion.
- The native GA source material was two sets of Harvard Sentences read by one American English speaker in his twenties.
- The pipeline produced 600 GA model samples total:
  - Half matched participant ACV characteristics.
  - Half matched participant SPV characteristics.
- Each participant practiced and was then tested with 20 GA samples, 5 per voice-quality condition.

## Results

- Subjective rankings favored self-matched voices over other-speaker voices:
  - Ease of imitation: `P-SPV = 1.40`, `P-ACV = 1.67`, `SG-ACV = 2.93`, `DG-ACV = 4.00` (`1 = easiest`).
  - Ease of comparison: `P-ACV = 1.80`, `P-SPV = 1.87`, `SG-ACV = 2.80`, `DG-ACV = 3.53`.
- Future preference:
  - `10/15` participants chose `P-SPV`.
  - `5/15` chose `P-ACV`.
  - `0/15` chose direct use of other voices.
- Objective articulatory similarity:
  - Mean PPG-DTW distance was lower for `P-SPV (0.254)` and `SG-ACV (0.250)` than for `P-ACV (0.270)` and `DG-ACV (0.272)`.
  - The paper reports significant advantages for `P-SPV` and `SG-ACV` over `P-ACV` and `DG-ACV` (`p < 0.01`).
- Objective prosodic similarity:
  - When the model used the participant's own timbre/pitch class (`P-ACV` and `P-SPV` combined), pitch imitation was better (`0.756`) than with same-gender other-speaker ACV (`0.697`) or different-gender other-speaker ACV (`0.709`), again reported as significant (`p < 0.01`).
- Interpretation:
  - The authors argue that `P-SPV` works because it better matches the learner's internal voice image, reducing voice-confrontation effects during imitation.

## Limitations / Notes

- The sample is small and includes relatively advanced or highly motivated learners.
- Eight participants were already English teachers or actors/actresses, so they may be more comfortable with voice-focused training than typical learners.
- The SPV synthesis used a two-stage pipeline and the paper explicitly notes audible vocoding artifacts in the SPV-converted output.
- The study used General American as the target pronunciation; the authors note that unfamiliar accents might behave differently.
- Table 2 in the local extraction is slightly messy for the pitch row, but the surrounding text makes the combined own-voice advantage clear.

## Relevance To Peacock

- Highly relevant to personalized model-voice generation for pronunciation coaching.
- Suggests that matching the learner's perceived voice, not only their recorded voice, may matter for user comfort and imitation quality.
- Supports evaluating personalized target voices with both user preference and alignment metrics, not only recognition accuracy.
- Also implies that voice-conversion artifacts could directly affect learning outcomes, so synthesis quality is product-critical.
