---
title: "Efficacy of Artificial Intelligence (AI) Voice Cloning in Phonetic Self-Imitation for L2 Pronunciation Training"
authors:
  - "Ewa Kusz"
  - "Judyta Pawliszko"
citation_author: "Kusz and Pawliszko"
year: 2025
doi: "10.1111/ijal.70014"
pages: 18
source_pdf: "paper.pdf"
extraction_method: "Manually curated from the local PDF, with section-level summarization and cleanup of extraction artifacts."
extracted_at: "2026-03-07"
llm_friendly: true
---

# Efficacy of Artificial Intelligence (AI) Voice Cloning in Phonetic Self-Imitation for L2 Pronunciation Training

## Metadata

- Type: Empirical pronunciation-training study.
- Venue: *International Journal of Applied Linguistics*.
- Population: Polish university learners of English using AI-cloned versions of their own voices for pronunciation practice.

## TL;DR

An 8-week self-imitation regimen using cloned learner voices improved listener-rated comprehensibility and fluency, with the clearest gains immediately after training. Revoicer and Speechify performed similarly, so the paper's main claim is about the self-imitation method rather than a specific tool. The evidence is promising but fragile because the control group was tiny, the task was read-aloud speech, and the delayed comprehensibility results are reported somewhat inconsistently.

## Abstract

The study tests whether AI voice cloning can make phonetic self-imitation more effective for L2 pronunciation training. Polish learners of English practiced by imitating AI-modified versions of their own voices over eight weeks, and their speech was rated for comprehensibility and fluency before training, immediately after training, and again later. The paper reports meaningful gains in both outcomes and argues that accessible consumer AI tools can support individualized pronunciation practice.

## Research Question

- Does AI-assisted phonetic self-imitation improve L2 comprehensibility and fluency over time?
- Does tool choice matter (`Revoicer` vs. `Speechify`)?
- Are any gains retained after training ends?

## Method

- Participants:
  - 25 Polish learners of English total.
  - Experimental group: 21 students (19 female, 2 male), roughly B2-C1.
  - Control group: 4 students from the same population.
- Design:
  - Pre-test, immediate post-test, and delayed post-test for the experimental group.
  - Control group completed only pre-test and immediate post-test.
- Training:
  - 8 weeks.
  - 3 sessions per week.
  - 15 minutes per session.
  - 45 minutes per week, about 6 hours total.
- Tool split:
  - `Speechify`: 10 learners.
  - `Revoicer`: 11 learners.
- Tasks:
  - Read-aloud tongue twisters and paragraph material designed around Polish-to-English pronunciation challenges.
  - Open-ended speech was recorded but not used in the main ratings.
- Rating setup:
  - 22 raters total: 7 native-English ESL teachers and 15 experienced Polish teachers of English.
  - 7-point Likert ratings for comprehensibility and fluency.
  - Inter-rater reliability was reported as excellent for both outcomes (ICC = 0.99).

## Data

- Voice cloning was built from 3-4 minutes of each learner's pre-test speech.
- The main training materials were weekly 2-3 minute cloned recordings containing three tongue twisters and one short paragraph.
- The paper reports 2,813 segmented test audio samples overall, then a rated subset of 320 samples organized into 8 surveys so each participant was represented in each rater's workload.
- Read-aloud items were split into trained and untrained sentences so the authors could test limited generalization.

## Results

- Comprehensibility:
  - Significant time effect: `F(2, 30) = 29.44, p < .001`.
  - Pre-test to immediate post-test improved significantly (`p = 0.005`).
  - Post-test to delayed post-test was not significant (`p = 0.343`).
  - Table 1 reports no significant overall pre-test vs. delayed-post-test difference (`p = 0.163`).
- Fluency:
  - Significant time effect: `F(2, 30) = 44.64, p < .001`.
  - Pre-test to immediate post-test improved (`Mdiff = 0.71, p < 0.001`).
  - Pre-test to delayed post-test also improved (`Mdiff = 0.44, p = 0.043`).
  - Post-test to delayed post-test was not significant (`p = 0.317`).
- Trained vs. untrained material:
  - Trained sentences were rated higher than untrained ones for comprehensibility at all three stages.
  - Fluency was also higher for trained than untrained material at post-test and delayed post-test (`p < 0.001` for both reported comparisons).
- Experimental vs. control:
  - Post-test comprehensibility favored the experimental group (`Mdiff = 1.12, p < 0.001`).
  - Post-test fluency also favored the experimental group (`Mdiff = 1.65, p < 0.001`).
  - However, the groups already differed at pre-test, which weakens the comparison.
- Tool comparison:
  - No significant Revoicer vs. Speechify differences for comprehensibility at pre, post, or delayed test.
  - No significant Revoicer vs. Speechify differences for fluency at pre, post, or delayed test.

## Limitations / Notes

- The control group was very small (`n = 4`) and did not provide delayed post-test data.
- Experimental and control groups were already different at pre-test.
- The tool-specific subgroups were also small.
- The main evaluation used read-aloud speech, not spontaneous speech.
- Recordings were made at home on phones or computers, so acoustic conditions were not tightly controlled.
- The paper's prose says delayed comprehensibility stayed significantly above pre-test, but Table 1 reports `p = 0.163` for overall pre-test vs. delayed post-test. I treated that as a real ambiguity in the local PDF rather than smoothing it over.

## Relevance To Peacock

- Directly relevant to personalized pronunciation feedback using a transformed version of the learner's own voice.
- Suggests the intervention design may matter more than which consumer cloning platform is used.
- Supports evaluating both immediate gains and short-delay retention, not just post-test effects.
- Also warns that read-aloud improvements can overstate real-world impact if spontaneous speech is not tested.
