# Lab Goals

This document records the research north star for the repo so that individual
projects do not drift into trying to solve every CAPT problem at once.

## 1. North Star

Build a pronunciation assessment stack that is useful in the real world for L2
language learners:

- works on live or near-live speech;
- supports multiple target languages and dialect/accent targets;
- gives actionable feedback, not just a scalar score;
- can be embedded in an interactive product surface (UI, avatar, dialogue, or
  game loop).

The product goal is broader than any single paper. The research program should
therefore decompose the problem into separable layers.

## 2. System Layers

The full application-level stack has at least four distinct layers:

1. Speech recognition / intelligibility layer

- What words did the system think were said?
- Can the utterance be transcribed reliably enough to ground downstream
  feedback?

2. Pronunciation scoring layer

- Given speech and a target phone sequence, how well was each phone realized?
- This is where GOP-SF, GOPT, ConPCO-style feature enrichment, and phoneme
  backbones live.

3. Semantic / coherence layer

- Does the recognized utterance make sense in context?
- Did the learner produce a coherent answer for the scenario?
- This is not the same task as pronunciation scoring and should not be folded
  into pronunciation metrics.

4. Feedback / interaction layer

- How do scores become visible and useful?
- Examples: word highlighting, phone-level red/yellow/green feedback, avatar
  repair behavior, dialogue branching, delayed correction, or teacher-style
  explanations.

These layers can interact in a product, but they should be evaluated
separately in research.

## 3. Current Project Mapping

### P001: Scripted, transcript-known pronunciation scoring

Question:

- With a fixed phoneme-posterior backend, what downstream scoring method works
  best?

This is the controlled scoring-layer paper. It is not the place to solve
unscripted speech, ASR grounding, or semantic judging.

### P002: Richer feature spaces and alternate scoring losses

Question:

- Do richer features and alternate losses improve pronunciation scoring once we
  move beyond the narrow `P001` baseline stack?

This is where duration, energy, SSL embeddings, ConPCO-style losses, and
related scorer changes belong.

### P003: Better phoneme-posterior backbones

Question:

- Can we train or adapt more efficient and/or better phoneme backbones for the
  pronunciation pipeline?

This is upstream model quality, not scoring-layer design.

### P004: Training from scratch

Question:

- Does training a phoneme model from scratch outperform adapter-style or
  lighter-weight backbone adaptation?

This is still upstream acoustic modeling. It is not the home for unscripted
  CAPT or semantic judging.

### P005: LLM-based pronunciation assessment

Question:

- Can multimodal or text-conditioned LLMs score pronunciation directly, and how
  do they compare with the CTC + GOP family?

This is the best current home for end-to-end LLM scoring and for experiments
where a model is asked to score pronunciation directly from audio and text.

### P006: Unscripted / ASR-conditioned CAPT

Questions:

- If canonical text is not known in advance, can ASR output serve as the
  temporary transcript for pronunciation scoring?
- How much does pronunciation quality degrade when the scoring target comes
  from an ASR hypothesis rather than a known prompt?
- Can such a system run with low enough latency for interactive feedback?

This is the right place for unscripted CAPT and ASR-conditioned scoring.
Low-latency and streaming variants are subproblems inside this project, not the
project definition itself.

## 4. Missing Future Tracks

Two important product-aligned tracks do not map cleanly to a finished paper
workspace yet.

### Future track: Conversational CAPT with semantic judging

Question:

- In an interactive scenario, can we combine pronunciation scoring with
  intelligibility and semantic/coherence judgment to drive useful feedback?

This is where an LLM-based judge or dialogue evaluator belongs. It should not
be treated as a substitute for pronunciation scoring. It is a separate layer
that can sit on top of ASR and pronunciation subsystems.

## 5. Research Boundaries

To keep papers publishable and claims defensible:

- `P001` should stay transcript-known and scoring-layer focused.
- `P002` should stay about richer features / alternate losses.
- `P003` and `P004` should stay about backbone quality and training strategy.
- `P006` should stay about ASR-conditioned / unscripted CAPT, with streaming as
  one systems constraint rather than the sole framing.
- Any ASR-first unscripted system should not be quietly mixed into `P001`.
- Any semantic/coherence judge should be evaluated with its own metrics and
  should not be reported as if it were a pronunciation metric.

## 6. Practical Product View

A production CAPT system may eventually run these components in parallel:

- streaming ASR for live transcript / intelligibility
- phoneme-posterior model for pronunciation evidence
- pronunciation scorer for phone/word feedback
- semantic or scenario judge for conversational coherence
- UI logic for rendering feedback in real time

That integrated system is a valid product goal. It is not one paper.

## 7. Immediate Takeaway

The current repo should treat the near-term research ladder as:

1. Finish `P001` and lock the scoring-layer benchmark.
2. Decide whether `P002` materially beats the `P001` stack.
3. Continue `P003`/`P004` for stronger phoneme backbones.
4. Push `P006` for unscripted ASR-conditioned CAPT.
5. Add conversational and semantic judging only after the ASR and
   pronunciation layers are separable and measurable.
