# Unscripted / ASR-Conditioned CAPT Plan

## Research Question

Can we score pronunciation when the learner is not reading a known transcript,
by using ASR output as the temporary transcript and then scoring pronunciation
against that hypothesis?

This is the "actual thing" for product use:

- the learner speaks freely or semi-freely;
- the system recognizes what was said;
- the system estimates how well it was pronounced;
- the system may optionally judge whether the utterance makes sense in context.

## Why This Is a Separate Task

This is not the same task as `P001`.

`P001` assumes:

- known canonical text;
- known target phone sequence;
- pronunciation scoring against that known target.

Unscripted / ASR-conditioned CAPT instead assumes:

- canonical text is unknown at inference time;
- the target phone sequence must be inferred from ASR output;
- pronunciation scoring must tolerate ASR transcript instability and error.

That changes the error model substantially.

## System Decomposition

An unscripted CAPT system should be evaluated as three linked but separable
layers:

1. Recognition / grounding

- ASR transcript
- partial transcript stability over time
- WER or another transcript-quality proxy

2. Pronunciation scoring

- convert recognized text to phones
- compare speech evidence against that phone hypothesis
- produce phone-, word-, or utterance-level pronunciation scores

3. Semantic / coherence judgment

- does the recognized utterance make sense in the scenario?
- is the answer coherent even if pronunciation was imperfect?

Do not collapse these into one score. They answer different questions.

## Target Product Scenarios

The likely product settings are:

- guided speaking: constrained prompts, but no exact transcript lock
- open-response speaking: learner answers freely inside a scenario
- dialogue / avatar interaction: pronunciation, intelligibility, and semantic
  success all matter

This means the evaluation should include both:

- pronunciation quality
- task success / intelligibility

## Literature Already In Repo

These are the most relevant local references for this track:

1. HiPPO: unscripted / spoken-language pronunciation assessment

- Citekey: `[@yan2025hippo]`
- Local paper:
  `/home/simon/github/peacock-asr/docs/papers/[Yan et al, 2025]-hippo-hierarchical-apa-unscripted-speech/paper.md`
- Why it matters:
  explicitly targets spoken / unscripted pronunciation assessment rather than
  only read-aloud scoring

2. MultiPA: open-response pronunciation assessment

- Citekey: `[@chen2024multipa]`
- Local paper:
  `/home/simon/github/peacock-asr/docs/papers/[Chen et al, 2023]-multipa-multitask-open-response-pronunciation/paper.md`
- Why it matters:
  shows that open-response assessment is already treated as a distinct problem

3. Read to Hear: textual descriptions + LLM scoring

- Citekey: `[@chen2025read_to_hear]`
- Local paper:
  `/home/simon/github/peacock-asr/docs/papers/[Chen et al, 2025]-read-to-hear-a-zero-shot-pronunciation-assessment-using-textual-descriptions-and-llms/paper.md`
- Why it matters:
  explicitly reasons about transcript quality and semantic coherence in the
  assessment pipeline

4. CoCA-MDD: streaming pronunciation work

- Citekey: `[@shi2021coca_mdd]`
- Why it matters:
  useful systems reference for streaming behavior, but it is MDD and not the
  same as continuous pronunciation scoring

5. Voxtral / Moshi / Moonshine / streaming ASR papers

- Citekeys:
  `[@nachmani2025voxtral]`, `[@defossez2024moshi]`,
  `[@rybakov2025moonshine_v2]`, `[@tang2025streaming_llm_asr]`
- Why they matter:
  they are candidate front ends for the grounding layer, not pronunciation
  scorers by themselves

## Proposed Baseline Pipeline

The first paper-grade baseline should be conservative:

1. ASR transcribes the utterance.
2. Transcript is normalized.
3. Transcript is converted to canonical phones.
4. Speech is passed through a phoneme-posterior backend.
5. GOP-style or related scorer evaluates pronunciation against the ASR-derived
   phone sequence.
6. Results are reported with both recognition and pronunciation metrics.

This is the minimum viable unscripted CAPT system.

## Candidate Experiment Ladder

### Phase 1: Offline ASR-conditioned pronunciation scoring

Goal:

- ignore streaming first
- answer whether the task is even viable offline

Protocol:

- run a strong offline ASR model on SpeechOcean762
- use ASR hypothesis instead of gold transcript
- compare pronunciation-score quality against the transcript-known baseline

Primary measurements:

- ASR WER
- pronunciation PCC / MSE
- degradation relative to transcript-known scoring

### Phase 2: Error sensitivity

Goal:

- measure how much pronunciation quality depends on transcript accuracy

Protocol:

- bucket examples by ASR quality
- measure score degradation versus transcript error
- compare oracle transcript vs ASR transcript vs lightly corrected transcript

### Phase 3: Low-latency / streaming grounding

Goal:

- move the offline ASR-conditioned pipeline toward live feedback

Protocol:

- replace offline ASR with a streaming or incremental ASR model
- measure transcript stability and pronunciation-score stability over time

### Phase 4: Conversational layer

Goal:

- add scenario coherence or semantic judging on top of the ASR and pronunciation
  layers

This phase should be kept separate from pronunciation metrics.

## Immediate Recommendation

Do not start with the semantic judge.

The first clean research question is:

- can ASR-conditioned pronunciation scoring work at all, and how much does it
  lose relative to transcript-known scoring?

If that answer is bad, adding an LLM judge will only hide the failure mode.

## Suggested Paper Angle

Working angle:

- "Pronunciation Assessment without a Known Transcript"
- or
- "ASR-Conditioned Pronunciation Scoring for Open-Response CAPT"

That is a legitimate paper question and is materially closer to the product
goal than another purely scripted benchmark.
