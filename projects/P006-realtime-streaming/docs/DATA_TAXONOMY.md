# Data Taxonomy for Pronunciation Research

This note separates the supervision types used across the repo. The distinction
is operationally important: a dataset that is sufficient to train a phoneme
model is not automatically sufficient to validate a pronunciation scorer.

## 1. Three Supervision Layers

### A. Acoustic / transcript supervision

Form:

- speech audio
- transcript text

What it supports:

- ASR training
- phoneme-CTC training after transcript -> phone conversion
- encoder / backbone adaptation

What it does not provide:

- human pronunciation quality targets

Examples:

- LibriSpeech-style corpora
- any large read speech corpus with reliable transcripts

### B. Phone-sequence / alignment supervision

Form:

- speech audio
- transcript text
- phone sequence and optionally timings

How it is obtained:

- lexicon or G2P for canonical phones
- forced alignment or MFA-style alignment for timings

What it supports:

- phoneme-head training
- phone-level feature extraction
- duration features
- alignment-sensitive diagnostics

What it does not provide:

- human pronunciation scores

Examples in this repo:

- `gilkeyio/librispeech-alignments`
- MFA-derived phone labels used by
  `projects/P003-compact-backbones/code/training/train_phoneme_head.py`

### C. Pronunciation-score supervision

Form:

- speech audio
- transcript / expected text
- human pronunciation scores

What it supports:

- training or validating pronunciation scorers against human judgment
- objective APA benchmarking
- deciding whether one scoring method is actually better than another

Examples in this repo:

- SpeechOcean762

## 2. Repo Mapping

### P003 / P004

Main need:

- A or B

Reason:

- these tracks are about phoneme backbones and phoneme-head training
- they do not require human pronunciation labels to make progress

### P001 / P002

Main need:

- C

Reason:

- these tracks are about scoring speech according to pronunciation quality
- without human pronunciation labels, they can still run heuristics, but they
  cannot claim the same kind of objective validation

### P006

Main need:

- combines A/B for the grounding stack
- still needs C to judge whether the pronunciation scorer survives the move
  from gold transcript to ASR-derived transcript

Reason:

- `P006` is not only an ASR problem
- it is an APA-under-transcript-uncertainty problem

## 3. What G2P Gives You

G2P gives you a phone sequence hypothesis from text.

That is enough to:

- define a target phone inventory
- train a phoneme model when transcripts are available
- map ASR hypotheses into phone sequences

G2P does not give you:

- a pronunciation score target
- a human notion of accent quality
- a guarantee that a phone realization is "good enough" for the chosen accent
  target

## 4. What SpeechOcean-Type Data Adds

SpeechOcean-style data gives you the missing top layer:

- human judgment of pronunciation quality

That is why it matters for `P001`, `P002`, and `P006`.

Without that layer, a system can still be:

- trainable
- demoable
- internally plausible

But it is harder to say:

- whether it is calibrated to human judgment
- whether one scoring mechanism is actually better than another

## 5. Multilingual Implication

For a new language, the situation is usually:

- if you have speech + transcript + G2P/lexicon, you can probably train a
  phoneme model;
- if you also have human pronunciation ratings, you can benchmark and calibrate
  a scorer properly;
- if you lack human ratings, you can still build a working system, but the
  evaluation standard is weaker.

So multilingual expansion is technically feasible before multilingual APA
benchmarking is solved. The engineering path and the scientific-validation path
are different.

## 6. Practical Takeaway

The main boundary is:

- backbone training asks "can the model represent phones well?"
- pronunciation scoring asks "do these scores match human judgment?"

Those are related, but they are not the same supervision problem.
