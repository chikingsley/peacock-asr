# P008 Architecture

## Goal

Ship a production-oriented known-text CAPT runtime that can:

1. take a target sentence in a supported language
2. produce a canonical pronunciation representation
3. align a learner utterance to the text for UI timing
4. score pronunciation quality using existing scorers
5. render word- and phone-level feedback

`P008` is not a new scorer project. It is the runtime/product layer on top of
the existing scoring work.

## The Four Boxes

### 1. Canonicalizer

Input:

- language code
- target text
- optional target accent / region

Output:

- normalized words
- canonical phones in internal IPA
- stress metadata
- source metadata per word (`dictionary`, `g2p`, `override`)

Design rule:

- dictionary-first
- G2P only on misses, ambiguities, or explicit override

### 2. Aligner

Input:

- audio
- target text
- language code

Output:

- word timestamps
- transcript / confidence
- omission candidates

Primary plan:

- post-utterance alignment is the default product mode
- live streaming is optional and provisional only

Current preferred default:

- `Qwen3-ASR` for transcript and optional live provisional text
- `Qwen3-ForcedAligner` for final post-utterance timestamps on supported languages

Why this is attractive for launch:

- the current Qwen forced aligner explicitly supports all planned launch
  languages in this project: English, Spanish, Italian, French, and Russian
- the ASR and aligner can come from the same family instead of mixing unrelated
  runtime stacks on day one

Why this is separate from scoring:

- the aligner solves timing and UI anchoring
- the scorer solves pronunciation quality
- these should remain swappable

### 3. Scorer

Input:

- audio
- canonical phones
- language metadata

Output:

- phone-level scores
- word-level aggregates
- likely confusion candidates

Current scorer stack:

- `P001` backend
- `GOP-SF`
- `GOPT`

Key constraint from the current repo:

- `P001` still expects canonical phones at the backend boundary through an
  `ARPABET`-shaped interface
- `xlsr-espeak` already maps those phones into an IPA-style backend vocabulary

So `P008` should introduce:

- internal IPA as the canonical cross-language representation
- an English adapter that converts `CMUdict` ARPABET into IPA
- backend adapters that map internal IPA into backend token IDs

### 4. Feedback

Input:

- canonical words/phones
- word timestamps
- phone-level scores
- omission candidates
- likely confusions

Output:

- word summary
- phone issues
- omission summary
- coach payload for text and optional speech output

Initial product stance:

- structured feedback first
- LLM polish later if needed
- TTS playback can be added immediately without making it the source of truth

## Why We Do Not Score From Predicted Phone Strings

The product should not depend on a single "predicted phone sequence" from ASR.

Reasons:

- if the target text is known, expected phones are a stronger anchor than a free
  phone decode
- ASR/phone-recognition errors can look like pronunciation errors
- the existing scorer already computes substitution-oriented evidence internally

In `P001`, the `LPP + LPR` features already encode how plausible each phone
position is under substitutions and deletions. That is better diagnostic
material than a single forced "predicted phone" string.

## Internal Phone Representation

### Canonical internal format

- IPA
- stress stored separately, not dropped
- original source form preserved for debugging

### Why IPA internally

- `ARPABET` is English-specific
- Spanish, Italian, French, and Russian do not fit cleanly into an
  English-centered symbol inventory
- the current `xlsr-espeak` backend already uses IPA-like backend symbols

### English exception

- keep `CMUdict` and `g2p-en` in their native ARPABET form at import time
- convert into internal IPA in the canonicalizer
- preserve original ARPABET for display/debug if useful

## Runtime Modes

### Default mode: after utterance

This is the primary release mode.

Flow:

1. canonicalize target text
2. collect learner utterance
3. align words
4. score phones
5. aggregate to words
6. render feedback

### Optional mode: live provisional + final replacement

This is the future-facing mode.

Flow:

1. stream provisional ASR while the learner is speaking
2. do not attempt high-confidence pronunciation grading mid-sentence
3. after end-of-utterance, replace provisional transcript with final alignment
4. run the better scorer and overwrite the UI with final feedback

Important implementation note:

- current Qwen streaming ASR is useful for provisional text
- timestamps should still be treated as a post-utterance artifact

This keeps the architecture ready for live UX without forcing the scoring layer
to become a streaming system.

## Launch Languages

Initial focus:

- English
- Spanish
- Italian
- French
- Russian

These were chosen because:

- they are product-relevant
- they overlap strongly with the languages supported by the current preferred
  Qwen aligner path

## Source Links

- Qwen3-ASR:
  `https://huggingface.co/Qwen/Qwen3-ASR-1.7B`
- Qwen3-ForcedAligner:
  `https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B`
- NVIDIA NeMo G2P docs:
  `https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/g2p.html`
- MFA pretrained models:
  `https://mfa-models.readthedocs.io/en/latest/`
