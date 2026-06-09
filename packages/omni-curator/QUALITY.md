# Quality thresholds (WER tiers, duration, per-second rates)

The bars a curated dataset is filtered against, recorded once so they are never argued from
memory. The code form lives in `omni_curator/quality.py`; this is the rationale + the numbers.

WER here means the **store-level Scribe-verification score** (`Sample.scribe_wer`, from
`omni_curator.verify`): the stored label scored against a fresh Scribe-v2 transcription of the same
clip. Low = label and audio agree. It is *not* a model's eval WER.

## WER tiers — pick by recording type, not one number for everything

NeMo Curator's tiers are split by how the audio was recorded, because the same Scribe WER means
different things for clean read speech vs. disfluent conversation.

| Recording type | excellent | good | acceptable |
|---|---|---|---|
| **Read / broadcast** (scripted CV, FLEURS, audiobooks) | ≤ 5% | ≤ 15% | ≤ 25% |
| **Conversational / spontaneous** (interviews, calls, drill audio) | ≤ 15% | ≤ 35% | ≤ 60% |

Coarser fallback when only the language's resource level is known: high-resource ≤ 20%, medium
≤ 30%, low-resource ≤ 50%.

NeMo's documented **lenient** preset (its defaults): `max_wer 50%`, `min_dur 0.3 s`,
`max_dur 60 s`, `min_words 1` — a "remove only clearly broken clips" floor, not a quality bar.

Map for our corpora:

- `commonvoice-scripted-*`, `fleurs` → **broadcast** tiers.
- `commonvoice-spontaneous-*`, YouTube drill/show audio → **conversational** tiers.

## Duration

- **Upper bound: 40 s — hard, model-imposed.** OmniASR truncates input audio at 40 s, so a longer
  clip trains a label whose tail has no matching audio. Never export above this. (`OMNI_MAX_DURATION_S`)
- **Lower bound: soft.** NeMo's 0.3 s floor drops segmentation artifacts (sub-word fragments,
  empty clips) that can't carry their label. There is no fixed Omni minimum in seconds — CTC's only
  hard requirement is that a clip emit at least as many encoder frames as its label has tokens, so
  the real floor scales with transcript length, not a constant. A 0.3 s floor is cheap insurance,
  not a model requirement; skip it on corpora whose shortest clip is already well above it.

## Per-second rates (misalignment catcher)

`chars_per_second = len(norm_text) / duration` and `words_per_second = len(norm_text.split()) /
duration`, computed on the **normalized** label. They catch audio↔text misalignment: a transcript
far too dense or too sparse for its duration is almost certainly mislabelled, even when its WER
looks fine. NeMo *defines* both metrics but publishes **no recommended values** — yet they are not
arbitrary either.

**There is a physiological ceiling.** Fluent adults articulate ~5 syllables/sec (range ~3.3–5.9;
fast speakers like Italian peak near 9), and speech carries a near-universal ~39 bits/sec across
languages (fast-syllable languages pack less info per syllable, slow ones more — they converge).
So a transcript implying *faster than humanly possible* speech for its duration is, by definition,
misaligned. `chars/sec = syllables/sec × chars-per-syllable`, so the bound is **script-specific**:
for roughly-phonemic Georgian Mkhedruli (~2–2.5 chars/syllable), ~5 syl/sec → ~12.5 cps expected,
and ~8 syl/sec (fast ceiling) → ~18–20 cps — which is exactly where the Georgian corpus sits
(median 11.7, p99.9 19.1) and why the NVIDIA recipe's ~18 cps cap works.

**How to set the cap for a new language** (portable, not a hand-picked constant):

1. anchor to the physiological ceiling: `cap_cps ≈ 8 syl/sec × the script's chars-per-syllable`; or
2. read the corpus distribution and cap just above **p99.9** (× a small margin).

Both agree, because the corpus distribution already bakes in physiology + script. Note the *primary*
misalignment filter is still the Scribe WER (what the low-resource dataset papers lean on:
CER/WER + edge-CER + alignment-confidence); cps/wps is a cheap physical-plausibility backstop, not
the main gate. Off by default in `Selection`.

## Sources

- WER filtering: <https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-assessment/wer-filtering>
- Audio quality metrics: <https://docs.nvidia.com/nemo/curator/about/concepts/audio/quality-metrics>
- Universal ~39 bits/sec speech rate: <https://www.science.org/content/article/human-speech-may-have-universal-transmission-rate-39-bits-second>
- Speech tempo (syllables/sec): <https://en.wikipedia.org/wiki/Speech_tempo>
- Low-resource dataset filtering (CER/WER + edge-CER + alignment): <https://arxiv.org/html/2406.12674v1>
