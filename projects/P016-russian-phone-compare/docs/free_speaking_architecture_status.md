# P016 Free-Speaking Architecture Status

Last checked: 2026-05-09

## What This Project Is Testing

P016 is a no-training free-speaking pronunciation pipeline.

The scoring path is:

```text
audio
  -> Qwen3-ASR-1.7B text hypothesis
  -> lane-specific G2P target phones from that ASR text
  -> ZIPA and XLSR-eSpeak phone recognizers on the same audio
  -> target phones vs recognized phones
  -> PER / PFER / word rows
```

Known dataset text is kept only as audit metadata. It is not used as the scoring
reference, so this still behaves like free-speaking.

## Fresh Dataset Check

Balanced FLEURS sample:

- manifest: `runs/free_speaking_check/manifest.jsonl`
- output: `runs/free_speaking_check/eval/`
- report: `runs/free_speaking_check/report.md`
- samples: 2 English (`en_us`) and 2 Russian (`ru`)

Command:

```bash
uv run python scripts/eval_audio_manifest.py \
  --manifest runs/free_speaking_check/manifest.jsonl \
  --out-dir runs/free_speaking_check/eval

uv run python scripts/summarize_eval.py \
  --eval-dir runs/free_speaking_check/eval \
  --out runs/free_speaking_check/report.md \
  --top-words 20
```

Fresh result:

| language | lane | n | avg PER | avg PFER |
| --- | --- | ---: | ---: | ---: |
| en_us | ZIPA | 2 | 0.1139 | 0.0358 |
| en_us | XLSR-eSpeak | 2 | 0.1695 | 0.0909 |
| ru | ZIPA | 2 | 0.3083 | 0.1150 |
| ru | XLSR-eSpeak | 2 | 0.4759 | 0.2199 |

## Current Answer

Does the pipeline run end to end on a small EN/RU dataset sample?

Yes.

Does the current architecture produce trustworthy learner-facing pronunciation
feedback overall?

No. English is bounded enough to debug. Russian is still diagnostic-only.

## What The Run Shows

1. The free-speaking path itself is viable as an experiment harness.
   Qwen ASR, target G2P, ZIPA, XLSR-eSpeak, alignment, CSV output, and report
   generation all ran on the four-sample FLEURS check.

2. English is close enough to be the calibration gate.
   On the fresh FLEURS check, ZIPA+eSpeak target had low PFER (`0.0358`) and
   moderate PER (`0.1139`). Remaining issues are mostly false-positive surfaces
   like abbreviation insertions, rhotics, and word-bucketing artifacts.

3. Russian is not ready for user feedback.
   ZIPA was better than XLSR-eSpeak on the fresh Russian FLEURS sample, but the
   score is still too noisy. The worst rows include `в`, `из`, `от`, `wi-fi`,
   numerals, and palatalized/sibilant-heavy words.

4. PFER is more useful than raw PER for triage.
   PER is still too literal for this mixed G2P/recognizer setup. PFER helps
   separate small phone-distance differences from harder mismatches, but it does
   not fix bad target generation or alignment errors by itself.

## What Needs To Change Next

1. Add an ASR text-normalization audit layer.
   Dataset truth should stay metadata, but the report needs a better audit for
   harmless text rewrites such as `U.N.` vs `UN`, `wifi` vs `Wi-Fi`, and numeric
   rewrites like `7000` vs `7 тысяч`. These should be classified as text drift
   before pronunciation scoring is blamed.

2. Make target G2P a first-class experiment variable.
   Current target backends:
   - English ZIPA lane: `espeak-ng:en-us`
   - English XLSR lane: `espeak-ng:en-us`
   - Russian ZIPA lane: `charsiu/g2p_multilingual_byT5_tiny_16_layers_100`
   - Russian XLSR lane: `espeak-ng:ru`

   The local environment now has a project-contained MFA install at
   `.mfa/env/bin/mfa`, with MFA's model/cache root under `.mfa/root`. That makes
   an MFA target lane practical, but the free-speaking report above still needs
   to be rerun with that lane before claiming results for MFA.

3. Fix known Russian target problems before scaling up.
   Short/context-sensitive words are still a major failure source. Rows like
   `в`, `из`, and `от` should be audited across Charsiu, eSpeak sentence-level,
   and MFA. If a target backend pronounces a function word like a standalone
   letter/name instead of the in-sentence word, that row is not useful learner
   feedback.

4. Separate phone-equivalence policy from ad hoc normalization.
   Current normalization already handles some observed notation mismatches:
   rhotic vowels, long vowels, dark `l`, `oː -> o ʊ`, and `ɲ -> nʲ`.
   The next version should make this explicit as:
   - exact equivalent
   - small acceptable phone distance
   - real pronunciation difference

5. Fix alignment bucketing for insertions.
   The report shows insertions from neighboring material can be bucketed under
   the previous word, especially around abbreviations such as `UN`. That makes
   word-level blame misleading even when the sentence-level alignment is usable.

6. Run a larger dataset slice only after the above gates.
   The right next larger run is not "all FLEURS". It is a clean EN/RU subset
   with:
   - no or separately-classified ASR text drift
   - target backend recorded per lane
   - worst-word report generated
   - enough samples to decide if ZIPA or XLSR is worth keeping for Russian

## Architecture Direction

Keep the free-speaking architecture:

```text
audio -> ASR -> target phones -> phone recognizer -> phone-distance report
```

Do not move back to prompt/read-aloud scoring for this project. The displayed
text can be useful for prompting the speaker, but the scoring reference should
come from ASR text if the goal is free-speaking.

Do not train a new recognizer yet. The current blocker is target/normalization
and diagnostic reliability, not model capacity.
