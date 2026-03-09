# Track 10 Execution Status

This file is the canonical operational status for `P003`.

It exists to prevent one failure mode:

- "`k2` scalar core is optimized" being misread as
- "`P003` end-to-end scoring/eval pipeline is fully optimized"

Those are not the same state.

## Status Labels

- `DONE`: completed and validated
- `PARTIAL`: implemented, but not the full pipeline end state
- `PENDING`: planned, not complete
- `BLOCKED`: cannot proceed without another dependency/result

## Current Reality

### 1. HF Backbone Training

- `w2v-BERT-2.0 600M`: `DONE`
- `wav2vec2-base 95M`: `DONE`
- `HuBERT-base 95M`: `DONE`
- `wav2vec2-large 317M`: `DONE`
- `Parakeet CTC 0.6B`: `PENDING`
- `omniASR CTC 300M v2 phoneme adaptation`: `PARTIAL`

### 2. Scoring Backend

- `k2` scalar denominator replacement: `DONE`
- exact parity for denominator/self-score/occupancy: `DONE`
- topology cache / prewarm flow: `DONE`
- `k2` as default scalar backend in `P003`: `DONE`

### 3. Performance

- scalar GOP core batching across utterances: `DONE`
- topology reuse / disk cache: `DONE`
- full end-to-end `GOPT` pipeline fully optimized: `NOT DONE`
- post-scalar collect/features/orchestration batching: `PENDING`

### 4. Automation

- train -> prewarm -> score chain: `DONE`
- queue next model after scoring: `PARTIAL`
- queueing independent of flaky W&B run state: `DONE` for the current chain via local agent/log watcher

## What Is Actually Finished

The following statement is safe:

- The `k2` scalar core rewrite is complete and production-usable.

The following statement is not safe:

- The full `P003` `--gopt` evaluation path is fully optimized end-to-end.

Why:

- `k2` fixed the old scalar denominator bottleneck.
- The next bottleneck is now downstream CPU-heavy collect/features/orchestration.

## Definition Of Done

`P003` is only "fully optimized" when all of the following are true:

1. `k2` scalar path is default and stable.
2. Warm-cache full `--gopt` run is faster than the old python path on real data.
3. Post-scalar collect/features/orchestration has been profiled and the remaining hot path is known.
4. No hidden queue/manual handoff is required after training finishes.
5. One canonical status file says what is done and what is still open.

Right now:

- `1`: yes
- `2`: yes for bounded warm runs, but live large-model sweeps are still CPU-heavy downstream
- `3`: partially
- `4`: mostly yes, but still being hardened
- `5`: yes, this file

So overall status is:

- `P003` infrastructure: strong
- `P003` scalar core optimization: done
- `P003` full end-to-end optimization: not done

## Next Bottleneck

The next performance target is:

- batch and simplify post-scalar collect/features/orchestration

Not:

- redoing the `k2` denominator math
- re-proving scalar parity

## Active Queue

Current intended run order:

1. score the `P004` conformer
2. train + score `Parakeet CTC 0.6B`
3. train + score `omniASR CTC 300M v2` phoneme path
