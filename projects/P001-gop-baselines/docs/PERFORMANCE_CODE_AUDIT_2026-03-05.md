# P001 Performance Code Audit

Date: 2026-03-05  
Scope: `projects/P001-gop-baselines/code/p001_gop/cli.py`,
`projects/P001-gop-baselines/code/p001_gop/gop.py`,
`projects/P001-gop-baselines/code/p001_gop/backends/*`,
`projects/P001-gop-baselines/code/p001_gop/evaluate.py`

This note updates the repo's older acceleration ideas against the code that is
actually executing in the live `P001` campaign.

## Bottom Line

The current `P001` wall-clock is not explained by one thing.

Three different costs matter:

1. posterior generation is still fully sequential and per utterance
2. feature-vector extraction is sequential and still does substantial
   Python-side prep per utterance
3. scalar GOP denominator computation remains CPU-heavy and Python-loop-heavy

The old "scalar denominator is the bottleneck" story is still directionally
true, but it is no longer the whole story for `A2`-style feature runs. The
current `xlsr` feature path is also paying for under-batched GPU inference,
transport of large posterior arrays, and sequential feature construction.

## Validated Bottlenecks

### 1. Posterior inference is still one utterance at a time

Relevant code:

- `_process_split()` builds `prepared` by calling `backend.get_posteriors()`
  for each utterance in sequence:
  `projects/P001-gop-baselines/code/p001_gop/scoring/runtime.py`
- `XLSREspeakBackend.get_posteriors()` runs processor -> model -> softmax on a
  single input and returns a CPU NumPy array:
  `projects/P001-gop-baselines/code/p001_backends/xlsr_espeak.py`
- `OriginalCTCGOPBackend.get_posteriors()` has the same structure:
  `projects/P001-gop-baselines/code/p001_backends/ctc_gop_original.py`

Assessment:

- This is not making the GPU work very hard.
- It is a good explanation for why `xlsr` feature runs can feel slow even
  before the scalar denominator pass dominates.

### 2. Feature extraction is sequential and still Python-heavy

Relevant code:

- In parallel mode, `_collect_split_outputs()` calls `compute_gop_features()`
  per utterance after the CPU scalar pass finishes:
  `projects/P001-gop-baselines/code/p001_gop/scoring/runtime.py`
- `_compute_lpr_features_batched()` still has Python work for:
  - building all substitution targets
  - padding them
  - chunking calls to `ctc_loss`
  - filling the final feature matrix in Python
  `projects/P001-gop-baselines/code/p001_gop/gop.py`

Assessment:

- This path is mixed GPU-bound and Python-overhead-bound.
- It is a first-class bottleneck for `A2` and any future feature-heavy
  backbone comparison work.

### 3. Scalar denominator DP is still a real CPU bottleneck

Relevant code:

- `_ctc_forward_denom()` in
  `projects/P001-gop-baselines/code/p001_gop/gop.py`
- helper dispatch through `_step_denom()`, `_step_blank()`,
  `_step_normal()`, `_step_after_arb()`, `_step_arb()`

Assessment:

- This remains the structurally worst part of the scalar GOP path.
- The existing multiprocessing split helps, but it does not change the fact
  that the core algorithm is still Python-controlled dynamic programming.

### 4. The multiprocessing path pays a real memory / IPC tax

Relevant code:

- `_process_split()` stores every posterior matrix in `prepared`, then passes
  full arrays to `ProcessPoolExecutor`
- both current backends return `float64` NumPy arrays after softmax

Assessment:

- The repo is paying to materialize, store, and copy large posterior matrices.
- That is avoidable overhead on top of the useful math.

### 5. Cache lifecycle and cache granularity still leave wins on the table

Relevant code:

- `cmd_run()` loads the backend and dataset before taking advantage of full
  split cache hits
- cache keys currently include score variant and alpha, which duplicates work
  for scalar variant sweeps

Assessment:

- There is dead startup work on cache-hit runs.
- There is also duplicated recompute in scalar-variant experimentation.

## Places Where Older Ideas Need Updating

### 6. The feature path now matters at least as much as the scalar path in some runs

The old mental model was:

- scalar denominator work is the one big problem

The current code says:

- that is still true for scalar runs
- but `A2`-style feature runs can spend a large amount of time in the
  feature-vector path and sequential posterior generation

So the next acceleration tranche should not focus only on `_ctc_forward_denom()`.

### 7. Parquet is useful, but it is not the first speed lever for P001

Relevant code:

- `projects/P003-compact-backbones/code/training/preprocess_features.py` already uses the generic
  Parquet loader
- `projects/P003-compact-backbones/code/training/preprocess_features.py` already uses the generic
  Parquet loader
  with `hf://.../*.parquet` for large training data
- `projects/P001-gop-baselines/code/p001_gop/dataset.py` loads SpeechOcean762 directly from the HF
  dataset repo and fully materializes it into Python `Utterance` objects

Assessment:

- For large training data, Parquet is already the right tool and the repo is
  already using it where it matters.
- For `P001`, the dominant costs are posterior inference, dynamic programming,
  feature construction, and array movement, not tabular scan efficiency.
- Converting SpeechOcean762 to another local format will not be the first
  serious wall-clock win.

## Ranked Rollout Order

### Tier 1: do first

1. Check cache completeness before `backend.load()` and before full dataset
   load/parsing.
2. Split cache layers:
   - posterior-derived baseline state
   - feature vectors
   - scalar variant projections / alpha mixes
3. Keep posterior transport and caching in `float32`; cast to `float64` only in
   the numerically sensitive GOP core.

### Tier 2: do next

4. Batch posterior inference across utterances instead of calling
   `get_posteriors()` one utterance at a time.
5. Reduce Python object churn in `_build_substitution_targets()` and feature
   matrix filling.
6. Rework the `prepared` -> `ProcessPoolExecutor` handoff so it does not rely on
   copying large posterior arrays between processes.

### Tier 3: heavier work

7. Attack `_ctc_forward_denom()` with a compiled or lower-level implementation.
8. Consider Rust/PyO3 for persistent CPU-heavy GOP scalar kernels if the Tier-1
   and Tier-2 fixes still leave too much wall-clock on the table.

## Not The First Target

- `evaluate_gop_feats()` is CPU-heavy, but it is not the main reason the
  long `A2` runs feel slow.
- `k2/icefall` is still strategic for other directions, but it is not the
  first fix for the current `P001` bottlenecks.

## Practical Next Step

If the goal is the best speedup per engineering hour, the next `P001` tranche
should be:

1. cache lifecycle fix
2. cache-layer redesign
3. `float32` transport/caching for posteriors
4. batched posterior inference
5. then re-profile before committing to a deeper rewrite of the denominator DP
