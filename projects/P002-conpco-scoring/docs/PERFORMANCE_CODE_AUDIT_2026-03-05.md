# P002 Performance Code Audit

Date: 2026-03-05  
Scope: `projects/P002-conpco-scoring/code/reproduce_conpco.py`,
`projects/P002-conpco-scoring/third_party/ConPCO/src/models/*`,
`projects/P002-conpco-scoring/third_party/ConPCO/src/traintest_eng_dur_ssl_3m_HierBFR_conPCO_norm.py`

This note validates and updates the longer
`PERFORMANCE_ACCELERATION_PLAYBOOK.md` against the current code that actually
executes in this repo.

## Bottom Line

The playbook is directionally correct: this path is not primarily blocked on
raw GPU math throughput. The main costs are host-side orchestration,
Python-heavy tensor preparation, and one expensive "strict reproduction" choice
that keeps an extra full train-set evaluation inside every epoch.

The most important additions from the code audit are:

1. The dataset files are still stored as `float64`, then loaded and copied into
   `torch.float32` tensors at startup.
2. The three SSL feature tensors are concatenated every batch, even though they
   are fixed precomputed features.
3. Strict reproduction mode evaluates the full train loader every epoch purely
   for RNG alignment.

Those three items matter more here than Parquet, Rust, or `k2` in the near
term.

## Validated From The Existing Playbook

### 1. Per-batch Python work in the model is real

Validated.

Relevant code:

- `make_word_pos_mask()` uses nested Python loops and creates a CPU tensor every
  forward pass:
  `projects/P002-conpco-scoring/third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py:412`
- Phone one-hot then linear projection:
  `projects/P002-conpco-scoring/third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py:532`
- Word one-hot then linear projection:
  `projects/P002-conpco-scoring/third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py:553`

Assessment:

- This is a real hotspot candidate.
- It is exactly the kind of pattern that reduces kernel efficiency and keeps
  Python in the critical path.

### 2. Single-GPU `DataParallel` is still present

Validated.

Relevant code:

- `projects/P002-conpco-scoring/code/reproduce_conpco.py:278`

Assessment:

- On a single GPU, this is overhead with no upside.
- This should be removed before more speculative work like compile or custom
  kernels.

### 3. Eval overhead is real

Validated, but the current code is worse than the old summary implied.

Relevant code:

- Full evaluation implementation:
  `projects/P002-conpco-scoring/code/reproduce_conpco.py:198`
- Per-epoch train eval for RNG alignment:
  `projects/P002-conpco-scoring/code/reproduce_conpco.py:437`

Assessment:

- The script evaluates the train loader and the test loader every epoch.
- The train eval exists to match official RNG behavior, not because it is
  needed for day-to-day sweep efficiency.
- For exploratory work, this is one of the biggest removable wall-clock taxes.

## Places Where The Playbook Needs Nuance

### 4. DataLoader tuning is worth trying, but the expected gain is probably overstated

Partially validated.

Relevant code:

- Dataset eagerly loads everything into RAM at construction time:
  `projects/P002-conpco-scoring/code/reproduce_conpco.py:96`
- Loaders are minimal:
  `projects/P002-conpco-scoring/code/reproduce_conpco.py:264`

Why this needs nuance:

- This is not a streaming dataset with expensive per-item decode.
- The training dataset already holds large in-memory tensors; worker processes
  may help somewhat, but the huge advertised gains are not guaranteed here.
- `pin_memory=True` is likely more defensible than assuming a large win from
  high `num_workers`.

My read:

- Keep this in the first optimization tranche.
- Treat it as a measured experiment, not a presumed `1.6x`.

### 5. `torch.compile` should come after tensorization fixes, not before

Partially validated.

Relevant code:

- Python loops and one-hot paths still exist in forward:
  `projects/P002-conpco-scoring/third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py:412`
- Python list-based masking also exists in the ConPCO loss:
  `projects/P002-conpco-scoring/third_party/ConPCO/src/models/conPCO_norm.py:75`

My read:

- `torch.compile(mode="reduce-overhead")` is worth trying.
- But it is less likely to pay off cleanly until the obvious Python control-flow
  in forward/loss is reduced first.

### 6. Rust/PyO3 is not the first rational move for this training loop

Mostly challenge, with a scope caveat.

Why:

- This path is already mostly tensor math once data reaches PyTorch.
- The highest-ROI wins are still inside PyTorch: remove `DataParallel`,
  tensorize masking, stop repeated concatenation, and split strict-vs-fast eval.
- Rust/PyO3 is still attractive for repo-wide CPU-heavy GOP scalar work, but
  that maps more naturally to `P001`/`P003` scalar scoring than this `P002`
  training loop.

## Important Additions The Old Playbook Did Not Emphasize Enough

### 7. The dataset is stored as `float64`, and that is wasteful here

High priority.

Relevant code:

- Loads use `np.load(...)` followed by `torch.tensor(..., dtype=torch.float)`:
  `projects/P002-conpco-scoring/code/reproduce_conpco.py:96`

Observed file shapes:

- `tr_hubert_feat_v2.npy`: `(2500, 50, 1024)`, `float64`, about `976.56 MiB`
- `tr_w2v_300m_feat_v2.npy`: `(2500, 50, 1024)`, `float64`, about `976.56 MiB`
- `tr_wavlm_feat_v2.npy`: `(2500, 50, 1024)`, `float64`, about `976.56 MiB`

Why this matters:

- You pay the disk and memory cost of `float64`.
- Then you immediately cast to `float32` anyway.
- That is unnecessary startup cost and memory pressure.

Recommendation:

- Convert the stored arrays offline to `float32`.
- Then load with `torch.from_numpy(...)` rather than `torch.tensor(...)` where
  safe.

This is one of the clearest low-risk wins in the current path.

### 8. The three SSL tensors are concatenated every batch even though they are static

High priority.

Relevant code:

- Training path:
  `projects/P002-conpco-scoring/code/reproduce_conpco.py:359`
- Eval path:
  `projects/P002-conpco-scoring/code/reproduce_conpco.py:210`

Why this matters:

- The inputs are precomputed and fixed.
- Repeating `torch.cat([ssl2, ssl1, ssl3], dim=-1)` every batch is avoidable.

Recommendation:

- Pre-concatenate once at dataset build/load time into a single `ssl` tensor.
- Drop the three-way per-batch concat from both train and eval.

### 9. ConPCO loss still contains Python list comprehensions on the critical path

High priority.

Relevant code:

- `projects/P002-conpco-scoring/third_party/ConPCO/src/models/conPCO_norm.py:75`
- `projects/P002-conpco-scoring/third_party/ConPCO/src/models/conPCO_norm.py:78`

Why this matters:

- These `torch.tensor([ ... for ... ])` constructions happen inside the batch
  loss, not just once at startup.
- They should be rewritten as tensor operations, for example with `torch.isin`
  or equivalent tensorized masking.

## What Parquet Is And Is Not Likely To Fix

Parquet is useful for Hub distribution, analytics, and efficient tabular access.
It is not the main speed lever for this specific training loop.

Why:

- This script does not repeatedly scan a row-oriented dataset from storage.
- It loads a fixed set of dense NumPy arrays into RAM up front, then spends most
  of its time in repeated train/eval forward passes plus Python-heavy tensor
  prep.

Inference from code:

- Better on-disk layout can help startup and dataset management.
- It will not remove the per-batch Python mask construction, repeated SSL
  concatenation, or extra per-epoch train evaluation.

So:

- Parquet is reasonable for dataset packaging and maybe analytics.
- It is not the first place to spend engineering time if the goal is to cut
  epoch wall-clock or GPU under-utilization.

## Ranked Rollout Order

### Tier 1: do first

1. Split strict reproduction mode from fast exploratory mode.
   - Keep the current train-eval-per-epoch path only for exact paper-faithful
     reruns.
   - For exploration, skip train eval and reduce eval frequency.
2. Remove single-GPU `DataParallel`.
3. Convert stored feature files to `float32`.
4. Pre-concatenate SSL features once instead of every batch.

### Tier 2: do next

5. Tensorize `make_word_pos_mask()`.
6. Tensorize ConPCO loss masking in `conPCO_norm.py`.
7. Replace one-hot + linear with embedding-style lookup, preserving checkpoint
   semantics carefully.
8. Try `pin_memory=True` and modest DataLoader tuning.

### Tier 3: only after the above

9. AMP/BF16 trial with metric checks.
10. `torch.compile(mode="reduce-overhead")` trial.

### Tier 4: strategic / repo-wide, not first for this path

11. Rust/PyO3 for CPU-heavy GOP scalar paths elsewhere in the repo.
12. `k2/icefall` only for broader graph-based training directions, not as the
    first fix for this reproduction loop.

## Recommendation

If the goal is "save real hours and real money before touching anything else",
the first implementation pass should be:

1. add a fast exploratory mode,
2. remove single-GPU `DataParallel`,
3. stop paying `float64` storage/load cost,
4. stop concatenating SSL features every batch,
5. then profile again before reaching for compile or Rust.
