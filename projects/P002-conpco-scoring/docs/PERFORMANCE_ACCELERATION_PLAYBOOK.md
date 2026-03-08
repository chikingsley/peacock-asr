# Track 09 Performance Acceleration Playbook (ConPCO/HierCB)

Last updated: 2026-03-04  
Scope: `projects/P002-conpco-scoring/code/reproduce_conpco.py` +
`projects/P002-conpco-scoring/third_party/ConPCO/src/*` training loop
performance

## 1. Executive Summary

Your current local v4 ConPCO run is not primarily "GPU compute limited". It is mostly "CPU/Python overhead limited".

Observed in this environment during active training:

- GPU memory allocated (~8.6 to 10.6 GB) while GPU SM utilization stayed mostly low (0% to 18%).
- Training process consumed very high CPU (~1100%+ aggregate), indicating host-side work dominates.
- Early run pace was about 73 seconds per epoch (`~586s / 8 epochs`).

Practical implication:

- The highest ROI is still PyTorch-first optimization in your current codebase.
- A framework rewrite (JAX/Rust/etc.) is not the first rational move for speed.

## 2. Confidence Rubric

This rubric is tailored to your request: "how drop-in is this for our exact setup?"

- **L5: Drop-in and low-risk**
  - Minimal code change, same training flow, high confidence of speedup for this project.
- **L4: Small adaptation**
  - Small targeted edits; likely speedup here, with mild risk to comparability.
- **L3: Medium refactor**
  - Non-trivial internal refactor; likely speedup but higher bug/regression risk.
- **L2: Major refactor**
  - Large code rework or architecture-level migration.
- **L1: Strategic rewrite bet**
  - New framework/system. Potential upside, but large execution risk/time.

## 3. Why This Pipeline Is Slow Right Now

### 3.1 Python-heavy hotspots in forward path

In the HierCB model:

- Word-position mask builds via Python nested loops per batch:  
  `projects/P002-conpco-scoring/third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py:412`
- One-hot expansion for phone and word IDs every batch:  
  `projects/P002-conpco-scoring/third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py:532`
  and `:553`

These patterns can bottleneck host-side preprocessing and kernel launch cadence.

### 3.2 Data loading and host-device transfer path is conservative

Current local training loader config in reproduce script:

- `DataLoader(..., num_workers=0 by default, pin_memory=False by default)`  
  `projects/P002-conpco-scoring/code/reproduce_conpco.py:265`

This often under-feeds the GPU on single-node training.

### 3.3 Single-GPU run still wrapped in `DataParallel`

- `model = nn.DataParallel(model)` on one GPU:  
  `projects/P002-conpco-scoring/code/reproduce_conpco.py:279`

`DataParallel` is designed for multi-GPU splitting/replication and can add overhead for single-GPU execution.

### 3.4 Evaluation path also has Python loops

- Token-level and word-level PCC are computed via Python loops and NumPy
  conversion each eval:  
  `projects/P002-conpco-scoring/code/reproduce_conpco.py:148`, `:170`, `:199`

This contributes to epoch-end latency.

## 4. Prioritized Optimization Backlog (For This Exact Project)

## 4.1 Add explicit profiling first (L5)

What:

- Add `torch.profiler` spans around train step, dataloader fetch, forward, backward, optimizer step.

Why:

- Validates bottlenecks with trace-level evidence before changing behavior.

Expected gain:

- No direct speedup; essential for safe optimization ordering.

Risk:

- Minimal; temporary instrumentation only.

## 4.2 DataLoader tuning (`num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`) (L5)

What:

- Tune loader for this fixed-size dataset and batch size.
- Keep semantics unchanged.

Why:

- Your utilization pattern suggests input/host pipeline starvation.

Expected gain:

- **1.15x to 1.6x** typical for host-bound loops, depending on CPU and storage.

Risk:

- Low. Main risk is higher RAM usage and occasional worker startup quirks.

Implementation confidence:

- **L5** for implementation.
- **L4** for guaranteed gain magnitude on your exact machine.

## 4.3 Remove `DataParallel` on single-GPU runs (L5)

What:

- If `torch.cuda.device_count() == 1`, use plain `model.to(device)` instead of `nn.DataParallel`.

Why:

- Avoids replication/scatter-gather wrapper overhead not needed on one GPU.

Expected gain:

- **1.03x to 1.15x**.

Risk:

- Very low if checkpoint loading/saving paths are handled consistently (`module.` key prefixes).

Implementation confidence:

- **L5**.

## 4.4 Enable AMP (`torch.autocast` + `GradScaler` where needed) (L4)

What:

- Use mixed precision in forward/loss.
- Prefer BF16 on modern NVIDIA cards when stable.

Why:

- Can reduce memory bandwidth pressure and speed up math kernels.

Expected gain:

- **1.15x to 1.8x** depending on kernel mix and numerical tolerance.

Risk:

- Moderate numerical behavior drift; must monitor stability and final PCC.

Implementation confidence:

- **L4** (routine PyTorch pattern).

## 4.5 `torch.compile(mode="reduce-overhead")` trial (L4)

What:

- Wrap model (or hot submodules) with `torch.compile`.

Why:

- Docs explicitly call out `reduce-overhead` mode for reducing Python overhead on CUDA, especially small-batch scenarios.

Expected gain:

- **1.05x to 1.4x** when graph breaks are limited.

Risk:

- Compile warmup overhead, graph-break sensitivity, occasional edge-case issues.

Implementation confidence:

- **L4** for trying safely, **L3** for getting stable wins across all runs.

## 4.6 Vectorize `make_word_pos_mask` and similar Python loops (L3)

What:

- Rewrite nested Python loops into tensor ops.

Why:

- This hotspot runs every forward pass and can materially reduce host overhead.

Expected gain:

- **1.1x to 1.4x** if this path is currently dominant.

Risk:

- Medium; correctness and padding behavior must be verified carefully.

Implementation confidence:

- **L3**.

## 4.7 Replace one-hot + linear with embedding lookup where equivalent (L3)

What:

- Replace full one-hot creation with `nn.Embedding` style lookup.

Why:

- Avoids dense one-hot materialization and can reduce memory traffic.

Expected gain:

- **1.05x to 1.25x** and reduced memory overhead.

Risk:

- Medium; must preserve exact indexing/padding semantics.

Implementation confidence:

- **L3**.

## 4.8 Reduce eval overhead or eval frequency during sweeps (L4)

What:

- Keep final reproducibility runs unchanged, but for exploratory sweeps evaluate every N epochs or compute lighter interim metrics.

Why:

- Current eval includes Python-heavy PCC loops and full test pass each epoch.

Expected gain:

- Can significantly reduce wall-clock during exploration.

Risk:

- Medium-to-high if used in final reported experiments (metric comparability).

Implementation confidence:

- **L4** for exploratory mode only.

## 5. Suggested Safe Rollout Order

1. Profiling instrumentation and baseline trace capture.
2. DataLoader tuning only.
3. Single-GPU `DataParallel` removal.
4. AMP trial with strict metric checks.
5. `torch.compile` trial.
6. Python-loop vectorization and one-hot refactors.

Rationale:

- Steps 1 to 4 are highest confidence with low risk to reproducibility.
- Steps 5 to 6 can add variance and should come after baseline stabilization.

## 6. JAX: Should You Move?

Short answer for this project right now: **not first**.

### 6.1 What JAX is excellent at

- Whole-program JIT and aggressive fusion.
- Strong multi-device and sharding model.
- Very strong scaling story in large, static-shape training systems.

### 6.2 Why JAX is not an immediate speed fix here

- Your current bottlenecks are Python loops and host overhead in an existing PyTorch model.
- JAX speedups generally require writing code in JAX-native style (`jit`-friendly pure functions, reduced Python control flow in hot paths, explicit sharding patterns where relevant).
- Porting your custom model/loss/data path is a real rewrite, not a drop-in swap.

### 6.3 Confidence for JAX migration in this repo

- Performance upside long-term: **possible**.
- Near-term productivity/risk for this track: **L1 to L2**.
- Recommendation: do PyTorch-first optimization now; revisit JAX only if you choose a strategic multi-quarter platform move.

## 7. Other Language/Runtime Options (Rust, Go, Mojo/MAX)

### 7.1 Rust frameworks (Burn, Candle)

- Promising ecosystems; good systems-level control.
- Not drop-in for your current training code and experiment stack.
- Migration would still be a substantial rewrite.

Confidence for immediate Track 09 speed gains: **L1 to L2**.

### 7.2 Go ML stacks (e.g., Gorgonia)

- Niche ecosystem for deep learning at this scale.
- Very low compatibility with your existing PyTorch model/training artifacts.

Confidence for practical near-term replacement: **L1**.

### 7.3 Mojo/MAX

- Strong positioning and docs around inference/runtime acceleration and serving workflows.
- Not a mature drop-in replacement for your current ConPCO PyTorch training loop.

Confidence as immediate training-system replacement: **L1 to L2**.

## 8. CUDA/Triton/k2 Opportunities (Keep On The Table)

This section is the "next-level movement" track that can be layered after the
safe PyTorch-first wins.

### 8.1 CUDA path: Attention kernel modernization (L4)

What:

- Replace custom attention math with PyTorch SDPA path where shape/mask
  constraints permit.

Where in this repo:

- `projects/P002-conpco-scoring/third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py:196`
- `projects/P002-conpco-scoring/third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py:546`

Why:

- SDPA can route to optimized fused kernels on CUDA and cut launch overhead.

Risk:

- Must preserve masking semantics and numerical behavior.

Confidence:

- **L4** for implementation trial.

### 8.2 CUDA path: custom op for mask-building hotspot (L3)

What:

- Move dynamic word-position mask creation to a native op.

Where in this repo:

- `projects/P002-conpco-scoring/third_party/ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py:412`

Why:

- Current nested Python loops execute every forward pass.

Risk:

- Correctness around padding and shape contracts.

Confidence:

- **L3**.

### 8.3 Triton path: selective kernel fusion, not full rewrite (L2 to L3)

Good Triton candidates:

- Repeated dense tensor math where control flow is simple and static.
- Potentially parts of loss-side distance computation if profiling shows
  arithmetic dominance.

Poor Triton candidates in this pipeline:

- Branch-heavy logic with `unique`/`isin`/dynamic filtering behavior, e.g.
  ConPCO token filtering paths.

Where to inspect first:

- `projects/P002-conpco-scoring/third_party/ConPCO/src/models/conPCO_norm.py:48`
- `projects/P002-conpco-scoring/code/p002_conpco/conpco_losses.py`

Confidence:

- **L3** for targeted use, **L2** as a broad pipeline strategy.

### 8.4 k2/icefall: strategic, but not a direct Track 09 speed lever (L2)

What it is good for:

- Graph-based ASR training/decoding and transducer/CTC pipelines, especially in
  dedicated ASR model tracks.

Why not first for current bottleneck:

- Current slowdown is in ConPCO/HierCB training loop overhead, not decoding
  graph complexity.

Keep on table for:

- Track 07/08 style ASR training and streaming research directions.

Confidence:

- **L2** for immediate Track 09 acceleration.

## 9. Rust/PyO3 Target Map For This Repo

Rust/PyO3 is most useful here for CPU-side data-plane and tooling loops. It is
not the first choice for core GPU training kernels.

### 9.1 High-value Rust/PyO3 candidates

#### A) Audio decode + resample preprocessor module (L4)

Where:

- `projects/P003-compact-backbones/code/training/preprocess_features.py:47`
- `projects/P003-compact-backbones/code/training/preprocess_features.py:67`
- `projects/P003-compact-backbones/code/training/preprocess_features.py:81`

Idea:

- Replace Python decode/resample hot path with Rust implementation
  (e.g., Symphonia + Rubato/libsamplerate bindings) exposed via PyO3.

Why:

- This path runs at very high volume in dataset preprocessing and can consume
  substantial CPU wall time.

Risk:

- Audio parity validation required (bitwise mismatch is acceptable if metric
  parity holds, but must be measured).

#### B) SpeechOcean parsing and normalization helper (L4)

Where:

- `projects/P001-gop-baselines/code/p001_gop/dataset.py`
- `projects/P001-gop-baselines/code/p001_gop/dataset.py`
- `projects/P001-gop-baselines/code/p001_gop/dataset.py`

Idea:

- Rust function to transform nested word/phone structures into flat phone,
  score, and word-accuracy arrays with stress stripping and filtering.

Why:

- Tight Python loops over 2.5k examples per split are an easy Rust acceleration
  target.

Risk:

- Schema drift sensitivity when upstream dataset fields change.

#### C) Metric kernels for evaluation loops (L4)

Where:

- `projects/P002-conpco-scoring/code/reproduce_conpco.py:148`
- `projects/P002-conpco-scoring/code/reproduce_conpco.py:170`
- `projects/P002-conpco-scoring/code/reproduce_conpco.py:199`
- `projects/P002-conpco-scoring/code/p002_conpco/evaluate.py`

Idea:

- Implement PCC and per-word aggregation in Rust using ndarray/polars-like
  vector operations and expose to Python.

Why:

- Repeated Python `.item()` and nested loops are expensive at eval time.

Risk:

- Must preserve exact masking and edge-case behavior.

#### D) GOP scalar inner-loop helper (L3)

Where:

- `projects/P001-gop-baselines/code/p001_gop/gop.py`

Idea:

- Rework per-phone denom loop orchestration in Rust if profiling shows host
  overhead around repeated calls.

Why:

- Loop-level orchestration overhead can add up.

Risk:

- Lower ROI than training-loop fixes because heavy math already sits in Torch.

### 9.2 Lower-value Rust/PyO3 candidates (for now)

- Core model forward/backward replacement.
- ConPCO loss tensor algebra replacement that already maps well to Torch/CUDA.

Confidence:

- **L1 to L2** as near-term choices.

### 9.3 Suggested Rust/PyO3 rollout

1. Build one focused extension crate with `maturin` and benchmark in isolation.
2. Integrate behind feature flags or optional imports.
3. Keep pure-Python fallback path for reproducibility and debugging.
4. Add parity tests (outputs and metrics) before enabling in sweeps.

## 10. Decision Matrix (Permanent-First Ordering)

This matrix is tuned to your preference: "try the bigger, more permanent thing
first when practical."

| Option | Example work in this repo | Confidence | Expected impact | Overlap/dependency |
|---|---|---|---|---|
| PyTorch core-path modernization (permanent) | Remove single-GPU `DataParallel`, vectorize word-mask path, reduce per-step Python one-hot/filtering work | L3 to L5 | High | Makes `torch.compile`/Inductor much more effective |
| Data-path modernization (permanent, low risk) | DataLoader workers/pinning/prefetch + async H2D (`non_blocking`) | L4 to L5 | Medium to high | Prerequisite for good compute/input overlap |
| Runtime modernization (permanent after stabilization) | AMP/BF16/TF32 + `torch.compile(mode=\"reduce-overhead\")` | L3 to L4 | High | Best after Python-heavy graph breaks are reduced |
| Native GPU extensions (targeted) | SDPA migration, optional custom C++/CUDA op, selective Triton kernels | L2 to L4 | Medium to high | Often only needed for residual hotspots after compile |
| Rust/PyO3 CPU acceleration (targeted) | GOP scalar kernels, eval/metrics helpers, optional dataset flattening helpers | L3 to L4 | Medium to high on CPU-bound phases | Mostly orthogonal to GPU kernel optimizations |
| JAX rewrite (strategic) | Rebuild train/eval and loss logic in JAX-native style | L1 to L2 | Uncertain near term, high long term potential | Supersedes many PyTorch-specific optimizations but major rewrite |

## 11. Overlap and Dependency Map

### 11.1 What should happen before what

1. Profile first (`torch.profiler` + CUDA events) so each next move is evidence-based.
2. Remove single-GPU `DataParallel` before compile trials.
3. DataLoader pinning/worker tuning should precede `non_blocking=True` H2D tuning.
4. Vectorized masks/filters/one-hot removal should precede heavy compile tuning.
5. `torch.compile` should run before custom Triton/CUDA work in most cases.

### 11.2 What can make other work unnecessary

- Better `torch.compile` + SDPA coverage can eliminate the need for many custom
  Triton kernels.
- Rust/PyO3 for GOP scalar loops can eliminate the need to over-invest in Python
  multiprocessing tuning for that path.
- A full JAX migration would replace most PyTorch- and Triton-specific
  investment, but with much higher execution cost/risk.

### 11.3 Where options are mostly orthogonal

- Rust/PyO3 data-plane acceleration and CUDA/Triton kernel acceleration are
  mostly complementary: one helps host/CPU loops, the other helps GPU kernels.
- k2/icefall is strategic for graph-based ASR pipelines, not a direct fix for
  current Track 09 ConPCO training-loop overhead.

## 12. Ordered Execution Plans

### 12.1 Permanent-First Plan (your requested ordering)

1. Consolidate onto a cleaner PyTorch core path (remove single-GPU
   `DataParallel`, reduce Python loops in forward/loss/eval).
2. Apply structural GPU-facing improvements in-model
   (vectorized mask path, one-hot replacement where possible, tensorized loss
   path parity checks).
3. Lock in runtime layer (`AMP`/BF16/TF32 and `torch.compile` after step 2).
4. Add targeted native acceleration where still justified by profiler:
   SDPA path first, then custom C++/CUDA or selective Triton for remaining
   hotspots.
5. Add Rust/PyO3 accelerators for persistent CPU-heavy phases
   (especially GOP scalar and possibly eval helpers).
6. Re-evaluate JAX only if you choose a strategic multi-quarter platform move.

### 12.2 Fastest-Wins Plan (for comparison)

1. DataLoader tuning + async H2D copy path.
2. Remove single-GPU `DataParallel`.
3. `AMP`/BF16 and compile trial.
4. Then do deeper model-path refactors and native extensions.

Use 12.1 when you want durable architecture movement first; use 12.2 when
immediate sweep throughput is the priority.

## 13. Bottom-Line Recommendation

For your exact setup, the highest-confidence conclusion remains:

- **Stay in PyTorch now** and modernize the existing path first.
- Treat Rust/PyO3 as a targeted CPU-acceleration layer (especially GOP scalar),
  not a replacement for model training kernels.
- Treat Triton/custom CUDA as a measured follow-up after compile/SDPA/vectorization.
- Treat JAX as a strategic rewrite option, not the first speed lever for v3/v4
  reproduction loops.

If this sequence is executed in order, you maximize the chance of large speed
gains while keeping comparability and reproducibility risk controlled.

## 14. External References

Primary docs referenced for implementation decisions:

1. PyTorch `DataLoader` API and performance-related options  
   https://docs.pytorch.org/docs/stable/data
2. PyTorch performance tuning guide  
   https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
3. PyTorch AMP examples (`autocast`, `GradScaler`)  
   https://docs.pytorch.org/docs/stable/notes/amp_examples.html
4. `torch.compile` API and modes (`reduce-overhead`)  
   https://docs.pytorch.org/docs/stable/generated/torch.compile.html
5. Profiling `torch.compile` performance  
   https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html
6. PyTorch profiler tutorial with TensorBoard trace handler  
   https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
7. `DataParallel` docs (recommendation to prefer DDP)  
   https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
8. `DistributedDataParallel` docs (performance note vs DataParallel)  
   https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
9. CUDA semantics and pinned/non-blocking transfer context  
   https://docs.pytorch.org/docs/stable/notes/cuda.html
10. Float32 matmul precision controls (`set_float32_matmul_precision`)  
    https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
11. PyTorch reproducibility and deterministic settings  
    https://docs.pytorch.org/docs/stable/notes/randomness.html
12. PyTorch SDPA API  
    https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
13. PyTorch C++ extension API (`torch.utils.cpp_extension`)  
    https://docs.pytorch.org/docs/stable/cpp_extension.html
14. PyTorch custom op landing page  
    https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html
15. Triton tutorials  
    https://triton-lang.org/main/getting-started/tutorials/index.html
16. k2 documentation  
    https://k2-fsa.github.io/k2/
17. icefall repository  
    https://github.com/k2-fsa/icefall
18. JAX JIT compilation guide  
    https://docs.jax.dev/en/latest/jit-compilation.html
19. JAX benchmarking guide (compile + async dispatch caveats)  
    https://docs.jax.dev/en/latest/benchmarking.html
20. JAX asynchronous dispatch  
    https://docs.jax.dev/en/latest/async_dispatch.html
21. JAX sharp bits / gotchas  
    https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
22. JAX distributed arrays and automatic parallelization  
    https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
23. JAX training cookbook  
    https://docs.jax.dev/en/latest/the-training-cookbook.html
24. Burn framework repository  
    https://github.com/tracel-ai/burn
25. Burn custom training loop docs  
    https://burn.dev/burn-book/custom-training-loop.html
26. Candle framework repository  
    https://github.com/huggingface/candle
27. Gorgonia repository  
    https://github.com/gorgonia/gorgonia
28. Modular MAX docs (platform scope and serving focus)  
    https://docs.modular.com/max/
29. PyO3 user guide  
    https://pyo3.rs/
30. PyO3 build/distribution guide  
    https://pyo3.rs/latest/building-and-distribution.html
31. maturin (build/publish PyO3 extensions)  
    https://www.maturin.rs/
