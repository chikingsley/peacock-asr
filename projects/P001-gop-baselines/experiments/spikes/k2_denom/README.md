## k2 Denominator Spike

This spike tests whether `k2` can reproduce the scalar GOP denominator used by
`p001_gop.gop._ctc_forward_denom()` closely enough to justify a deeper
`P001-C` rewrite.

The workflow is:

1. Create an isolated env for the spike.
2. Install `torch==2.8.0` and a matching official `k2` CUDA wheel.
3. Run a parity script that compares:
   - baseline `_ctc_forward()`
   - baseline `_ctc_forward_denom()`
   - `k2` standard CTC total score
   - `k2` custom denominator graph total score

The script is intentionally small and focuses on one utterance / one phone
position first. If parity is good, the next step is batching.

## Bootstrap

```bash
cd projects/P001-gop-baselines/experiments/spikes/k2_denom
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python \
  'torch==2.8.0' \
  'numpy>=2.0' \
  'k2 @ https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20250807+cuda12.8.torch2.8.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
```

## Run

```bash
.venv/bin/python run_k2_denom_spike.py --device cpu
.venv/bin/python run_k2_denom_spike.py --device cuda
.venv/bin/python benchmark_exact_k2_denom.py --device cpu
.venv/bin/python benchmark_exact_k2_denom.py --device cuda
.venv/bin/python benchmark_topology_k2_denom.py --device cpu
.venv/bin/python benchmark_topology_k2_denom.py --device cuda
```

## Current Result

The environment and runtime path work on both CPU and CUDA.

Current parity status:
- exact unrolled standard CTC graph matches baseline (`~2.1e-9` absolute diff)
- exact unrolled denominator graph matches baseline (`~3.0e-8` absolute diff)
- exact unrolled occupancy recovered from `k2` forward scores also matches baseline (`~1.8e-8` absolute diff)
- compact dense denominator graph is still not semantically correct (`~14.37` absolute diff)

Recorded outputs:
- [cpu_result.json](./cpu_result.json)
- [cuda_result.json](./cuda_result.json)
- [cpu_benchmark.json](./cpu_benchmark.json)
- [cpu_benchmark_no_occ.json](./cpu_benchmark_no_occ.json)
- [cpu_benchmark_u32_no_occ.json](./cpu_benchmark_u32_no_occ.json)
- [cuda_benchmark.json](./cuda_benchmark.json)
- [cuda_benchmark_u32_no_occ.json](./cuda_benchmark_u32_no_occ.json)
- [topology_cpu_benchmark.json](./topology_cpu_benchmark.json)
- [topology_cuda_benchmark.json](./topology_cuda_benchmark.json)
- [topology_cpu_benchmark_u32.json](./topology_cpu_benchmark_u32.json)
- [topology_cuda_benchmark_u32.json](./topology_cuda_benchmark_u32.json)
- [topology_cpu_benchmark_occ.json](./topology_cpu_benchmark_occ.json)
- [topology_cuda_benchmark_occ.json](./topology_cuda_benchmark_occ.json)
- [topology_cpu_benchmark_occ_u32.json](./topology_cpu_benchmark_occ_u32.json)
- [topology_cuda_benchmark_occ_u32.json](./topology_cuda_benchmark_occ_u32.json)

Interpretation:
- `k2` is a viable runtime/install target for `P001-C`
- the denominator recurrence can be represented exactly as a `k2` graph
- the existing occupancy term can also be recovered from the exact graph
- the remaining work is optimization/compression, not basic mathematical feasibility
- the next open question is no longer whether topology-only scoring works; it does
- the remaining optimization question is whether we can recover batched occupancy cleanly and/or compress the topology further

## Benchmark

`benchmark_exact_k2_denom.py` compares:
- baseline `_ctc_forward_denom()` loop over all `(utterance, phone-position)` cases
- exact unrolled `k2` graph build cost
- exact unrolled `k2` parse cost
- exact unrolled `k2` batched total-score cost
- exact unrolled occupancy recovery cost

This is the next decision point for `P001-C`: if exact batched graphs are
already competitive, compression work becomes optional; if not, compact graph
construction is the next engineering target.

Current CPU takeaway:
- score-only `k2` total-score evaluation is very fast (`~50x` faster than the
  baseline denominator loop on the tested synthetic cases)
- exact graph construction dominates end-to-end time
- exact unrolled `k2` is therefore a viable correctness engine, but not yet a
  clear full-pipeline speed win without graph compression or graph caching

Current CUDA takeaway on the local RTX 5070:
- score-only `k2` total-score evaluation is also very fast (`~13x` faster on
  the 64-case run, `~82x` faster on the 256-case run)
- exact graph construction still dominates total time on GPU
- occupancy recovery on GPU is relatively expensive in the current Python-heavy
  implementation
- exact unrolled `k2` remains a strong correctness path, but the next speed win
  has to come from graph compression/caching rather than from the scoring
  kernel itself

## Topology-Only Exact Benchmark

`benchmark_topology_k2_denom.py` replaces utterance-specific weighted graphs
with a topology-only exact denominator graph plus `DenseFsaVec` acoustics.

Current takeaway:
- the topology-only exact path preserves denominator parity (`~9.5e-7` max abs diff)
- it is a real end-to-end win over both the baseline CPU loop and the weighted
  exact `k2` path
- the main win comes from removing utterance-specific graph construction, not
  from changing the denominator math

Current CPU results (`64` cases, score-only):
- baseline loop: `0.779s`
- weighted exact `k2`: `1.122s`
- topology-only exact `k2`: `0.640s`
- topology speedup vs baseline: `~1.22x`
- topology speedup vs weighted exact `k2`: `~1.75x`

Current CUDA results on the local RTX 5070 (`64` cases, score-only):
- baseline loop: `0.786s`
- weighted exact `k2`: `1.312s`
- topology-only exact `k2`: `0.592s`
- topology speedup vs baseline: `~1.33x`
- topology speedup vs weighted exact `k2`: `~2.21x`

Current CPU results (`256` cases, score-only):
- baseline loop: `3.081s`
- weighted exact `k2`: `4.435s`
- topology-only exact `k2`: `2.929s`
- topology speedup vs baseline: `~1.05x`
- topology speedup vs weighted exact `k2`: `~1.51x`

Current CUDA results on the local RTX 5070 (`256` cases, score-only):
- baseline loop: `3.082s`
- weighted exact `k2`: `4.537s`
- topology-only exact `k2`: `2.296s`
- topology speedup vs baseline: `~1.34x`
- topology speedup vs weighted exact `k2`: `~1.98x`

Current caveat:
- batched occupancy is now implemented for the topology-only path
- topology occupancy matches the existing exact `k2` occupancy behavior to the
  same tolerance on the tested batches
- occupancy is no longer the dominant cost after the latest vectorization pass
- the remaining bottleneck is topology build plus `intersect_dense`, not the
  occupancy calculation itself
- at larger synthetic batches, both exact and topology `k2` occupancy drift
  from the pure Python baseline in the same way, so the current benchmark gate
  treats topology-vs-exact agreement as the important correctness condition

## Topology-Only Occupancy Benchmark

With occupancy enabled, the topology path still preserves the denominator
parity, and its occupancy matches the existing exact `k2` path:
- `64` cases, CPU:
  - topology total: `1.047s`
  - exact weighted total: `1.350s`
  - baseline loop: `0.829s`
  - topology vs exact occupancy max abs diff: effectively `0`
- `64` cases, CUDA:
  - topology total: `0.990s`
  - exact weighted total: `2.224s`
  - baseline loop: `0.794s`
  - topology vs exact occupancy max abs diff: effectively `0`
- `256` cases, CPU:
  - topology total: `4.479s`
  - exact weighted total: `5.243s`
  - baseline loop: `3.204s`
  - topology vs exact occupancy max abs diff: effectively `0`
- `256` cases, CUDA:
  - topology total: `3.702s`
  - exact weighted total: `7.914s`
  - baseline loop: `3.057s`
  - topology vs exact occupancy max abs diff: effectively `0`

Interpretation:
- topology-only exact scoring is ready to use for denominator scores
- topology-only occupancy is correct relative to the existing `k2` path
- the occupancy path is now cheap enough that it is no longer the blocking
  issue for `P001-C`
- the next optimization target is topology construction and/or caching of the
  topology graph itself
