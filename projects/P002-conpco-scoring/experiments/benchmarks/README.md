# P002 Benchmark Notes

Environment:

- Local RTX 5070
- `projects/P002-conpco-scoring/code/benchmark_reproduce_conpco.py`
- Train split
- `batch_size=25`
- ConPCO loss enabled

Purpose:

- Measure short-step throughput before changing the main reproduction script.
- Validate whether obvious low-risk changes actually help on this machine.

## Current Probe Results

All three rows below use `--steps 10`.

| Variant | Mean step (s) | First batch (s) | Dataset load (s) | Interpretation |
|---|---:|---:|---:|---|
| baseline (`DataParallel`, workers=0) | 0.0775 | 0.0041 | 1.1554 | Current reproduction baseline |
| `--no-data-parallel` | **0.0750** | 0.0047 | 1.1428 | Small win on single GPU |
| `--no-data-parallel --preconcat-ssl` | 0.1000 | 0.0072 | 1.2198 | Worse on this setup |
| `--no-data-parallel --num-workers 4 --pin-memory` | 0.1050 | 0.0367 | 1.1569 | Worse on this setup |

JSON artifacts:

- `baseline_dp_on_10.json`
- `no_data_parallel_10.json`
- `no_dp_preconcat_ssl_10.json`
- `no_dp_workers4_pin_10.json`

## Current Read

- `DataParallel` on one GPU is not helping.
- The gain from removing it is real but modest on this local benchmark.
- Worker-heavy loader tuning is counterproductive here because the dataset is
  already loaded into memory and worker/process overhead dominates.
- Pre-concatenating SSL features inside the dataset also did not help on this
  machine in the current form.

## What This Does Not Prove Yet

- These are short micro-benchmarks, not full-epoch or full-run timings.
- They do not include epoch-end evaluation overhead.
- They do not measure cloud GPUs yet.
- They do not test AMP or `torch.compile`.

## Recommended Next Benchmark Order

1. Make single-GPU `DataParallel` optional in the canonical reproduction path.
2. Re-run a longer benchmark on the same machine.
3. Compare local RTX 5070 vs cloud RTX 4070 on the same micro-benchmark script.
4. Only after that, test AMP and `torch.compile`.
