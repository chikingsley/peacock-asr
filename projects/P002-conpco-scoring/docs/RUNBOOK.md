# P002 Runbook (ConPCO Scoring)

Canonical W&B project for final P002 runs:
- `peacockery/peacock-asr-p002-conpco-scoring`

Canonical sweep specs:
- [`projects/P002-conpco-scoring/experiments/sweeps/final/README.md`](/home/simon/github/peacock-asr/projects/P002-conpco-scoring/experiments/sweeps/final/README.md)
- [`projects/P002-conpco-scoring/experiments/sweeps/final/reproduce_conpco_v4_rng_fix.yaml`](/home/simon/github/peacock-asr/projects/P002-conpco-scoring/experiments/sweeps/final/reproduce_conpco_v4_rng_fix.yaml)
- [`projects/P002-conpco-scoring/experiments/sweeps/final/reproduce_conpco_v3.yaml`](/home/simon/github/peacock-asr/projects/P002-conpco-scoring/experiments/sweeps/final/reproduce_conpco_v3.yaml)
- [`projects/P002-conpco-scoring/experiments/sweeps/final/track09_p1_ablation.yaml`](/home/simon/github/peacock-asr/projects/P002-conpco-scoring/experiments/sweeps/final/track09_p1_ablation.yaml)
- [`projects/P002-conpco-scoring/experiments/benchmarks/README.md`](/home/simon/github/peacock-asr/projects/P002-conpco-scoring/experiments/benchmarks/README.md)

Create a sweep:

```bash
uv run --project projects/P002-conpco-scoring wandb sweep projects/P002-conpco-scoring/experiments/sweeps/final/reproduce_conpco_v4_rng_fix.yaml
```

Run agent:

```bash
nohup uv run --project projects/P002-conpco-scoring wandb agent peacockery/peacock-asr-p002-conpco-scoring/<SWEEP_ID> > projects/P002-conpco-scoring/experiments/agents/sweep_<SWEEP_ID>.log 2>&1 &
```

Micro-benchmark the reproduction path before changing defaults:

```bash
uv run --project projects/P002-conpco-scoring python projects/P002-conpco-scoring/code/benchmark_reproduce_conpco.py --split train --steps 3
uv run --project projects/P002-conpco-scoring python projects/P002-conpco-scoring/code/benchmark_reproduce_conpco.py --split train --steps 3 --no-data-parallel
uv run --project projects/P002-conpco-scoring python projects/P002-conpco-scoring/code/benchmark_reproduce_conpco.py --split train --steps 3 --no-data-parallel --preconcat-ssl
```

Notes:
- Canonical code now lives at `projects/P002-conpco-scoring/code/`.
- Canonical third-party code now lives at `projects/P002-conpco-scoring/third_party/ConPCO/`.
- Root-level `runs/*.py` compatibility wrappers have been removed.
- Naming is strict final scheme via `PEACOCK_WANDB_*` vars embedded in sweep YAMLs.
- Current wav2vec2-base sweep process is unaffected by these file updates.
- `P002` should be treated as the richer-feature continuation of `P001`; see
  `docs/NEXT_PHASE_PLAN.md` before adding new sweeps.
