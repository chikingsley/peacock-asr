# P002 Final Sweeps (W&B-first)

W&B project for all final P002 runs:
- `peacockery/peacock-asr-p002-conpco-scoring`

Create a sweep:
```bash
uv run --project projects/P002-conpco-scoring wandb sweep projects/P002-conpco-scoring/experiments/sweeps/final/<sweep_file>.yaml
```

Run an agent:
```bash
nohup uv run --project projects/P002-conpco-scoring wandb agent peacockery/peacock-asr-p002-conpco-scoring/<SWEEP_ID> > projects/P002-conpco-scoring/experiments/agents/sweep_<SWEEP_ID>.log 2>&1 &
```

Sweep files:
- `reproduce_conpco_v3.yaml`
- `reproduce_conpco_v4_rng_fix.yaml`
- `track09_p1_ablation.yaml`
