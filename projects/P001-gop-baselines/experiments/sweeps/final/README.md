# P001 Final Sweeps (W&B-first)

W&B project for all final P001 runs:
- `peacockery/peacock-asr-p001-gop-baselines`

Canonical campaign spec:
- [`projects/P001-gop-baselines/docs/FINAL_CAMPAIGN_SPEC.md`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/docs/FINAL_CAMPAIGN_SPEC.md)

These sweep files now inject canonical run metadata through env vars:
- `PEACOCK_WANDB_TRACK=track05`
- `PEACOCK_WANDB_PROJECT_ID=P001`
- `PEACOCK_WANDB_PHASE=<phase1|phase2>`
- `PEACOCK_WANDB_JOB_ID=<a1..a3|b1..b5>`
- `PEACOCK_WANDB_RUN_PREFIX=p001-paper-close`
- `PEACOCK_WANDB_JOB_TYPE=eval`
- `PEACOCK_CHECKPOINTS_DIR=projects/P001-gop-baselines/experiments/final/checkpoints`
  on the `A3` GOPT sweeps

Create a sweep:
```bash
uv run --project projects/P001-gop-baselines wandb sweep projects/P001-gop-baselines/experiments/sweeps/final/<sweep_file>.yaml
```

Run an agent:
```bash
mkdir -p projects/P001-gop-baselines/experiments/final/agents
nohup uv run --project projects/P001-gop-baselines wandb agent peacockery/peacock-asr-p001-gop-baselines/<SWEEP_ID> > projects/P001-gop-baselines/experiments/final/agents/sweep_<SWEEP_ID>.log 2>&1 &
```

Sweep files:
- `phase1_original_a1_scalar.yaml`
- `phase1_original_a2_feats.yaml`
- `phase1_original_a3_gopt.yaml`
- `phase1_xlsr_a1_scalar.yaml`
- `phase1_xlsr_a2_feats.yaml`
- `phase1_xlsr_a3_gopt.yaml`
- `phase2_original_b1_gopsf.yaml`
- `phase2_original_b2_logit_margin.yaml`
- `phase2_original_b3_logit_combined_a025.yaml`
- `phase2_original_b4_logit_combined_a050.yaml`
- `phase2_original_b5_logit_combined_a075.yaml`
- `phase2_xlsr_b1_gopsf.yaml`
- `phase2_xlsr_b2_logit_margin.yaml`
- `phase2_xlsr_b3_logit_combined_a025.yaml`
- `phase2_xlsr_b4_logit_combined_a050.yaml`
- `phase2_xlsr_b5_logit_combined_a075.yaml`

Dense alpha sweeps are not part of these W&B sweep YAMLs. Run them with
`uv run --project projects/P001-gop-baselines peacock-asr sweep-alpha` using
the `phase2b` commands in the campaign spec.
