# P011 Experiment Log

## Purpose

P011 exists to stop mixing two different claims:

1. MuFFIN replication on phone-level SSL features.
2. Paper-faithful HConv, where the interface runs on frame-level SSL hidden states before
   phone pooling.

P010 had inherited code and run habits that made both provenance and experiment identity
sloppy. P011 isolates the faithful HConv path and records each run with an explicit manifest
and unique checkpoint root.

## What Changed

- The HConv path now loads frame-level shards from `ssl_frame_store_v1` instead of applying
  HConv directly to phone-pooled `*_all_layers.npy`.
- Phone pooling now happens **after** HConv inside the model wrapper, matching the method
  order in Shih & Harwath 2024.
- `ssl_models` is configurable, so we can run one upstream at a time.
- `ssl_output_dim` is derived by default from the selected upstream subset, with an explicit
  override only when we intentionally want a shape-preserving ablation.
- `grad_accum_steps` is exposed so low-VRAM runs can preserve effective batch size instead of
  silently changing optimization.
- `train`, `sweep`, and `pretrain` now create unique checkpoint roots and write
  `run_manifest.json` / `sweep_manifest.json`.
- `WANDB_DIR` defaults to the local P011 project so runs stay isolated from P010.

## Data Source

- Features root: `/home/simon/data/p010-features`
- Frame store root: `/home/simon/data/p010-features/ssl_frame_store_v1`
- Verified locally on 2026-03-27 that pooling the frame store by phone durations reproduces
  the local `*_all_layers.npy` tensors up to tiny numerical error.

## Feature Lineage Diagnostic

Date: 2026-03-27

Compared the first 200 training utterances across three sources for each SSL model:

1. pooled frame-store last layer
2. cached `*_all_layers.npy[:, :, -1, :]`
3. cached `*_feat_v2.npy`

Result:

- pooled frame-store last layer matches cached `all_layers` last layer almost exactly
  (`mean_abs_diff ~= 1e-5` for all three SSL models)
- pooled frame-store last layer does **not** match `feat_v2`
- `feat_v2` phone counts only matched the frame-store/all-layers path on `131/200`
  utterances in this sample

Per-model summary:

- `hubert`: `mean_abs(frame_last, feat_v2)=0.2191`, cosine `0.1169`
- `w2v_300m`: `mean_abs(frame_last, feat_v2)=0.4099`, cosine `-0.0202`
- `wavlm`: `mean_abs(frame_last, feat_v2)=0.1753`, cosine `0.0976`

Interpretation:

- The frame-store path is internally self-consistent.
- The current HConv experiments are **not** operating on the same SSL representation lineage
  as the original `feat_v2` MuFFIN baseline.
- The next control is therefore a frame-store run with `ssl_interface=last`, which keeps the
  frame-store lineage but removes HConv from the equation.

Control command:

```bash
cd /home/simon/github/peacock-asr/projects/P011-hconv-faithful
uv run p011 train --ssl-interface last --ssl-models hubert --batch-size 5 --grad-accum-steps 5
```

## Smoke Validation

Date: 2026-03-27

Command:

```bash
cd /home/simon/github/peacock-asr/projects/P011-hconv-faithful
WANDB_MODE=offline uv run p011 train --ssl-models hubert --batch-size 5 --grad-accum-steps 5 --n-epochs 1
```

Result:

- Completed successfully on real data.
- Offline W&B run: `wandb/wandb/offline-run-20260327_160341-4uxe2nix`
- Auto-generated run name: `hconv-hubert-seed22-20260327-160340`
- Best/test phone PCC after 1 epoch: `0.28356`

This smoke run only establishes that the faithful frame-first path executes end to end on
the real dataset without the old phone-level shortcut.

## Sweep Launch Record

Primary sweep target:

```bash
cd /home/simon/github/peacock-asr/projects/P011-hconv-faithful
uv run p011 sweep --ssl-models hubert --batch-size 5 --grad-accum-steps 5
```

Runtime policy:

- Run inside `tmux`
- Keep W&B online unless explicitly disabled
- Let the CLI create the sweep root automatically under:
  `checkpoints/sweeps/hconv/hubert/<timestamp>/`

## Provenance

- Root repo snapshot before P011 branch creation: `81a926b`
- P011 commit: pending at the time this log entry was written; commit after validating the
  real-data smoke run and before launching the tmux sweep.
