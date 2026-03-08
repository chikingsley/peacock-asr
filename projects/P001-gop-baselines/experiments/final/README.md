# P001 Final Local Artifacts

This directory is the canonical local landing zone for the `P001` paper-close
campaign.

Recommended subdirectories:

- `batches/` for `peacock-asr batch` outputs such as `summary.tsv` and
  `aggregates.tsv`
- `alpha_sweeps/` for dense alpha sweep tables and metadata
- `checkpoints/` for local `A3` GOPT checkpoint folders
- `agents/` for background W&B agent logs when using `nohup`
- `manifests/` for host snapshots and run-context provenance JSON
- `results/` for campaign-wide per-run and aggregate summaries derived from the
  final sweep logs

Launch helpers:

- `launch_p001_paper_close.py` is the preferred `uv` Python launcher
- `capture_machine_manifest.py` writes a project-local host manifest and
  run-context manifest for the live campaign, including optional RunPod
  control-plane metadata when `--pod-id` is provided

RunPod companion capture example:

- `uv run --project projects/P001-gop-baselines python projects/P001-gop-baselines/experiments/final/capture_machine_manifest.py --campaign paper-close --phase phase1 --pod-id <POD_ID> --runpod-hourly-cost 0.39 --runpod-note "RunPod L4 training pod"`

Cross-project remote training example:

- `uv run --project projects/P001-gop-baselines python projects/P001-gop-baselines/experiments/final/capture_machine_manifest.py --project-id P003 --project-slug compact-backbones --campaign runpod-train --phase training --dataset-name LibriSpeech --wandb-group-prefix p003-runpod-train --output-root projects/P003-compact-backbones/experiments/final --pod-id <POD_ID> --runpod-template-id <TEMPLATE_ID> --runpod-hourly-cost 1.39`

Not stored here by default:

- feature caches in `.cache/features/`
- transient backend/model caches in `.cache/models/`

Reason:

- local evidence that should survive paper drafting belongs under the project
  workspace
- disposable acceleration state should stay out of the project tree
