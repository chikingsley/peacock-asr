# Citrinet P2-B Train-Clean-100 Readiness

Date:

- 2026-03-07

Decision:

- skip `P2-A`
- proceed with `P2-B`
- accept the `44`-class `wpe` tokenizer path for the first real experiment

Artifacts:

- full train manifest:
  [train.jsonl](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/train_clean_100_full/manifests/train.jsonl)
- full eval manifest:
  [eval.jsonl](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/train_clean_100_full/manifests/eval.jsonl)
- asset summary:
  [asset_summary.tsv](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/train_clean_100_full/asset_summary.tsv)
- full-manifest smoke report:
  [report.json](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/smoke_run_fullmanifest/report.json)
- training entrypoint:
  [train_citrinet_p2b.py](/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/scripts/train_citrinet_p2b.py)

Measured dataset export:

- train rows: `28,538`
- train hours: `98.595`
- eval rows: `2,703`
- eval hours: `5.133`
- on-disk size:
  `6.6G`

What was validated:

- stock `nvidia/stt_en_citrinet_256_ls` loads
- tokenizer swap to the repo phoneme target works through NeMo
- effective decoder vocabulary becomes `44` because `wpe` adds:
  - `[CLS]`
  - `[SEP]`
  - `[MASK]`
- the full `train_clean_100` and `dev_clean` manifests load cleanly
- a 1-step dry run on the full manifests completed and saved a `.nemo` artifact

First real run command:

```bash
projects/P003-compact-backbones/env/citrinet/.venv/bin/python \
  projects/P003-compact-backbones/code/citrinet/scripts/train_citrinet_p2b.py \
  --train-manifest projects/P003-compact-backbones/experiments/citrinet/train_clean_100_full/manifests/train.jsonl \
  --eval-manifest projects/P003-compact-backbones/experiments/citrinet/train_clean_100_full/manifests/eval.jsonl \
  --output-dir projects/P003-compact-backbones/experiments/citrinet/checkpoints/citrinet_256_p2b_train_clean_100 \
  --max-steps 5000 \
  --batch-size 16 \
  --accumulate-grad-batches 1 \
  --num-workers 4 \
  --lr 1e-4 \
  --seed 17
```

Current blocker:

- none on the ML side
- the remaining work is GPU placement and launch automation
- the current plan is to use a project-local Vast lane with automatic destroy on
  completion because Vast compute billing is per-second while an instance is
  running and storage continues billing while the instance exists

Interpretation:

- Citrinet is ready for the first real GPU-backed run
- the next task is launch and monitor the first Vast-backed run, not more
  Citrinet prototyping

Vast infra status:

- dedicated `P003` SSH template created:
  - `p003_citrinet_nemo_ssh`
  - hash `ab21436ee2fe8894e2aef98578790fe9`
- current orchestrator now defaults to that template
- volume-aware reruns are supported through:
  - `--attach-volume-name <volume_name>`
  - `--volume-mount /data`
  - `--create-volume-on-success <volume_name>`
