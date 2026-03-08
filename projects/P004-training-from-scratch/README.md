# P004 - Training From Scratch

P004 is the lab workspace for building and validating a modern from-scratch audio
training path for phoneme and ASR models. This is not the polished final stack.
It is the place where we de-risk the path, keep the evidence, and decide what
earns its way into the canonical lab setup.

## Working Model

- `reference lane`: stay close to upstream `icefall` so data, labels, k2/lhotse,
  and the training loop can be proven correct with minimal moving parts.
- `canonical lane`: rebuild the validated pieces into the lab stack for newer
  GPUs, `uv`, current PyTorch, modern tracking, and later model variants.
- Rule: if a change can affect correctness or efficiency, isolate it in its own
  run and record the outcome.

## Current Assets

- Vendored upstream recipe base: `third_party/icefall`
- Dedicated reference environment: `.venv-icefall`
- Canonical control-plane project: `pyproject.toml` + `.venv`
- Phone manifest builder: `code/build_librispeech_phone_cuts.py`
- Phone lang prep: `code/prepare_lang_phone.py`
- Reference Conformer launcher: `code/run_conformer_phone_ctc.py`
- Reference setup validator: `code/validate_reference_setup.py`
- Canonical local preflight: `p004-canonical-preflight`
- Built manifests: `experiments/data/manifests_phone_raw`
- Built phone lang dir: `experiments/data/lang_phone`
- Reference checkpoints/logs: `experiments/checkpoints/conformer_ctc_phone`

## Experiment Loop

1. Validate environment, data, and recipe wiring.
2. Run one bounded experiment.
3. Record the result in `docs/EVIDENCE_LEDGER.md`.
4. Decide whether to keep, change, or drop that idea.
5. Only then add the next optimization, architecture branch, or tooling layer.

## Current Status (2026-03-07)

- `R0` proved the reference lane exists, but it is still an older
  `icefall`/`k2` stack in `.venv-icefall`.
- `R1` was attempted on a rented `RTX PRO 4000 Blackwell` and failed at the
  first CUDA model forward with `no kernel image is available for execution on
  the device`.
- `R1` was then re-run on a rented non-Blackwell `RTX 4090` and passed the
  smoke gate: the run logged real optimizer steps, wrote `epoch-0.pt`, and
  exited cleanly.
- `R2` then passed on a RunPod `RTX A5000`: one clean fresh reference run wrote
  `epoch-0.pt` and `epoch-1.pt`, and an explicit resume run loaded `epoch-1.pt`
  and wrote `epoch-2.pt`.
- `C0` local canonical preflight now runs on the local `RTX 5070` with
  `torch 2.9.1+cu128`, `torchaudio 2.9.1+cu128`, `accelerate 1.13.0`, and
  offline W&B smoke logging.
- `C0` local canonical train smoke now runs on the local `RTX 5070` with real
  WAV audio, real phone targets, `torch.compile`, `Accelerate`, a checkpoint,
  machine manifest capture, and offline-safe W&B logging.
- A bounded canonical validation run also passed locally on the same GPU:
  `24` train cuts, `8` dev cuts, `3` epochs, mean train loss falling from
  `12.19 -> 10.02 -> 8.39`, and a recoverable checkpoint written under
  `experiments/checkpoints/canonical_phone_ctc/`.
- That reference env is currently pinned around `torch 2.4.1+cu124`,
  `k2`, and `lhotse`, and it only works with the prepared runtime env that
  injects `LD_LIBRARY_PATH` and `PYTHONPATH`.
- So the blocker is not "the reference lane is broken". The blocker is that the
  old reference stack is not compatible with Blackwell-class GPUs.
- Decision: treat the old `icefall` lane as validated on compatibility-first
  hardware and keep Blackwell for the canonical lane.
- Current canonical blocker: `torchcodec 0.10.0` fails to load on this Arch
  workstation with `torch 2.9.1+cu128`, so the canonical lane currently uses a
  WAV fallback loader for local training while native codec compatibility is
  investigated.
- `C2.0` is now green locally on the `RTX 5070`: the first structured encoder
  variant (`conformer_like`, `1.7M` params) ran for `18` optimizer steps,
  wrote per-epoch checkpoints, and stayed stable under `torch.compile`.
- `C2.1` is also green locally: the canonical trainer can resume from
  `epoch-1.pt`, continue training into later epochs, and write new epoch
  checkpoints in a follow-on run.
- `C2.2` is now green locally: the compile-vs-eager benchmark and direct SDPA
  microbenchmark are written under
  `experiments/benchmarks/canonical_phone_ctc/c2_2_compile_sdpa_20260306_a`.
- Result: on this `RTX 5070` structured-smoke workload, eager beat
  `torch.compile` in both total elapsed time and post-warmup step time, while
  the default SDPA path was healthy with all stable CUDA SDPA backends enabled.
- `C2.3` is now open locally behind `env/nightly-fa4` on the same `RTX 5070`
  with `torch 2.12.0.dev20260306+cu128` and `flash-attn-4 4.0.0b4`.
- Result: compiled nightly `flex_attention` passed with `AUTO` and explicit
  `TRITON`, while the current `FLASH` backend and direct `flash_attn_func`
  failed on this local GPU with `Unsupported compute capability. Supported:
  9.x, 10.x, 11.x`.
- On the bounded nightly microbenchmark at `batch_size=2`, `seq_len=512`,
  `attention_heads=4`, `head_dim=64`, compiled `TRITON` beat nightly SDPA
  forward time (`0.002985s` vs `0.010209s`), while compiled `AUTO` was slower
  (`0.015405s`).
- The first nightly trainer smoke also passed locally with
  `attention_backend=flex_triton`: real optimizer steps, `epoch-0.pt`, and a
  recoverable checkpoint were written under
  `experiments/checkpoints/canonical_phone_ctc/c2_3_local_flex_triton_smoke_20260306_a`.
- The bounded `C2.4` validation failed locally and then reproduced on an
  isolated Vast `RTX PRO 4000 Blackwell` at about `$0.242/hr`: non-finite
  train loss appeared at `epoch=0, batch_index=1` in both places, so the
  current `flex_triton` trainer path is still experimental and not promotable.
- `C2.5` is now also closed red on isolated Vast Blackwell hardware:
  lowering LR to `1e-4` and `1e-5`, switching to `flex_auto`, and forcing
  the CTC loss path to `float32` all still produced the same non-finite train
  loss at `epoch=0, batch_index=1`.
- `F0` is now probed on remote `H100 SXM` at about `$1.49/hr`: the nightly env
  plus `flash-attn-4` installs cleanly, but the current FLASH / FA4 path still
  is not promotable for this project. The H100 benchmark kept `AUTO` and
  `TRITON` green, while FLASH failed with
  `OpError: expects the M-mode to be 64, but got 32`, and a real
  `attention_backend=flex_flash` trainer smoke failed with the same error.
- Immediate next move: keep `enable_compile=false` as the stable local baseline
  for the current canonical trainer, keep nightly `flex_attention` /
  `TRITON` as benchmark-only for now, treat FA4 as not yet viable on our
  current path, and choose between two next branches:
  a full-precision / first-step diagnostic branch on the nightly path, or
  demotion of nightly training work until upstream support changes.
  Keep FA4 in the backlog only as an upstream-retry path, not an active
  higher-cost training option today.
- The promoted stable production baseline is now frozen as:
  `torch 2.9.1+cu128`, `torchaudio 2.9.1+cu128`, `accelerate 1.13.0`,
  `bf16`, `attention_backend=mha`, `enable_compile=false`,
  `model_type=conformer`, `hidden_dim=192`, `encoder_layers=3`,
  `attention_heads=4`, `conv_kernel_size=15`, `dropout=0.1`,
  `learning_rate=3e-4`, machine-manifest capture, and W&B-enabled tracking.
- That frozen path is now exposed as `uv run p004-canonical-train`.
- The first promoted real-Conformer bounded validation
  `canonical_local_conformer_prod_b1_20260307_a` passed locally on the
  `RTX 5070` with online W&B sync, `18` optimizer steps, `epoch-0.pt` through
  `epoch-2.pt`, mean train loss moving `18.31 -> 4.03 -> 4.00`,
  `dev_loss=3.95`, and `peak_memory_reserved_mb=508.0`.
- The first larger raw-manifest bounded run
  `canonical_local_conformer_b2_raw_20260307_a` also passed locally on the
  same stack with `2048` train cuts, `256` dev cuts, `2` epochs, `1024`
  optimizer steps, `dev_loss=1.83`, `dev_per=0.602`, and
  `peak_memory_reserved_mb=588.0`.
- The lazy manifest-backed dataset and duration-aware batching path are now
  green on the same stack: the larger bounded run
  `canonical_local_conformer_b2_raw_lazy_20260307_a` started W&B immediately,
  completed the same `2048` / `256` slice cleanly, and improved to
  `dev_loss=1.7656`, `dev_per=0.5697`, and
  `steady_state_mean_step_seconds=0.0231`.
- A full prepared-manifest epoch is now green locally on the `RTX 5070`:
  `canonical_local_conformer_full_trainclean100_e1_20260307_a` trained on all
  `28,538` `train-clean-100` cuts and evaluated on all `2,703` `dev-clean`
  cuts, finishing with `dev_loss=0.9180`, `dev_per=0.2697`, and
  `peak_memory_reserved_mb=820.0`.
- Full-manifest resume is also green on the same stack:
  `canonical_local_conformer_full_trainclean100_e3_resume_20260307_a`
  resumed from `epoch-0.pt`, completed epochs `1` and `2`, and finished at
  `dev_loss=0.6969` and `dev_per=0.1982`.
- Production next move: keep this real Conformer on the stable trainer and
  scale the actual training problem, not the loader. The next useful branches
  are larger data coverage or a wider / deeper Conformer on the same frozen
  stack. Nightly `TRITON` and FA4 remain separate experiments.

## GPU Activation Rule

Do not start or rent a GPU unless all five items below are already written down:

- exact command
- expected success signal
- explicit stop or kill conditions
- output directory and run ID
- next action if the run passes

If any item is missing, stay local and keep editing, validating, or testing.

## Run Classes

- `P0 local preflight`: seconds to minutes, no rented GPU, used for import
  checks, manifest checks, dry-runs, and env capture
- `S1 smoke run`: the first rented or dedicated-GPU run, bounded to prove one
  real train step and artifact creation
- `B1 bounded validation`: short real training window used to prove repeatable
  loss movement and resume behavior
- `A1 ablation batch`: only after the prior gate is green; this is where we
  change one lever and measure the delta

## Near-Term Ladder

- `G0`: make the next GPU run exact enough to deserve GPU time.
- `R0`: validate the reference lane workspace and capture the current blockers.
- `R1`: run a bounded Conformer smoke train on the phone manifests.
- `R2`: green on RunPod `RTX A5000`.
- `C0`: bootstrap the canonical lane with `uv`, exact locking, run metadata, and
  artifact layout.
- `C1`: machine manifests, offline W&B, and recoverable checkpoints are working
  in the canonical lane.
- `C2.0`: green on local `RTX 5070` with the first structured encoder variant.
- `C2.1`: green on local `RTX 5070` with canonical fresh-plus-resume.
- `C2.2`: green on local `RTX 5070`; eager beat `torch.compile` on this
  structured-smoke benchmark.
- `C2.3`: active locally in `env/nightly-fa4`; compiled nightly
  `flex_attention` works with `AUTO` and `TRITON`, but current `FLASH` / FA4 is
  blocked on compute capability `12.0`.
- `C2.4`: failed locally and on isolated Vast `RTX PRO 4000 Blackwell`; the
  current `flex_triton` trainer path goes non-finite on the bounded validation
  run.
- `C2.5`: failed on isolated Vast `RTX PRO 4000 Blackwell`; lower LR,
  backend choice, and `float32` CTC loss did not stabilize the nightly branch.
- `C2.6`: next diagnostic branch is either full-fp32 / first-step finiteness
  instrumentation or demotion of nightly flex training back to benchmark-only.
- `F0`: initial remote `H100 SXM` FA4 probe is red; the nightly env installs,
  but the FLASH backend and `flex_flash` trainer path currently fail with
  `OpError: expects the M-mode to be 64, but got 32`.
- `Z0`: only branch to Zipformer after the Conformer reference path is stable.

## UV Project Layout

The new `uv` project owns the control plane and other typed helpers that should
be clean, linted, and testable.

- `pyproject.toml`: project metadata, dependencies, entrypoints, Ruff, pytest
- `.python-version`: pins the project interpreter target for `uv`
- `src/p004_training_from_scratch/runpod/client.py`: the only module that talks to
  the official `runpod` Python SDK
- `src/p004_training_from_scratch/vast/models.py`: typed request and response models
- `src/p004_training_from_scratch/vast/query.py`: query and env-string builders
- `src/p004_training_from_scratch/vast/client.py`: the only module that talks to
  `vastai-sdk`
- `src/p004_training_from_scratch/cli/`: thin JSON-only entrypoints
- `tests/`: pure unit tests around query building, normalization, and SDK call shaping

Boundary rule:

- `code/`: reference-lane training and validation scripts, still close to the old
  recipe path
- `src/`: canonical lane control-plane code that must stay typed and reusable

## Canonical Commands

Bootstrap the project env:

```bash
uv sync --group dev
```

Bootstrap the canonical local env:

```bash
uv sync --group dev --group canonical
```

Set auth:

```bash
export RUNPOD_API_KEY=...
export VAST_API_KEY=...
```

If `RUNPOD_API_KEY` is unset, the RunPod wrapper will also fall back to the
existing `~/.runpod/config.toml` `apikey` on this machine.

Project `.env`:

- keep API keys in `.env`, not in the Vast template
- the template may carry non-secret defaults like `WANDB_ENTITY`,
  `WANDB_PROJECT`, cache paths, and `WANDB_MODE`
- the SSH reference template does not need explicit `-p` TCP mappings or UDP
  mappings because it is not exposing an application service yet

Search offers:

```bash
uv run p004-vast-search-offers --gpu-name "RTX 4090" --num-gpus 1
```

List RunPod GPUs with pricing:

```bash
uv run p004-runpod-list-gpus --priced
```

List current RunPod pods:

```bash
uv run p004-runpod-list-pods
```

Create a standard PyTorch pod on RunPod:

```bash
uv run p004-runpod-create-pod \
  --name p004-trainclean360 \
  --gpu-id "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
  --template-id runpod-torch-v280 \
  --volume-in-gb 200 \
  --container-disk-in-gb 40
```

Search templates:

```bash
uv run p004-vast-search-templates
```

Create or update a template:

```bash
uv run p004-vast-upsert-template --name p004_ref_icefall_ssh --image nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

Delete a template:

```bash
uv run p004-vast-delete-template --name p004_ref_icefall_ssh_debug
```

Run the reference-lane preflight checks:

```bash
uv run p004-reference-checks
```

Capture a machine manifest:

```bash
uv run p004-capture-machine-manifest
```

Run the local canonical preflight:

```bash
uv run p004-canonical-preflight
```

Run the local canonical real-data smoke:

```bash
uv run p004-canonical-train-smoke
```

Run the promoted stable canonical trainer with the frozen production defaults:

```bash
uv run p004-canonical-train
```

Run the bounded local canonical validation with an explicit run ID:

```bash
uv run p004-canonical-train \
  --run-id canonical_local_conformer_prod_b1_20260307_a
```

Run a larger bounded validation on the raw prepared manifests:

```bash
uv run p004-canonical-train \
  --run-id canonical_local_conformer_b2_raw_20260307_a \
  --train-manifest experiments/data/manifests_phone_raw/librispeech_cuts_train-clean-100.jsonl.gz \
  --dev-manifest experiments/data/manifests_phone_raw/librispeech_cuts_dev-clean.jsonl.gz \
  --train-limit 2048 \
  --dev-limit 256 \
  --epochs 2 \
  --batch-size 4
```

Run the first structured local Blackwell experiment:

```bash
uv run p004-canonical-train-smoke \
  --run-id canonical_local_c2_structured_r2_fresh_20260306_a \
  --train-limit 24 \
  --dev-limit 8 \
  --epochs 3 \
  --batch-size 4 \
  --model-type conformer_like \
  --hidden-dim 192 \
  --encoder-layers 3 \
  --attention-heads 4 \
  --conv-kernel-size 15 \
  --dropout 0.1
```

Resume a canonical run from an epoch checkpoint:

```bash
uv run p004-canonical-train \
  --run-id canonical_local_prod_resume_20260307_a \
  --epochs 4 \
  --resume-from \
  experiments/checkpoints/canonical_phone_ctc/canonical_local_conformer_prod_b1_20260307_a/epoch-1.pt
```

Run the stable-lane compile versus eager benchmark:

```bash
uv run p004-canonical-benchmark \
  --output-dir experiments/benchmarks/canonical_phone_ctc/c2_2_compile_sdpa_20260306_a
```

Bootstrap the separate nightly attention env:

```bash
uv venv env/nightly-fa4 --python 3.11
uv pip install --python env/nightly-fa4/bin/python --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install --python env/nightly-fa4/bin/python --prerelease=allow flash-attn-4
uv pip install --python env/nightly-fa4/bin/python "accelerate>=1.13.0"
uv pip install --python env/nightly-fa4/bin/python -e .
```

Run the nightly `C2.3` attention benchmark:

```bash
env/nightly-fa4/bin/python -m p004_training_from_scratch.cli.nightly_attention_benchmark \
  --output-dir experiments/benchmarks/canonical_phone_ctc/c2_3_nightly_attention_20260306_a
```

Run the nightly `flex_triton` trainer smoke:

```bash
env/nightly-fa4/bin/python -m p004_training_from_scratch.cli.canonical_train_smoke \
  --output-dir experiments/checkpoints/canonical_phone_ctc/c2_3_local_flex_triton_smoke_20260306_a \
  --train-limit 12 \
  --dev-limit 4 \
  --epochs 1 \
  --batch-size 4 \
  --model-type conformer_like \
  --attention-backend flex_triton \
  --hidden-dim 192 \
  --encoder-layers 3 \
  --attention-heads 4 \
  --conv-kernel-size 15 \
  --dropout 0.1 \
  --disable-wandb
```

Allow online W&B tracker init during the canonical preflight:

```bash
uv run p004-canonical-preflight --allow-online-trackers
```

Show instances:

```bash
uv run p004-vast-show-instances
```

Launch an instance:

```bash
uv run p004-vast-launch-instance \
  --gpu-name "RTX 4090" \
  --image "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04" \
  --template-hash "86ba9f2d188dfb4a3215babe232434f6" \
  --disk-gb 100 \
  --order dph \
  --label "p004-ref" \
  --query-clause "reliability > 0.995" \
  --query-clause "cpu_cores >= 16"
```

Destroy an instance:

```bash
uv run p004-vast-destroy-instance --instance-id 32477137
```

Quality gates:

```bash
uv run ruff check src tests
uv run ruff format src tests
uv run ty check src tests
uv run pytest
```

## Source Of Truth

- This file: workspace operating model
- `docs/EVIDENCE_LEDGER.md`: what has actually been validated
- `docs/ABLATION_PLAN.md`: next experiment ladder and acceptance gates
- `docs/README.md`: manuscript/supporting-doc index

## References

```text
PyTorch 2.9 release blog
https://pytorch.org/blog/pytorch-2-9/

PyTorch FlexAttention + FlashAttention-4 on Hopper/Blackwell
https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/

uv locking and syncing
https://docs.astral.sh/uv/concepts/projects/sync/

Hugging Face Accelerate trackers
https://huggingface.co/docs/accelerate/main/en/package_reference/tracking

Hugging Face Hub uploads
https://huggingface.co/docs/huggingface_hub/guides/upload

W&B Artifacts
https://docs.wandb.ai/models/artifacts

icefall LibriSpeech Conformer CTC recipe
https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conformer_ctc
```
