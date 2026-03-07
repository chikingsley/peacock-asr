# Track 07 Evidence Ledger: Training from Scratch

Scope:

- From-scratch Conformer/Zipformer phoneme CTC training
- Comparison with fine-tuned SSL encoders (Track 05)
- Data requirements and compute cost analysis

Citation policy:

- Use numbered citations in text: `[1]`, `[2]`, ...
- Use `./refs.bib` as canonical bib source.

---

## 0. Current Workspace State (2026-03-06)

Validation ID:

- `R0` reference workspace validation

Current status:

- `partial pass`
- artifact written: `experiments/validation/reference_setup.json`
- workspace readiness on `2026-03-07`: `ready for R2 on a compatible non-Blackwell GPU`

Validated facts:

- Phone manifests already exist at `experiments/data/manifests_phone_raw`.
- `build_summary.json` reports `28,538` kept cuts for `train_clean_100` and
  `2,703` kept cuts for `dev_clean`.
- The manifest builder materialized local `16 kHz` wav files and Lhotse-style
  cuts for the phone-label path.
- `lang_phone` exists with `tokens.txt`, `lexicon.txt`, `phone_list.txt`, and
  related files under `experiments/data/lang_phone`.
- `third_party/icefall` is vendored locally and `.venv-icefall` exists.
- The reference launcher already prepends `LD_LIBRARY_PATH` and `PYTHONPATH`
  before invoking the upstream `icefall` Conformer recipe.

Observed issues:

- Raw imports from `.venv-icefall` fail in a plain shell with
  `ImportError: libnvrtc.so.12` unless the runtime library path is prepared.
- The `R0` validator succeeded with prepared-env imports and manifest checks, but
  reported `overall=partial` because of the raw import issue above and because
  the local GPU was already occupied by another run.
- The existing `conformer_ctc_phone` log from `2026-03-06 02:51:30` shows
  initialization and `Converting L.pt to Linv.pt`, but no logged train step or
  completed epoch.
- The local `RTX 5070 12GB` was already occupied by a separate `P003` HuBERT
  run at the time of validation, so no new reference train was started.

Implication:

- P004 is beyond the stub stage. The workspace already has a real reference
  path, but it still needs a clean, rerunnable `R1` smoke train before the
  canonical lane should absorb it.

Next action:

- Run `code/validate_reference_setup.py` and store the output under
  `experiments/validation/`.
- Complete `G0` for the next run: exact command, output dir, success signal,
  stop conditions, monitoring cadence, and next action.
- Use the successful `RTX 4090` smoke run as the anchor for `R2`.
- Do not spend more reference-lane time on Blackwell for the old stack.

---

## 0.1 `R1` Vast Smoke Run (2026-03-06)

Validation ID:

- `R1` bounded reference smoke train on rented `RTX PRO 4000 Blackwell`

Current status:

- `failed with a concrete hardware-stack incompatibility`
- artifacts written:
  `experiments/checkpoints/conformer_ctc_phone/r1_smoke_vast_20260307/train.log`
- command captured:
  `experiments/checkpoints/conformer_ctc_phone/r1_smoke_vast_20260307/run_command.sh`
- machine manifest captured:
  `experiments/validation/r1_smoke_vast_20260307_machine_manifest.json`

Validated facts:

- The old reference lane is now proven to launch on a remote node far enough to
  build the model, load the phone lexicon, construct the train and dev
  dataloaders, and enter icefall's pessimistic pre-training batch scan.
- The remote smoke subset with `train-clean-100`, `dev-clean`, and a temporary
  `dev-other` alias was sufficient to get beyond the manifest and dataset
  plumbing.
- The first hard failure on the rented node was not data, not launcher wiring,
  and not `k2` graph compilation.

Observed issues:

- PyTorch emitted an explicit warning that the rented `RTX PRO 4000 Blackwell`
  (`sm_120`) is not supported by the current reference env
  (`torch 2.4.1+cu124`).
- The actual training failure was:
  `RuntimeError: CUDA error: no kernel image is available for execution on the device`
  during the first pessimistic batch scan in the model forward path.
- The rented `RTX PRO 4000` instance was therefore the wrong compatibility
  target for the old reference lane, even though it was cheap.

Implication:

- The reference lane is now blocked specifically by `Blackwell + torch 2.4.1`
  incompatibility.
- The next reference-lane rented run should use a non-Blackwell GPU
  (`RTX 4090`-class or similar), or the lane itself must be rebuilt on a newer
  PyTorch and `k2` stack.

Next action:

- Do not rent `RTX PRO 4000 Blackwell` again for the old `icefall` reference
  lane.
- Re-run `R1` on a non-Blackwell GPU, or explicitly promote this failure into a
  canonical-lane rebuild task.

---

## 0.2 `R1` Vast Smoke Run (2026-03-07)

Validation ID:

- `R1` bounded reference smoke train on rented `RTX 4090`

Current status:

- `passed`
- run contract written:
  `experiments/validation/r1_smoke_vast_4090_20260306_191106_contract.json`
- artifacts written:
  `experiments/checkpoints/conformer_ctc_phone/r1_smoke_vast_4090_20260306_191106/train.log`
- checkpoints written:
  `experiments/checkpoints/conformer_ctc_phone/r1_smoke_vast_4090_20260306_191106/epoch-0.pt`
- machine manifest captured:
  `experiments/validation/r1_smoke_vast_4090_20260306_191106_machine_manifest.json`

Validated facts:

- The old reference lane is not generically broken. On a non-Blackwell
  `RTX 4090`, the same old `torch 2.4.1+cu124 + k2 + lhotse + icefall`
  stack launched cleanly, got through the same pessimistic OOM scan that
  failed on Blackwell, logged real optimizer steps, and wrote checkpoints.
- The smoke subset is sufficient for a true end-to-end reference-lane launch:
  manifest load, lang load, model init, dataloaders, train step, and checkpoint.
- The reference launcher remained close to stock upstream `icefall`; the local
  modifications were limited to env prep, manifest/lang wiring, and bounded
  smoke arguments.

Observed issues:

- The remote `icefall` repo emitted `git safe.directory` warnings because the
  copied tree ownership differed on the rented node. This did not block the
  run, but it left `icefall` git metadata null in the log.
- Reusing the copied local `.venv-icefall` required copying the linked
  `uv`-managed Python runtime under `/home/simon/.local/share/uv/python`.
- The rented image did not have `uv` on `PATH`, so the remote smoke command was
  executed with `python3 code/run_conformer_phone_ctc.py` rather than the
  `uv run --script` shebang path.

Implication:

- `R1` is now green for the reference lane on compatibility-first hardware.
- The next gate is `R2`: prove the small reference run is repeatable and that
  resume or restart behavior is understood.
- The local Blackwell GPU should still be reserved for the canonical lane, not
  for the old reference stack.

Next action:

- Re-run the same small reference path as `R2`, either as a clean fresh rerun
  on another compatible node or as a bounded resume or restart validation.
- Fold the non-blocking remote bootstrap issues into the control plane before
  spending more rented time.

---

## 0.2.1 `R2` RunPod Reference Repeatability (2026-03-07)

Validation ID:

- `R2` stable small reference run on RunPod `RTX A5000`

Current status:

- `passed`
- run contract written:
  `experiments/validation/r2_reference_runpod_a5000_20260306_202527_contract.json`
- fresh-run artifacts written:
  `experiments/checkpoints/conformer_ctc_phone/r2_reference_runpod_a5000_20260306_202527/train_fresh.log`
- resume-run artifacts written:
  `experiments/checkpoints/conformer_ctc_phone/r2_reference_runpod_a5000_20260306_202527/train_resume.log`
- checkpoints written:
  `experiments/checkpoints/conformer_ctc_phone/r2_reference_runpod_a5000_20260306_202527/epoch-0.pt`,
  `epoch-1.pt`, and `epoch-2.pt`
- machine manifest captured:
  `experiments/validation/r2_reference_runpod_a5000_20260306_202527_machine_manifest.json`

Validated facts:

- The old `icefall` reference lane is now repeatable on compatible non-Blackwell
  hardware, not just able to pass a single smoke launch.
- A clean fresh run on RunPod `RTX A5000` trained for two epochs and wrote
  `epoch-0.pt` and `epoch-1.pt`.
- An explicit resume run then loaded `epoch-1.pt`, resumed at epoch `2`, and
  wrote `epoch-2.pt`.
- The reference launcher still remained close to upstream `icefall`; the result
  did not require a recipe rewrite or canonical-lane code path.

Observed issues:

- The RunPod box had an unrelated active `P003` sweep process using a small
  amount of GPU memory, so `R2` should be treated as a correctness and resume
  gate, not as a clean throughput benchmark.
- Reusing the copied `.venv-icefall` on the pod still required the linked
  `uv`-managed CPython runtime tree under
  `/home/simon/.local/share/uv/python`.

Implication:

- `R2` is green.
- The old reference lane no longer blocks canonical local Blackwell work.
- The next questions belong to the canonical lane: structured encoder shape,
  explicit resume behavior, and then stable-kernel benchmarking.

Next action:

- Keep the reference lane frozen unless a later architecture comparison needs it.
- Move the local `RTX 5070` canonical lane into structured encoder and resume
  validation.

---

## 0.3 `C0` Local Canonical Preflight (2026-03-07)

Validation ID:

- `C0` local canonical preflight on the workstation `RTX 5070`

Current status:

- `partial pass`
- artifacts written:
  `experiments/validation/canonical_local_preflight.json`
- machine manifest written:
  `experiments/validation/canonical_local_machine_manifest.json`

Validated facts:

- The local canonical lane now has a reproducible `uv` dependency group with
  `torch 2.9.1+cu128`, `torchaudio 2.9.1+cu128`, `accelerate 1.13.0`, `wandb`,
  and `torchcodec 0.10.0`.
- On the local `RTX 5070` (`compute_capability=12.0`), the canonical preflight
  passed:
  `bf16_matmul_backward`, `scaled_dot_product_attention`,
  `compiled_ctc_train_smoke`, `accelerate_train_step`, and `wandb_smoke`.
- The compiled tiny CTC smoke used real local WAV audio, real mel features,
  `torch.compile`, and three optimizer steps with falling loss values.
- W&B smoke logging worked in forced offline mode and produced a local offline
  run directory.

Observed issues:

- `torchcodec_runtime` failed on this Arch workstation. The current
  `torchcodec 0.10.0` build could not load its shared library against the local
  runtime, with an undefined symbol error under the FFmpeg 8 path and missing
  FFmpeg sonames for older variants.
- Because of that codec issue, the preflight currently uses a WAV fallback
  loader for the real-audio smoke batch instead of native TorchCodec decode.
- The canonical preflight therefore reports `overall_passed=false` even though
  the core CUDA, compile, Accelerate, and W&B checks succeeded.

Implication:

- The local Blackwell-compatible canonical stack is materially alive.
- The immediate blocker is not CUDA, not `torch.compile`, not bf16, and not
  Accelerate. The immediate blocker is codec-runtime integration on this local
  workstation.

Next action:

- Build the first canonical local train-smoke script on top of the now-validated
  CUDA path and WAV fallback loader.
- Separately isolate the `torchcodec` mismatch and decide whether to pin a
  different codec version, install matching FFmpeg libs, or defer native codec
  use until a later lane.

---

## 0.4 `C0` Local Canonical Train Smoke (2026-03-07)

Validation ID:

- `C0` local canonical real-data train smoke on the workstation `RTX 5070`

Current status:

- `passed`
- artifacts written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_smoke_20260306_a/report.json`
- checkpoint written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_smoke_20260306_a/checkpoint.pt`
- metrics written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_smoke_20260306_a/metrics.jsonl`

Validated facts:

- The canonical lane can now run a true local train step on the Blackwell-class
  workstation GPU using real WAV audio and real phone-token targets from the
  prepared smoke manifests.
- The run used the project-local `uv` env, `torch 2.9.1+cu128`,
  `torchaudio 2.9.1+cu128`, `Accelerate`, `torch.compile`, machine-manifest
  capture, and W&B offline tracking in one path.
- The run wrote a recoverable checkpoint, metrics log, and report without any
  rented GPU and without touching the old `icefall` reference stack.

Observed issues:

- The tiny smoke model is only a stack-validation model. After one epoch on
  `12` train cuts and `4` dev cuts, the dev PER remained effectively random
  (`0.9864`), even though train loss fell within the short run.
- Native `torchcodec` decode is still broken on this Arch workstation, so the
  local canonical lane remains on the WAV fallback path for now.

Implication:

- The local canonical lane is no longer blocked on "can it really train with
  real data." That answer is now yes.
- The next question is not environment bring-up. The next question is whether a
  slightly less toy setup shows stable bounded improvement.

Next action:

- Run a bounded local validation with more cuts and more epochs on the same
  stack.
- Keep `torchcodec` investigation separate from the core train-smoke path.

---

## 0.5 `B1` Local Canonical Bounded Validation (2026-03-07)

Validation ID:

- `B1` bounded local canonical validation on the workstation `RTX 5070`

Current status:

- `passed`
- artifacts written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_b1_20260306_a/report.json`
- checkpoint written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_b1_20260306_a/checkpoint.pt`
- metrics written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_b1_20260306_a/metrics.jsonl`

Validated facts:

- The canonical lane can repeat the same real-data path in a stronger bounded
  window on the local `RTX 5070`: `24` train cuts, `8` dev cuts, `3` epochs,
  `18` optimizer steps.
- Mean train loss fell by epoch:
  `12.19 -> 10.02 -> 8.39`.
- Dev loss improved materially relative to the one-epoch smoke (`48.91 -> 15.05`)
  while the run still wrote a checkpoint, metrics log, manifest, and offline W&B
  run in one pass.

Observed issues:

- Dev PER is still very high (`0.9669`), which is expected at this stage
  because the model is intentionally tiny and the dataset is extremely small.
- This run proves stack behavior, not model quality.

Implication:

- The local Blackwell-compatible canonical lane is good enough to move into
  actual model work instead of more environment churn.
- The next leverage point is model structure and resume behavior, not basic CUDA
  or tracking integration.

Next action:

- Keep the tiny-model results as the stable baseline.
- Move into the first structured encoder variant and then explicit resume
  validation on the same canonical lane.

---

## 0.6 `C2.0` Local Structured Encoder Baseline (2026-03-07)

Validation ID:

- `C2.0` first structured canonical encoder on the workstation `RTX 5070`

Current status:

- `passed`
- run contract written:
  `experiments/validation/canonical_local_c2_structured_b1_20260306_a_contract.json`
- artifacts written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_c2_structured_b1_20260306_a/report.json`
- checkpoint written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_c2_structured_b1_20260306_a/checkpoint.pt`
- metrics written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_c2_structured_b1_20260306_a/metrics.jsonl`

Validated facts:

- The canonical lane can train a first structured encoder variant on the local
  `RTX 5070` under `torch 2.9.1+cu128`, bf16, `torch.compile`, `Accelerate`,
  and offline W&B.
- The new `conformer_like` model uses attention plus depthwise-convolution
  blocks rather than the earlier tiny conv-FFN stack.
- The run completed `18` optimizer steps over `3` epochs with a `1.7M`
  parameter model and wrote a recoverable checkpoint.

Observed issues:

- This first structured run was a fresh-only baseline. It completed before
  canonical resume support was added, so it did not yet prove resume behavior.
- Dev quality remained poor on the tiny smoke subset (`dev_per=0.9914`), so the
  result should be treated as a systems baseline, not a model-quality win.

Implication:

- The local Blackwell-compatible canonical lane can absorb a more realistic
  encoder shape without tripping over CUDA, bf16, or `torch.compile`.
- The next canonical gate is explicit fresh-plus-resume on that same structured
  path.

Next action:

- Add recoverable epoch checkpoints and resume support to the canonical trainer.
- Re-run the structured encoder path with explicit resume.

---

## 0.7 `C2.1` Local Structured Fresh And Resume (2026-03-07)

Validation ID:

- `C2.1` structured canonical fresh-plus-resume on the workstation `RTX 5070`

Current status:

- `passed`
- fresh artifacts written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_c2_structured_r2_fresh_20260306_a/report.json`
- fresh epoch checkpoints written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_c2_structured_r2_fresh_20260306_a/epoch-0.pt`,
  `epoch-1.pt`, and `epoch-2.pt`
- resume artifacts written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_c2_structured_r2_resume_20260306_a/report.json`
- resume epoch checkpoints written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_c2_structured_r2_resume_20260306_a/epoch-2.pt`
  and `epoch-3.pt`

Validated facts:

- The canonical trainer now writes recoverable per-epoch checkpoints plus a
  latest `checkpoint.pt`.
- The canonical trainer can resume from `epoch-1.pt`, restore optimizer state,
  continue training at epoch `2`, and write later epoch checkpoints in a
  follow-on run.
- A real `torch.compile` checkpoint compatibility bug was found and fixed:
  compiled checkpoints initially saved model weights under `_orig_mod.*`, and
  the trainer now normalizes those keys on save and load.
- The resumed structured run improved dev loss from `15.19` on the fresh
  structured baseline to `13.49` while holding the same tiny dev PER
  (`0.9914`) on the smoke subset.

Observed issues:

- The smoke subset is still too small and undertrained to treat PER as a useful
  model-selection metric.
- Native `torchcodec` decode remains out of scope for this result; the canonical
  lane is still using the WAV fallback loader.

Implication:

- `C2.1` is green.
- The stable local Blackwell lane now has the three properties we needed before
  kernel experiments: real data, a structured encoder, and explicit resume.

Next action:

- Start `C2.2`: benchmark the stable local lane with and without
  `torch.compile`, and measure the default SDPA attention path before opening a
  nightly or FA4 branch.

---

## 0.8 `C2.2` Local Stable-Lane Compile And SDPA Benchmark (2026-03-07)

Validation ID:

- `C2.2` stable local benchmark on the workstation `RTX 5070`

Current status:

- `passed`
- benchmark summary written:
  `experiments/benchmarks/canonical_phone_ctc/c2_2_compile_sdpa_20260306_a/report.json`
- eager variant written:
  `experiments/benchmarks/canonical_phone_ctc/c2_2_compile_sdpa_20260306_a/compile_off/report.json`
- compile variant written:
  `experiments/benchmarks/canonical_phone_ctc/c2_2_compile_sdpa_20260306_a/compile_on/report.json`

Validated facts:

- The stable local canonical lane now has a direct compile-vs-eager benchmark
  on the same structured `conformer_like` model used for `C2.0` and `C2.1`.
- The direct SDPA microbenchmark on the same local machine completed cleanly at
  `seq_len=1024`, `batch_size=4`, `hidden_dim=192`, `attention_heads=4`, and
  bfloat16.
- All stable CUDA SDPA backends reported enabled in the benchmark environment:
  flash, mem-efficient, and math.
- On this workload, eager beat `torch.compile`:
  `elapsed_seconds=1.136` vs `3.442`,
  `steady_state_mean_step_seconds=0.011842` vs `0.054540`,
  `peak_memory_reserved_mb=356.0` vs `370.0`.
- Even after excluding the earliest warmup-heavy steps from the raw metrics,
  the compile variant remained slower than eager on this benchmark.

Observed issues:

- The compile variant paid two very expensive early steps
  (`0.973949s` and `0.879331s`) before settling down.
- This benchmark is still a bounded smoke-scale workload, so it does not prove
  that `torch.compile` is always a loss at larger sequence counts or larger
  models. It proves only that it is not the right stable default for the
  current local structured baseline.

Implication:

- `C2.2` is green.
- The stable local baseline for the next branch should keep
  `enable_compile=false`.
- The next branch is now cleanly defined: `C2.3` can focus on nightly or FA4
  kernel experiments without also debating whether the stable baseline should be
  compiled.

Next action:

- Open `C2.3` behind a separate nightly or FA4 environment.
- Keep the current stable baseline on the local `RTX 5070` as:
  structured encoder + default SDPA path + eager execution.

---

## 0.9 `C2.3` Local Nightly Attention Branch (2026-03-07)

Validation ID:

- `C2.3` nightly attention benchmark on the workstation `RTX 5070`

Current status:

- `passed` for the local nightly branch
- benchmark summary written:
  `experiments/benchmarks/canonical_phone_ctc/c2_3_nightly_attention_20260306_a/report.json`
- machine manifest written:
  `experiments/benchmarks/canonical_phone_ctc/c2_3_nightly_attention_20260306_a/machine_manifest.json`

Validated facts:

- A dedicated nightly env now exists at `env/nightly-fa4`.
- The local nightly stack installed cleanly with
  `torch 2.12.0.dev20260306+cu128`, `torchaudio 2.11.0.dev20260306+cu128`, and
  `flash-attn-4 4.0.0b4`.
- The local `RTX 5070` reports compute capability `12.0` in that env.
- Direct `flash_attn_func` failed immediately with:
  `Unsupported compute capability. Supported: 9.x, 10.x, 11.x`.
- Compiled PyTorch nightly `flex_attention` completed successfully with:
  `AUTO` and explicit `TRITON`.
- Compiled nightly `flex_attention` with `BACKEND=FLASH` failed with the same
  unsupported compute capability assertion as the direct FA4 package call.
- On the bounded local microbenchmark, compiled `TRITON` beat nightly SDPA
  forward time:
  `0.002985s` vs `0.010209s`.
- Compiled `AUTO` was slower than nightly SDPA on the same workload:
  `0.015405s` vs `0.010209s`.

Observed issues:

- The first compiled `flex_attention` attempt hit a real CUDAGraph reuse error.
- The working benchmark required explicit `torch.compiler.cudagraph_mark_step_begin()`
  boundaries and cloning the compiled output before reuse.
- So the local nightly path is usable, but it is not yet drop-in safe for the
  training loop without carrying those runtime rules forward.

Implication:

- `C2.3` is green as a nightly local branch.
- `C2.3` is not green as a true local FA4 branch on this `RTX 5070`.
- The local “fun path” is currently compiled nightly `flex_attention` with
  `TRITON`, not FA4.

Next action:

- Decide whether to integrate compiled `flex_attention` with `TRITON` into the
  experimental canonical model path on the local workstation.
- Keep FA4-specific work for hardware where the current package accepts the
  compute capability.

---

## 0.9.1 `C2.3` Local Nightly Flex-TRITON Trainer Smoke (2026-03-07)

Validation ID:

- `C2.3` local trainer smoke with `attention_backend=flex_triton`

Current status:

- `passed`
- trainer smoke written:
  `experiments/checkpoints/canonical_phone_ctc/c2_3_local_flex_triton_smoke_20260306_a/report.json`

Validated facts:

- The nightly env can run the real canonical trainer, not just the isolated
  attention microbenchmark.
- The local `RTX 5070` completed a one-epoch smoke with:
  `model_type=conformer_like`, `attention_backend=flex_triton`,
  `hidden_dim=192`, `encoder_layers=3`, `attention_heads=4`.
- The run wrote `epoch-0.pt`, `checkpoint.pt`, `model_state.pt`, and
  `metrics.jsonl`.
- The run completed `3` optimizer steps and recorded finite train and dev loss.

Observed issues:

- Compile overhead remains very large on the first step:
  `first_step_seconds=13.986556`.
- The nightly flex path still depends on explicit CUDAGraph step markers and
  cloned attention outputs.

Implication:

- The nightly local branch is good enough for real smoke tests.
- It still needs a bounded validation gate before it can be promoted.

Next action:

- Run `C2.4` bounded validation on the same local nightly branch.

---

## 0.10 `C2.4` Local Nightly Flex-TRITON Bounded Validation (2026-03-07)

Validation ID:

- `C2.4` local bounded validation with `attention_backend=flex_triton`

Current status:

- `failed`
- bounded validation report written:
  `experiments/checkpoints/canonical_phone_ctc/c2_4_local_flex_triton_bounded_20260306_b/report.json`

Validated facts:

- The non-finite-loss gate now works correctly in the canonical trainer.
- On the bounded local nightly validation (`24` train cuts, `8` dev cuts,
  `3` epochs, `batch_size=4`), the current `flex_triton` branch failed fast.
- The failure happened at:
  `epoch=0`, `batch_index=1`, `train_loss=NaN`.

Observed issues:

- The current local `flex_triton` training path is not numerically stable under
  the bounded validation setting.
- This means the branch cannot be promoted from experimental smoke status into
  the canonical training path yet.

Implication:

- `C2.4` is red.
- The stable local canonical baseline remains:
  eager execution + default SDPA path.
- The local nightly `flex_triton` branch remains experimental until the
  non-finite-loss issue is explained and fixed.

Next action:

- Decide whether to try one stabilization branch locally
  (for example LR, dtype, or narrower integration changes) or drop the local
  `flex_triton` path back to benchmark-only status.
- Keep the true FA4 path as a separate remote validation target on
  `H100/H200/B200`-class hardware.

---

## 0.10.1 `C2.4` Remote Blackwell Reproduction On Vast (2026-03-07)

Validation ID:

- `C2.4` isolated bounded validation on Vast `RTX PRO 4000 Blackwell`

Current status:

- `failed`
- remote bounded validation report written:
  `experiments/checkpoints/canonical_phone_ctc/c2_4_remote_vast_rtxpro4000_20260307_a/report.json`

Validated facts:

- The same bounded nightly trainer configuration reproduced on an isolated
  remote Blackwell box using `torch 2.12.0.dev20260306+cu128` and
  `attention_backend=flex_triton`.
- The remote GPU was not sharing the run with the local `P003` process, so this
  run removes local workstation contention as the primary explanation.
- The failure again happened at:
  `epoch=0`, `batch_index=1`, `train_loss=NaN`.
- The rented Vast box cost about `$0.242/hr`, and the instance was destroyed
  immediately after the validation completed.

Observed issues:

- The instability is reproducible across two Blackwell environments:
  local `RTX 5070` and remote `RTX PRO 4000 Blackwell`.
- That means the current nightly `flex_triton` trainer path should be treated as
  a real numerical-stability bug, not a one-off scheduling artifact.

Implication:

- `C2.4` remains red after remote reproduction.
- The stable canonical path still remains:
  eager execution + default SDPA path.
- The nightly branch stays experimental until a stabilization branch produces a
  bounded run with finite loss.

Next action:

- Run the stabilization ladder in a fixed order:
  lower LR first, then compare `flex_auto` vs `flex_triton`, then narrow dtype
  scope if needed.
- Keep true FA4 as a separate future branch on `H100/H200/B200`-class hardware.

---

## 1. Claim Map

| ID | Claim | Evidence Status | Primary Citations |
|----|-------|----------------|------------------|
| C1 | ZIPA (Zipformer from scratch on 17K hours) achieves 2.71 PFER on seen languages | Paper result | [1] |
| C2 | ZIPA's 127-char IPA vocabulary breaks GOP scoring (diphthongs as single tokens) | Architectural analysis | [1], internal |
| C3 | PRiSM finds encoder-CTC is most stable approach for phone recognition | Paper result | [3] |
| C4 | POWSM (multi-task Whisper-style 350M) outperforms ZIPA and wav2vec2-phoneme | Paper result | [2] |
| C5 | icefall TIMIT TDNN-LSTM achieves 17.66% PER | Paper result | [7] |
| C6 | Conformer is the standard modern ASR architecture (CNN+Transformer hybrid) | Widely established | [4] |
| C7 | From-scratch training needs 1K+ hours minimum for competitive phone posteriors | Estimate from ZIPA data scale | [1], [2] |
| C8 | Fine-tuned w2v-BERT 2.0 achieves PCC 0.648 on SpeechOcean762 GOP scoring | Internal result (Track 05) | Internal |
| C9 | LibriSpeech 960h phone alignments available (gilkeyio/librispeech-alignments) | Dataset exists | [6] |
| C10 | Zipformer outperforms Conformer in icefall benchmarks | icefall documentation | [7] |

---

## 2. External Reference Papers

| # | Paper | Key Result | Status |
|---|-------|-----------|--------|
| [1] | Zhu et al. (2025) "ZIPA: Efficient Multilingual Phone Recognition" | Zipformer on 17K hours, 2.71 PFER | PDF in papers/, code at github.com/lingjzhu/zipa |
| [2] | Bigi et al. (2025) "POWSM: Phonetic Open Whisper-Style Speech Foundation Model" | 350M multi-task, beats ZIPA on phone recognition | PDF in papers/ |
| [3] | (2025) "PRiSM: Benchmarking Phone Realization in Speech Models" | encoder-CTC most stable for phone recognition | PDF in papers/ |
| [4] | Gulati et al. (2020) "Conformer: Convolution-augmented Transformer for Speech Recognition" | CNN+Transformer hybrid, standard ASR architecture | PDF in papers/ |
| [5] | Conneau et al. (2021) "Simple and Effective Zero-Shot Cross-Lingual Phoneme Recognition" | wav2vec2 zero-shot phoneme transfer | PDF in papers/ |
| [6] | Pratap et al. (2023) "Scaling Speech Technology to 1000+ Languages" (MMS) | Massive multilingual phone recognition | PDF in papers/ |
| [7] | k2-fsa/icefall | TIMIT TDNN-LSTM 17.66% PER, Zipformer LibriSpeech recipes | github.com/k2-fsa/icefall |
| [8] | Yao et al. (2025) "Towards Accurate Phonetic Error Detection Through Phoneme Similarity Modeling" | Phoneme similarity for error detection | PDF in papers/ (2507.14346) |
| [9] | (2025) "LCS-CTC: Leveraging Soft Alignments" | CTC variant with soft alignments | PDF in papers/ (2508.03937) |
| [10] | Gutkin et al. (2022) "ByT5 model for massively multilingual G2P" | G2P across 100 languages, needed for data labeling | PDF in papers/ |

---

## 3. Internal Evidence Anchors

- Track 05 / P001 scoring baseline evidence:
  `projects/P001-gop-baselines/experiments/final/results/aggregate_summary.tsv`
- GOP feature extraction pipeline: `/home/simon/github/peacock-asr/projects/P001-gop-baselines/code/p001_gop/gop.py`
- CTC posterior extraction: `/home/simon/github/peacock-asr/projects/P001-gop-baselines/code/p001_gop/backends/ctc_gop_original.py`
- GOPT scorer: `/home/simon/github/peacock-asr/projects/P001-gop-baselines/code/p001_gop/gopt_model.py`
- LibriSpeech alignments: gilkeyio/librispeech-alignments on HuggingFace (960h, ARPABET)

---

## 4. Key Technical Details

### ZIPA Architecture

- Model: Zipformer (k2-fsa variant of Conformer)
- Training data: IPAPack++ 17K hours, 88 languages
- Output vocabulary: 127 IPA characters
- PFER: 2.71 on seen languages, 0.66 on English
- Critical limitation: IPA character-level vocabulary means diphthongs (AW, AY, EY, OW, OY)
  and affricates (CH, JH) are split into multiple tokens, breaking ARPABET-based GOP scoring

### POWSM Architecture

- Model: Whisper-style encoder-decoder, 350M parameters
- Training: multi-task (phone recognition + ASR + G2P + P2G)
- Framework: ESPnet
- Result: outperforms ZIPA and wav2vec2-phoneme on phone recognition tasks

### PRiSM Benchmark Findings

- Compared encoder-CTC, encoder-decoder, and other approaches for phone recognition
- Finding: encoder-CTC is the most stable architecture for phone-level output
- This supports using CTC (not seq2seq) for our GOP posterior extraction

### Conformer Architecture (Gulati 2020)

- Combines CNN and Multi-Head Self-Attention in each block
- CNN captures local acoustic patterns; MHSA captures global context
- Standard modern ASR architecture, basis for Zipformer and FastConformer
- icefall has Conformer LibriSpeech recipes but only BPE/word-level output

### Data Labeling Pipeline

For large unlabeled datasets, G2P labeling is needed:

1. LibriSpeech 960h: pre-labeled ARPABET via forced alignment (gilkeyio/librispeech-alignments)
2. CommonVoice: text available, needs G2P (ByT5-based multilingual G2P [10])
3. IPAPack++: ZIPA's dataset, G2P-labeled via CharsiuG2P — IPA output, would need ARPABET conversion

### GOP Posterior Format Requirement

The GOP pipeline (`gop.py`, `ctc_gop_original.py`) expects:

- Frame-level log-probabilities from a CTC output layer
- Shape: `(T, num_phones)` where `num_phones` is ARPABET vocabulary size (~40)
- No beam search or decoding — raw CTC output layer activations
- Any from-scratch model must produce this format for the pipeline to work

### Compute Estimates (rough)

| Training | Data | Estimated GPU-hours (RTX 5070 12GB) |
|---------|------|-------------------------------------|
| TIMIT TDNN-LSTM | 5h | ~1-2 hours |
| Conformer-S on LS-100h | 100h | ~8-24 hours |
| Conformer-M on LS-960h | 960h | ~3-7 days |
| Zipformer-M on LS-960h | 960h | ~3-5 days |

Note: These are rough estimates. Actual compute depends on batch size, gradient accumulation,
and whether the RTX 5070 can fit the model in 12GB VRAM. Conformer-M may need smaller batch
sizes than larger GPU configurations in the literature.

---

## 5. Open Questions

- Q1: Can Conformer-M fit in 12GB VRAM at a practical batch size?
- Q2: Is gilkeyio/librispeech-alignments the best source for ARPABET labels, or should we
  use a G2P + forced alignment pipeline?
- Q3: What is the minimum PER threshold that predicts useful GOP posteriors for scoring?
- Q4: Does icefall's k2 graph compilation work with ARPABET vs BPE vocabularies without
  significant recipe changes?
- Q5: Can we reuse ZIPA's trained Zipformer encoder with a new ARPABET head
  (hybrid approach) to avoid full from-scratch training?
