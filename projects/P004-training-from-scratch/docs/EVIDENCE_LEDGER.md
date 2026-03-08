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

## 0.10.2 `C2.5a` Remote LR Stabilization Ladder (2026-03-07)

Validation ID:

- `C2.5a` isolated bounded validation on Vast `RTX PRO 4000 Blackwell`
  with lower LR only

Current status:

- `failed`
- reports written:
  `experiments/checkpoints/canonical_phone_ctc/c2_5a_remote_vast_rtxpro4000_lr1e4_20260307_a/report.json`
  and
  `experiments/checkpoints/canonical_phone_ctc/c2_5a_remote_vast_rtxpro4000_lr1e5_20260307_b/report.json`

Validated facts:

- Lowering LR from `3e-4` to `1e-4` did not change the failure signature.
- Lowering LR again to `1e-5` also did not change the failure signature.
- Both runs still failed at:
  `epoch=0`, `batch_index=1`, `train_loss=NaN`.

Implication:

- LR alone does not stabilize the nightly `flex_triton` trainer path on this
  Blackwell target.

Next action:

- Hold LR fixed at the lower value and change backend only.

---

## 0.10.3 `C2.5b` Remote Backend Comparison (2026-03-07)

Validation ID:

- `C2.5b` isolated bounded validation on Vast `RTX PRO 4000 Blackwell`
  with `attention_backend=flex_auto`

Current status:

- `failed`
- report written:
  `experiments/checkpoints/canonical_phone_ctc/c2_5b_remote_vast_rtxpro4000_flex_auto_lr1e5_20260307_a/report.json`

Validated facts:

- Switching from `flex_triton` to `flex_auto` while keeping the lowered
  `1e-5` LR did not stabilize the trainer.
- The run again failed at:
  `epoch=0`, `batch_index=1`, `train_loss=NaN`.

Implication:

- The instability is not fixed by backend choice alone between the currently
  working nightly flex backends.

Next action:

- Keep the original `flex_triton` target and narrow dtype scope for the CTC
  loss path.

---

## 0.10.4 `C2.5c` Remote Float32 CTC Loss Path (2026-03-07)

Validation ID:

- `C2.5c` isolated bounded validation on Vast `RTX PRO 4000 Blackwell`
  with `loss_compute_dtype=float32`

Current status:

- `failed`
- report written:
  `experiments/checkpoints/canonical_phone_ctc/c2_5c_remote_vast_rtxpro4000_triton_lr1e5_lossfp32_20260307_a/report.json`

Validated facts:

- Promoting the CTC log-probs / loss path to `float32` did not stabilize the
  nightly `flex_triton` trainer.
- The run still failed at:
  `epoch=0`, `batch_index=1`, `train_loss=NaN`.

Implication:

- `C2.5` is red end-to-end.
- The stable canonical path remains:
  eager execution + default SDPA path.
- The nightly flex branch is valuable for benchmark work, but not promotable
  for bounded training on this Blackwell setup yet.

Next action:

- Either open a full-fp32 / first-step finiteness diagnostic branch, or demote
  nightly flex training back to benchmark-only status until upstream behavior
  changes.
- Keep true FA4 as a separate future branch on `H100/H200/B200`-class hardware.

---

## 0.10.5 `F0` Remote H100 FA4 Benchmark Probe (2026-03-07)

Validation ID:

- `F0` remote FA4 benchmark on Vast `H100 SXM`

Current status:

- `mixed`
- benchmark report written:
  `experiments/benchmarks/canonical_phone_ctc/f0_remote_h100_fa4_benchmark_20260307_c/report.json`

Validated facts:

- The remote `H100 SXM` nightly env installed cleanly with:
  `torch 2.12.0.dev20260307+cu128` and `flash-attn-4 4.0.0b4`.
- The hardware is in the expected FA4 support range:
  compute capability `9.0`.
- Compiled nightly `flex_attention` with `AUTO` and explicit `TRITON` both
  worked on this H100 target.

Observed issues:

- The FA4-specific paths did not clear:
  direct FLASH and compiled `BACKEND=FLASH` both failed with:
  `OpError: expects the M-mode to be 64, but got 32`.
- So the benchmark was only green for the non-FLASH nightly backends, not for
  the actual FA4 backend we wanted to validate.

Implication:

- Supported hardware alone does not make the current FA4 path viable for this
  project.
- FA4 remains unproven even on the correct class of GPU.

Next action:

- Try one real `flex_flash` trainer smoke before deciding whether FA4 is even a
  semi-stable higher-cost option.

---

## 0.10.6 `F0` Remote H100 Flex-FLASH Trainer Smoke (2026-03-07)

Validation ID:

- `F0` remote trainer smoke with `attention_backend=flex_flash`

Current status:

- `failed`
- smoke report written:
  `experiments/checkpoints/canonical_phone_ctc/f0_remote_h100_flex_flash_smoke_20260307_c/report.json`

Validated facts:

- The canonical trainer now records hard backend exceptions into the run report,
  so this failure is captured as structured evidence rather than only a raw
  traceback.
- On Vast `H100 SXM`, a real one-epoch smoke with:
  `attention_backend=flex_flash`, `hidden_dim=256`, `attention_heads=4`
  failed in the trainer path.

Observed issues:

- The trainer failed with the same backend error seen in the benchmark:
  `OpError: expects the M-mode to be 64, but got 32`.

Implication:

- `F0` is red.
- FA4 is not currently a legitimate semi-stable higher-cost option for this
  project, even on supported H100 hardware.
- The only nightly path that remains practically usable is still benchmark-only
  `AUTO` / `TRITON`, not FLASH / FA4 training.

Next action:

- Keep the production path on eager + SDPA.
- If we revisit nightly work, choose either `C2.6` diagnostics on the TRITON
  branch or wait for upstream FA4 changes before spending more H100 time.

---

## 0.10.7 `P1` Promoted Stable Canonical Runner (2026-03-07)

Validation ID:

- `P1` promoted stable canonical trainer on the local `RTX 5070`

Current status:

- `passed`
- promoted bounded-validation report written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_prod_b1_20260307_a/report.json`
- W&B run:
  `peacockery/peacock-asr-p004-training-from-scratch/y3h2393n`

Validated facts:

- The stable canonical choices are now frozen in code behind a non-smoke
  entrypoint: `uv run p004-canonical-train`.
- The promoted runner keeps the stable local baseline:
  `attention_backend=mha`, `enable_compile=false`, `bf16`,
  `model_type=conformer_like`, `hidden_dim=192`, `encoder_layers=3`,
  `attention_heads=4`, `conv_kernel_size=15`, `dropout=0.1`,
  `learning_rate=3e-4`.
- A real bounded validation passed locally with online W&B sync, per-epoch
  checkpoints, machine-manifest capture, and final model-state export.
- The run completed `18` optimizer steps over `3` epochs, wrote `epoch-0.pt`
  through `epoch-2.pt`, and reported mean train loss moving
  `20.66 -> 8.83 -> 8.59`.
- The run finished with `dev_loss=15.0485`, `dev_per=0.9914`,
  `steady_state_mean_step_seconds=0.0151`, and
  `peak_memory_reserved_mb=354.0`.

Observed issues:

- The current promoted model is still `conformer_like`, not yet a literal
  full Conformer implementation.
- `dev_per` remains poor on this tiny bounded dataset, so this result validates
  the trainer and stable stack, not end-task quality.

Implication:

- `P1` is green.
- The stable production branch is now a real, named canonical trainer instead
  of only a smoke harness.
- The next production step is architectural: implement a real Conformer module
  on this frozen stack and rerun the same bounded gate before scaling up.

Next action:

- Keep nightly `TRITON` and FA4 as separate experiment branches.
- Build the real Conformer module on the promoted stable trainer.
- Rerun bounded validation through `p004-canonical-train`, then scale toward a
  full run.

---

## 0.10.8 `P1` Real Conformer On The Promoted Stable Runner (2026-03-07)

Validation ID:

- `P1` bounded validation with the real Conformer on the local `RTX 5070`

Current status:

- `passed`
- bounded-validation report written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_conformer_prod_b1_20260307_a/report.json`
- W&B run:
  `peacockery/peacock-asr-p004-training-from-scratch/r79clsmu`

Validated facts:

- The frozen production entrypoint `uv run p004-canonical-train` now defaults
  to `model_type=conformer`, not the temporary `conformer_like` block.
- The real Conformer path stayed stable on the same frozen stack:
  `attention_backend=mha`, `enable_compile=false`, `bf16`,
  `hidden_dim=192`, `encoder_layers=3`, `attention_heads=4`,
  `conv_kernel_size=15`, `dropout=0.1`, `learning_rate=3e-4`.
- The bounded validation passed with online W&B sync, `18` optimizer steps,
  `epoch-0.pt` through `epoch-2.pt`, machine-manifest capture, and final
  model-state export.
- The run reported mean train loss moving `18.31 -> 4.03 -> 4.00`,
  `dev_loss=3.9507`, `peak_memory_reserved_mb=508.0`, and
  `steady_state_mean_step_seconds=0.0214`.
- The real Conformer model has `2,593,771` parameters in this bounded local
  configuration.

Observed issues:

- `dev_per=1.0` on this tiny bounded dataset, so the result validates stable
  training behavior and stack correctness, not useful recognition quality yet.
- This run still uses the smoke-manifest subset, not the full prepared
  manifests.

Implication:

- The stable production branch now has a real Conformer, not just a placeholder
  structured encoder.
- The next production task is no longer architecture replacement. It is data
  scale-up on the same stable trainer.

Next action:

- Keep the same frozen stack and real Conformer defaults.
- Move from smoke manifests to a larger bounded subset or the full prepared
  manifests.
- Only revisit nightly `TRITON` or FA4 separately from this production path.

---

## 0.10.9 `P1` Larger Raw-Manifest Bounded Run (2026-03-07)

Validation ID:

- `P1` larger bounded validation on raw prepared manifests with the real
  Conformer on the local `RTX 5070`

Current status:

- `passed`
- bounded-validation report written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_conformer_b2_raw_20260307_a/report.json`
- W&B run:
  `peacockery/peacock-asr-p004-training-from-scratch/evi2eptj`

Validated facts:

- The stable production stack can train the real Conformer on the raw prepared
  manifests, not just the tiny smoke subset.
- A bounded run with `2048` train cuts, `256` dev cuts, `2` epochs, and
  `batch_size=4` completed successfully.
- The run completed `1024` optimizer steps, wrote `epoch-0.pt` and `epoch-1.pt`,
  and reported mean train loss moving `3.52 -> 2.47`.
- Final metrics were `dev_loss=1.8293`, `dev_per=0.6021`,
  `peak_memory_reserved_mb=588.0`, and
  `steady_state_mean_step_seconds=0.0226`.

Observed issues:

- The current trainer still eagerly materializes features for all selected cuts
  before W&B init and the first optimizer step.
- On this run, that front-loaded preprocessing consumed several minutes of wall
  time before actual training began, even though the recorded training window
  itself was only about `55` seconds.

Implication:

- The production path is now validated beyond the smoke subset.
- The next blocker to a full run is no longer model stability. It is the
  current data-loader and batching design.

Next action:

- Keep the same stable Conformer defaults.
- Replace eager feature materialization with lazy loading and more appropriate
  batching on the same trainer.
- After that, run the next larger bounded slice or the full prepared manifests.

---

## 0.10.10 `P1` Lazy Loader And Bounded Raw-Manifest Re-Run (2026-03-07)

Validation ID:

- `P1` stable production stack with the lazy manifest-backed dataset and
  duration-aware batching on the local `RTX 5070`

Current status:

- `passed`
- bounded-validation report written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_conformer_b2_raw_lazy_20260307_a/report.json`
- W&B run:
  `peacockery/peacock-asr-p004-training-from-scratch/q9oem355`

Validated facts:

- The stable production stack now starts training from the prepared manifests
  without the old front-loaded feature materialization delay.
- W&B initialized immediately at run start instead of waiting for a long
  preprocessing phase.
- The same `2048` train-cut / `256` dev-cut bounded run completed cleanly with
  the lazy loader and duration-aware batching.
- Final metrics improved to `dev_loss=1.7656`, `dev_per=0.5697`,
  `peak_memory_reserved_mb=592.0`, and
  `steady_state_mean_step_seconds=0.0231`.

Implication:

- The data path is now compatible with real prepared-manifest training on the
  frozen production stack.
- The production lane can move from bounded slices to full manifests without a
  separate loader rewrite branch.

Next action:

- Keep the same frozen stack and real Conformer defaults.
- Run the full prepared manifests on `train-clean-100` / `dev-clean`.

---

## 0.10.11 `P1` Full Prepared-Manifest Train-Clean-100 Run And Resume (2026-03-07)

Validation ID:

- `P1` full prepared-manifest Conformer training on the local `RTX 5070`

Current status:

- `passed`
- full-epoch report written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_conformer_full_trainclean100_e1_20260307_a/report.json`
- resumed full-run report written:
  `experiments/checkpoints/canonical_phone_ctc/canonical_local_conformer_full_trainclean100_e3_resume_20260307_a/report.json`
- W&B runs:
  `peacockery/peacock-asr-p004-training-from-scratch/195j4c71`
  and `peacockery/peacock-asr-p004-training-from-scratch/yfupd8rl`

Validated facts:

- The stable production stack trained on all `28,538` cuts in
  `train-clean-100` and evaluated on all `2,703` cuts in `dev-clean`.
- The first full epoch completed cleanly with `7135` optimizer steps,
  `dev_loss=0.9180`, `dev_per=0.2697`, `peak_memory_reserved_mb=820.0`, and
  `steps_per_second=41.60`.
- Resume from `epoch-0.pt` also worked on the full manifests: the follow-on run
  loaded the checkpoint, completed epochs `1` and `2`, and finished with
  `dev_loss=0.6969`, `dev_per=0.1982`, `mean_train_loss=0.7906 -> 0.6911`,
  and `steps_per_second=38.53`.

Implication:

- The production lane is now proven on a real full `train-clean-100` Conformer
  run, not just smoke subsets or bounded slices.
- The stable Blackwell-compatible stack is ready for actual scaling work.

Next action:

- Keep the same frozen stack and real Conformer defaults.
- Choose the next scale branch explicitly: bigger data coverage, a larger
  Conformer, or stronger decoding / eval on the same production lane.

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
