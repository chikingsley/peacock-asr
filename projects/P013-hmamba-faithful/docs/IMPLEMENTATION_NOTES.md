# P013 Implementation Notes

`P013-hmamba-faithful` is the cleaned HMamba branch for reproducing the public
`P014-hmamba-original` release without editing the vendor repo in place.

## Current status

- The main completed rerun is `runs/repro-20260403-094834`.
- That rerun executed the official 5-seed list from the original repo: `824`,
  `17`, `2413`, `168`, `623`.
- Trackio is now the active experiment logger for this branch via the local
  project `p013-hmamba-faithful`.
- The 5-seed rerun is complete and all five seeds are present on disk and in
  Trackio.

## Paper-faithful fixes applied

1. `deXent` now supports dataset-level pronunciation priors instead of
   recomputing the correct/mispronounced ratio from each minibatch.
2. Training saves `best_phone_mse_model.pth`, `best_mdd_f1_model.pth`, and
   `best_audio_model.pth`.
3. The default checkpoint selection metric is now `phone_mse`, matching the
   original `P014-hmamba-original/traintest.py` behavior where
   `best_audio_model.pth` tracks the best phone MSE.
4. Appendix defaults are restored in config: `d_conv=4`, `feat_drop=0.1`.
5. Score-conditioned utterance pooling matches Eq. 14 more closely: linear
   score projection over `[phone, word_acc, word_stress, word_total]` without
   the extra GELU nonlinearity.
6. The broken stress-head `mapping()` path was removed. The paper describes a
   simple FFN regressor; the public code created fresh unregistered parameters
   inside `forward`.
7. Validation and decoding move `mask` and `bies` to the model device. The
   public release mixed CPU and GPU tensors in evaluation paths.
8. The active runtime is the official CUDA path from `mamba-ssm` and
   `causal-conv1d`.
9. Recognition defaults to `best_audio_model.pth`, which is the checkpoint the
   original stage-2 `recog.py` path uses after stage-1 training.

## Evaluation-path correction

The biggest drift uncovered during this pass was MDD evaluation:

1. The inline `mdd_test_f1` written into `result.csv` is a batch-averaged proxy
   computed during validation.
2. The original repo reports MDD after stage-2 transcript generation and
   stage-3 corpus-level alignment/evaluation.
3. `P013` now includes `p012-mdd-eval`, a Python port of the original
   `eval_mdd/utils/ins_del_sub_cor_analysis.py` counting logic, so post-train
   MDD can be computed directly from `rel_nosil`, `can_nosil`, and `hyp_nosil`.
4. Exact stage-3 parity is now also verified locally through the original
   `P014-hmamba-original/eval_mdd/mdd_result.sh` shell path with Kaldi
   `align-text` and `compute-wer` built under `third_party/kaldi`.
5. The exact Kaldi stage-3 outputs are written per seed as
   `mdd_result_kaldi_raw.txt` and `mdd_result_kaldi.txt`.

This means the inline `mdd_test_f1` should be treated as a training-side proxy,
not the final paper-comparable MDD score.

## Trackio

Trackio is now wired as the default logger for `P013`:

1. Training initializes a local Trackio run under project
   `p013-hmamba-faithful`.
2. Each experiment writes `run_config.json` into its `exp_dir`.
3. The finished official-style rerun is visible in Trackio as:
   `seed824`, `seed17`, `seed2413`, `seed168`, `seed623`.

## Official-style rerun outcome

Reference run directory:

- `runs/repro-20260403-094834`

5-seed means from that rerun:

- Phone PCC: `0.7153`
- Phone MSE: `0.0664`
- Utterance total PCC: `0.8083`
- Word total PCC: `0.6991`
- PER: `0.02727`
- Inline validation MDD F1 proxy: `0.5288`
- Corpus-level post-recog MDD precision: `0.6791`
- Corpus-level post-recog MDD recall: `0.5103`
- Corpus-level post-recog MDD F1: `0.5818`

Best individual seeds:

- Best phone PCC: seed `17` with `0.7215`
- Best corpus-level MDD F1: seed `168` with `0.6109`

Comparison to the paper:

1. APA is in the right neighborhood, but still below the paper's reported
   HMamba averages.
2. PER is effectively on target.
3. MDD remains the clearest unresolved gap. The paper reports HMamba
   precision/recall/F1 of `0.6435 / 0.6341 / 0.6385`, while the completed P013
   rerun reached `0.6791 / 0.5103 / 0.5818` on corpus-level post-recog
   evaluation.

## Kaldi stage-3 parity

The prior evaluation-tooling caveat is now closed:

1. The original repo expects compiled Kaldi tools through `path.sh`.
2. `P013` now carries a repo-local Kaldi checkout under `third_party/kaldi`
   pinned to the original commit `d6198906fbb0e3cfa5fae313c7126a78d8321801`.
3. Running the original `eval_mdd/mdd_result.sh` with those Kaldi binaries on
   the finished 5-seed rerun reproduces the current corpus-level MDD values to
   rounding error across every seed.
4. The reproducible entrypoint for that exact stage-3 path is
   `tools/run_kaldi_mdd_eval.sh`.

The remaining MDD shortfall is therefore a model/result gap, not an evaluation
parity gap.

## Local toolchain notes

The Kaldi stage-3 path required local compatibility fixes for the current Arch
toolchain:

1. OpenFST 1.6.7 needed a local source fix in
   `third_party/kaldi/tools/openfst-1.6.7/src/include/fst/bi-table.h` for GCC
   15.
2. Kaldi's 2020-era OpenBLAS wrapper expected the older LAPACK header calling
   convention, so a local vendored `lapack.h`/`lapacke.h` pair was placed under
   `third_party/kaldi/src/matrix` to disable `LAPACK_FORTRAN_STRLEN_END`.
3. These are toolchain-only fixes. No HMamba model or evaluation logic was
   changed in `P013` to obtain the final MDD numbers above.

## Still intentionally conservative

1. The core HMamba block structure remains close to the released code.
2. No architectural change was made beyond the mismatches found in the audit.
3. The current branch prefers explicit reproducibility artifacts
   (`run_config.json`, per-seed directories, Trackio logs) over silent
   in-place overwrite behavior.
