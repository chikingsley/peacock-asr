# P012 Implementation Notes

`p012-hmamba-faithful` is a clean fork of the public `hmamba` release with the audited
paper/code drift corrected in-tree instead of patching the vendor repo directly.

## Paper-faithful fixes applied

1. `deXent` now supports dataset-level pronunciation priors instead of recomputing the
   correct/mispronounced ratio from each minibatch.
2. Training saves both `best_phone_mse_model.pth` and `best_mdd_f1_model.pth`, and the
   default selection metric is `mdd_f1` for the main `best_audio_model.pth`.
3. Appendix defaults are restored in config:
   `d_conv=4`, `feat_drop=0.1`.
4. Score-conditioned utterance pooling now matches Eq. 14 more closely:
   linear score projection over `[phone, word_acc, word_stress, word_total]` without the
   extra GELU nonlinearity.
5. The broken stress-head `mapping()` path was removed. The paper describes a simple FFN
   regressor; the public code created fresh unregistered parameters inside `forward`.
6. Validation/decoding now move `mask` and `bies` to the model device. The public release
   mixed CPU and GPU tensors in evaluation paths.
7. The active runtime is the official CUDA path from `mamba-ssm` and `causal-conv1d`.
   End-to-end forward tests are skipped when CUDA is unavailable instead of carrying a local
   alternate runtime path.
8. The paper-faithful architecture is now a single code path. Alternate block and pooling
   branches were removed instead of being left as dormant configuration knobs.

## Low-risk upgrades included

1. `wandb` is optional instead of a hard import.
2. Checkpoint loading strips `module.` prefixes so single-GPU and old DataParallel
   checkpoints can both be loaded.
3. Positional embeddings are sliced to sequence length instead of assuming an exact fixed
   length at runtime.
4. `torch.compile` can be enabled on the top-level HMamba module, and checkpoint save/load
   unwraps compiled models so `_orig_mod.*` keys do not leak into artifacts.
5. The vendored selective-scan stack has been removed from the active model path. HMamba
   now uses the maintained `mamba-ssm` package directly via a small bidirectional wrapper.
6. Tooling and core runtime packages are pinned to the current releases verified during this
   cleanup pass: `torch 2.11.0`, `mamba-ssm 2.3.1`, `causal-conv1d 1.6.1`,
   `ruff 0.15.8`, `ty 0.0.26`, `pydantic 2.12.5`.
7. The config schema is strict and rejects unknown keys. Structural assumptions that used to
   live only in `forward` are now validated up front, including the minimum `raw_dim`.

## Still intentionally conservative

1. The core HMamba block structure remains close to the released code.
2. No architectural change was made beyond the mismatches found in the audit.
3. External Kaldi/MDD shell tooling was not reworked in this pass.
