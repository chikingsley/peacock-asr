# Extend-tokenizer + restore-decoder/joint recipe for v3 (ready-to-implement fallback)

Status: **designed + codex 5.5 xhigh APPROVED-TO-IMPLEMENT (2 rounds, 2026-06-16). NOT built/run.**
Deploy ONLY if the simple v3 run (fresh BPE-1024 + reinit decoder/joint + Adafactor) plateaus high.
Community recipe provenance: NeMo discussion #14728 (see README §problem+recipe).

## Why
The simple recipe keeps v3's encoder (Cyrillic prior) but **discards v3's decoder/joint** (its token
prior). This recipe preserves the decoder/joint by extending the tokenizer (keep v3 IDs 0..8191, append
Tajik) and restoring the pretrained weight rows — only the new Tajik rows stay random.

## Spec (codex-converged)
1. **Reduction fix (applies to BOTH recipes):** rebuild `RNNTLoss(..., reduction=model.cfg.get("rnnt_reduction","mean_batch"))`.
   `extract_rnnt_loss_cfg` drops `rnnt_reduction`; v3 cfg = `mean_volume`, RNNTLoss defaults `mean_batch`.
   **Caveat (ours):** changing reduction changes loss scale → lr must be co-tuned; treat as an ablation,
   not a blind flip (the simple 110M run used default `mean_batch` + lr 3e-4 → 19%, so it's validated, not broken).
2. **Tokenizer builder** (`tokenizer_extend.py`): read v3 `tokenizer.model`; train a candidate Tajik BPE on
   omni-normalized text; select only pieces containing >=1 of `ғ қ ҳ ҷ ӯ ӣ Ғ Қ Ҳ Ҷ Ӯ Ӣ`; always include those
   12 singletons; skip dups + shared-Cyrillic-only pieces; target ~512 by rank (never pad with shared-only);
   append to v3's `ModelProto.pieces` (old pieces untouched), set `vocab_size=8192+K`; emit standard NeMo dir.
   Assert IDs 0..8191 byte-identical.
3. **`--recipe extend-restore`** in `tdt.py`: snapshot decoder+joint → `change_vocabulary(bpe, extended_dir)`
   → restore. Mapping (K = appended pieces, extra = num_extra_outputs = 5, old_blank=8192, new_blank=8192+K):
   - decoder embed: `new[:8192]=old[:8192]`; `new[new_blank]=old[8192]`; rows `8192:8192+K` random.
   - dec_rnn, joint.pred, joint.enc: copy wholesale (same shape).
   - joint final Linear: `new_out[:8192]=old_out[:8192]`; `new_out[new_blank:new_blank+1+extra]=old_out[8192:8192+1+extra]`; new rows random.
4. **`--freeze-warmup-steps N`**: don't pre-freeze; Lightning callback freezes encoder `on_fit_start`,
   unfreezes at `global_step>=N` (all if `--unfreeze-top 0`, else top-N). Separate from permanent `--freeze-encoder`.

## Dip-a-toe gates (ALL must pass before any long run)
- **Tokenizer gate:** IDs 0..8191 identical old/new; 12 singletons encode w/o `<unk>`; Tajik train/dev
  round-trip; a shared-Cyrillic sample yields identical old/new ID sequence.
- **Shape gate:** `num_classes_with_blank == 8198+K`; blank `8192+K`; durations `8193+K..8197+K`.
- **Weight gate:** copied tensors `torch.equal`; new Tajik rows == post-`change_vocabulary` random init.
- **Logit-preservation gate:** fp32 fixed input; old-token + remapped blank/duration logits match base v3
  (`max_abs_diff==0` for direct final-linear rows, `<=1e-6` full module path).
- **Overfit-32 gate:** fp32 tiny run; loss collapses + >=5/6 fixed clips decode exact (WER alone insufficient).

## Worth-it bar
Skip entirely if the simple v3 run lands near/below omni 16.9. Extend-restore adds tokenizer surgery +
row-mapping risk + a larger output layer, and still leaves all *truly Tajik-specific* rows random.
