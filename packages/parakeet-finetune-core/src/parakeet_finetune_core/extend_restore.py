"""Restore pretrained decoder/joint rows after extending a TDT tokenizer.

The ``extend-restore`` recipe (see ``extend_restore_recipe.md``) keeps base tokenizer IDs
0..old_vocab-1 and appends K target-language pieces. ``change_vocabulary`` then rebuilds the
decoder embedding and joint output layer at the new size with *random* rows. This module
snapshots the pretrained decoder/joint before ``change_vocabulary`` and copies the matching
rows back afterwards, so only the truly new (target-language) rows stay random.

v3 layout (EncDecRNNTBPEModel, verified on CPU):
  decoder.prediction.embed.weight : (old_vocab + 1, 640)  -- row old_vocab is the blank embed
  decoder.prediction.dec_rnn.*    : LSTM (shape-invariant to vocab)
  joint.pred / joint.enc          : Linear (shape-invariant to vocab)
  joint.joint_net[2].weight/bias  : (old_vocab + 1 + extra, 640) -- final output Linear
     rows 0..old_vocab-1 = tokens, row old_vocab = blank, rows old_vocab+1.. = durations.

Mapping (K appended, extra = num_extra_outputs, old_blank = old_vocab, new_blank = old_vocab+K):
  embed:  new[:old_vocab] = old[:old_vocab]; new[new_blank] = old[old_blank]; tajik rows random.
  dec_rnn, joint.pred, joint.enc: copy wholesale (same shape).
  joint final Linear: new[:old_vocab] = old[:old_vocab];
     new[new_blank : new_blank + 1 + extra] = old[old_blank : old_blank + 1 + extra]; tajik random.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch  # ty: ignore[unresolved-import]

if TYPE_CHECKING:
    from collections.abc import Sequence

# full-module-path logit tolerance for the logit-preservation gate
_PATH_TOL = 1e-6

_DEC_EMBED = "prediction.embed.weight"
_DEC_RNN_PREFIX = "prediction.dec_rnn."
_JOINT_FINAL_W = "joint_net.2.weight"
_JOINT_FINAL_B = "joint_net.2.bias"


def _decoder(model: Any) -> Any:
    return model.decoder


def _joint(model: Any) -> Any:
    return model.joint


def snapshot_decoder_joint(model: Any) -> dict[str, torch.Tensor]:
    """Clone (detached, CPU fp32) every decoder + joint parameter prior to vocab change."""
    snap: dict[str, torch.Tensor] = {}
    for name, param in _decoder(model).named_parameters():
        snap[f"decoder.{name}"] = param.detach().to(dtype=torch.float32, device="cpu").clone()
    for name, param in _joint(model).named_parameters():
        snap[f"joint.{name}"] = param.detach().to(dtype=torch.float32, device="cpu").clone()
    return snap


def _copy_into(dst: torch.Tensor, src: torch.Tensor) -> None:
    with torch.no_grad():
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device))


def restore_extended_decoder_joint(
    model: Any,
    snapshot: dict[str, torch.Tensor],
    *,
    old_vocab: int = 8192,
) -> dict[str, Any]:
    """Copy pretrained rows back into the (re-sized) decoder/joint. Tajik rows stay random.

    Must be called *after* ``model.change_vocabulary(...)``. Returns a summary dict.
    """
    joint = _joint(model)
    extra = int(joint.num_extra_outputs)
    new_vocab_with_blank = int(joint.num_classes_with_blank) - extra  # tokens + blank
    new_blank = new_vocab_with_blank - 1
    old_blank = old_vocab
    k = new_blank - old_blank
    if k < 0:
        msg = f"new_blank {new_blank} < old_blank {old_blank}: vocab did not grow"
        raise ValueError(msg)

    dec_params = dict(_decoder(model).named_parameters())
    joint_params = dict(joint.named_parameters())

    # --- decoder embedding: tokens 0..old_vocab-1 + remapped blank row ---
    embed = dec_params[_DEC_EMBED]
    old_embed = snapshot[f"decoder.{_DEC_EMBED}"]
    with torch.no_grad():
        embed[:old_vocab].copy_(old_embed[:old_vocab].to(embed.dtype))
        embed[new_blank].copy_(old_embed[old_blank].to(embed.dtype))

    # --- decoder dec_rnn: shape-invariant, copy wholesale ---
    for name, param in dec_params.items():
        if name.startswith(_DEC_RNN_PREFIX):
            _copy_into(param, snapshot[f"decoder.{name}"])

    # --- joint.pred / joint.enc: shape-invariant, copy wholesale ---
    for name, param in joint_params.items():
        if name.startswith(("pred.", "enc.")):
            _copy_into(param, snapshot[f"joint.{name}"])

    # --- joint final Linear: tokens + remapped blank/duration block ---
    for tensor_name in (_JOINT_FINAL_W, _JOINT_FINAL_B):
        new_t = joint_params[tensor_name]
        old_t = snapshot[f"joint.{tensor_name}"]
        with torch.no_grad():
            new_t[:old_vocab].copy_(old_t[:old_vocab].to(new_t.dtype))
            block = 1 + extra  # blank + durations
            new_t[new_blank : new_blank + block].copy_(
                old_t[old_blank : old_blank + block].to(new_t.dtype)
            )

    return {
        "old_vocab": old_vocab,
        "k": k,
        "new_blank": new_blank,
        "extra": extra,
        "num_classes_with_blank": int(joint.num_classes_with_blank),
    }


# --------------------------------------------------------------------------- #
# Freeze-warmup Lightning callback
# --------------------------------------------------------------------------- #


def make_freeze_warmup_callback(steps: int, unfreeze_top: int = 0) -> Any:
    """Build a Lightning ``Callback`` that freezes the encoder then unfreezes after warmup.

    Freezes the encoder ``on_fit_start``; at ``global_step >= steps`` unfreezes all encoder
    params (``unfreeze_top == 0``) or only the top-N layers. Lazily subclasses
    ``pl.Callback`` so importing this module does not require Lightning.
    """
    import lightning.pytorch as pl  # ty: ignore[unresolved-import]

    class FreezeWarmup(pl.Callback):
        def __init__(self, warmup_steps: int, top: int) -> None:
            super().__init__()
            self.warmup_steps = warmup_steps
            self.top = top
            self._unfrozen = False

        def on_fit_start(self, _trainer: Any, pl_module: Any) -> None:
            pl_module.encoder.freeze()
            self._unfrozen = False
            print(f"[freeze-warmup] encoder frozen until step {self.warmup_steps}", flush=True)

        def on_train_batch_start(
            self, trainer: Any, pl_module: Any, *_args: object
        ) -> None:
            if self._unfrozen or trainer.global_step < self.warmup_steps:
                return
            self._unfreeze(pl_module)
            self._unfrozen = True

        def _unfreeze(self, pl_module: Any) -> None:
            encoder = pl_module.encoder
            if self.top <= 0:
                encoder.unfreeze()
                msg = "all encoder layers"
            else:
                for layer in encoder.layers[-self.top :]:
                    for parameter in layer.parameters():
                        parameter.requires_grad_(requires_grad=True)
                    layer.train()
                msg = f"top {self.top}/{len(encoder.layers)} encoder layers"
            trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            print(
                f"[freeze-warmup] unfroze {msg} ({trainable / 1e6:.0f}M trainable)",
                flush=True,
            )

    return FreezeWarmup(steps, unfreeze_top)


# --------------------------------------------------------------------------- #
# Logit-preservation gate
# --------------------------------------------------------------------------- #


def _fixed_joint_input(joint: Any, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic (encoder, decoder) hidden states for the joint forward."""
    gen = torch.Generator().manual_seed(seed)
    enc_dim = joint.enc.in_features
    pred_dim = joint.pred.in_features
    f = torch.randn(1, 4, enc_dim, generator=gen, dtype=torch.float32)  # (B, T, D_enc)
    g = torch.randn(1, 3, pred_dim, generator=gen, dtype=torch.float32)  # (B, U, D_pred)
    return f, g


def _joint_logits(joint: Any, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    joint = joint.float().eval()
    with torch.no_grad():
        return joint.joint(f, g)  # (B, T, U, num_classes_with_blank)


def validate_restore(
    restored_model: Any,
    base_model: Any,
    *,
    old_vocab: int = 8192,
) -> dict[str, Any]:
    """Logit-preservation gate (CPU fp32).

    Compares the restored model's joint logits on a fixed input against the base v3 model:
      - direct final-Linear rows (old tokens + remapped blank/duration) must match exactly
        (``max_abs_diff == 0``);
      - the full joint module path on those rows must match to <= 1e-6.
    Returns a dict of measured diffs and PASS booleans.
    """
    base_joint = _joint(base_model)
    new_joint = _joint(restored_model)
    extra = int(new_joint.num_extra_outputs)
    new_blank = int(new_joint.num_classes_with_blank) - extra - 1
    block = 1 + extra  # blank + durations

    # --- (a) direct final-Linear weight/bias rows: exact equality ---
    base_w = base_joint.joint_net[2].weight.detach().float().cpu()
    new_w = new_joint.joint_net[2].weight.detach().float().cpu()
    base_b = base_joint.joint_net[2].bias.detach().float().cpu()
    new_b = new_joint.joint_net[2].bias.detach().float().cpu()

    def _rows(t: torch.Tensor, blank: int) -> torch.Tensor:
        return torch.cat([t[:old_vocab], t[blank : blank + block]], dim=0)

    w_diff = float((_rows(base_w, old_vocab) - _rows(new_w, new_blank)).abs().max())
    b_diff = float((_rows(base_b, old_vocab) - _rows(new_b, new_blank)).abs().max())

    # --- (b) full module path on a fixed input ---
    f, g = _fixed_joint_input(new_joint)
    base_logits = _joint_logits(base_joint, f, g)  # (.., old_classes)
    new_logits = _joint_logits(new_joint, f, g)  # (.., new_classes)

    base_sel = torch.cat(
        [base_logits[..., :old_vocab], base_logits[..., old_vocab : old_vocab + block]], dim=-1
    )
    new_sel = torch.cat(
        [new_logits[..., :old_vocab], new_logits[..., new_blank : new_blank + block]], dim=-1
    )
    path_diff = float((base_sel - new_sel).abs().max())

    direct_pass = w_diff == 0.0 and b_diff == 0.0
    path_pass = path_diff <= _PATH_TOL
    return {
        "weight_max_abs_diff": w_diff,
        "bias_max_abs_diff": b_diff,
        "path_max_abs_diff": path_diff,
        "direct_pass": direct_pass,
        "path_pass": path_pass,
        "pass": direct_pass and path_pass,
    }


def new_tajik_rows(model: Any, *, old_vocab: int = 8192) -> dict[str, torch.Tensor]:
    """Return the random (post-change_vocabulary) Tajik rows for the weight gate."""
    joint = _joint(model)
    extra = int(joint.num_extra_outputs)
    new_blank = int(joint.num_classes_with_blank) - extra - 1
    embed = dict(_decoder(model).named_parameters())[_DEC_EMBED]
    final_w = dict(joint.named_parameters())[_JOINT_FINAL_W]
    return {
        "embed_tajik": embed[old_vocab:new_blank].detach().float().cpu().clone(),
        "joint_tajik": final_w[old_vocab:new_blank].detach().float().cpu().clone(),
    }


def assert_shapes(model: Any, *, old_vocab: int = 8192, k: int, extra: int = 5) -> None:
    """Shape gate: assert the resized joint has the expected TDT layout."""
    joint = _joint(model)
    expected_classes = old_vocab + k + 1 + extra
    if int(joint.num_classes_with_blank) != expected_classes:
        msg = (
            f"num_classes_with_blank {joint.num_classes_with_blank} != "
            f"expected {expected_classes}"
        )
        raise AssertionError(msg)
    if int(joint.num_extra_outputs) != extra:
        msg = f"num_extra_outputs {joint.num_extra_outputs} != {extra}"
        raise AssertionError(msg)


def expected_blank_and_durations(
    old_vocab: int, k: int, extra: int = 5
) -> tuple[int, Sequence[int]]:
    blank = old_vocab + k
    durations = tuple(range(blank + 1, blank + 1 + extra))
    return blank, durations
