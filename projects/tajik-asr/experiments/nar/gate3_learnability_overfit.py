"""Gate 3 — NAR editor learnability (overfit 100 examples) with a real training recipe.

Builds the real editor: lift the pretrained omni-LLM-300M `llama_decoder` (de-causalised via
IdentityBias -> bidirectional), `text_frontend`, `final_proj`, and `encoder_proj` (the
projector — shape (4096,1024) matches our CTC's 1024-dim encoder exactly). Freeze the body;
train only the projector + LoRA. Editor input = [encoder_proj(H) ++ embed(epsilon-interleaved
draft)]; one bidirectional pass; CTC loss over the 2N+1 edit positions (blank = epsilon = EOS)
+ copy-regulariser. If projector+LoRA can drive editor WER on these 100 rows below the CTC
draft WER, the bidirectional + CTC-loss + frozen-body + copy-bias setup *can learn* edits.

Recipe (disambiguates "edits don't beat draft = bad recipe" from "= weak conditioning"):
copy-only warmup, then a linear ramp of the CTC weight (gentle onset), AdamW with cosine LR
(after linear warmup) and gradient clipping. `--tie` ties the output head to the input
embeddings (required: residual-identity copy bias; untied -> copy is unlearnable).

  uv run python experiments/nar/gate3_learnability_overfit.py --tie --epochs 80
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

import omnilingual_asr.models.wav2vec2_llama.factory as wllf
import torch
from gate2_memory_probe import LoRA
from torch import Tensor, nn

if TYPE_CHECKING:
    import sentencepiece as spm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CKPT = ROOT / "src/tajik_omnilingual_asr/models/omniASR-LLM-300M-v2.pt"
TOKENIZER = ROOT / "src/tajik_omnilingual_asr/models/omniASR_tokenizer_written_v2.model"
CACHE = HERE / "gate3_cache_fleurs.pt"
EOS, VOCAB, DIM, ENC, NSPECIAL = 2, 10288, 4096, 1024, 1
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "output_proj", "gate_proj", "inner_proj")


def build_editor(rank: int, dtype: torch.dtype, device: str) -> tuple:
    """Lift+freeze the pretrained decoder/embeddings/head; return trainable projector + LoRA."""
    from fairseq2.models.llama import LLaMAConfig
    from fairseq2.models.transformer.attention_bias import IdentityBias

    cfg = LLaMAConfig(model_dim=DIM, max_seq_len=8192, vocab_size=VOCAB, pad_idx=1,
                      num_layers=12, num_attn_heads=8, num_key_value_heads=8,
                      ffn_inner_dim=4096, rope_theta=10000.0, dropout_p=0.0)
    orig = wllf.CausalAttentionBias
    wllf.CausalAttentionBias = IdentityBias            # bidirectional editor
    try:
        dec = wllf.OmnilingualASRLLamaFactory(cfg).create_decoder()
    finally:
        wllf.CausalAttentionBias = orig

    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["model"]
    dec_sd = {k.removeprefix("llama_decoder."): v for k, v in sd.items()
              if k.startswith("llama_decoder.")}
    missing, unexpected = dec.load_state_dict(dec_sd, strict=False)
    miss = [m for m in missing if "freqs" not in m]  # RoPE freqs are a recomputed buffer
    print(f"decoder loaded: {len(dec_sd)} tensors, missing(non-freq) {miss}, "
          f"unexpected {unexpected}", flush=True)

    text_frontend = nn.Embedding(VOCAB + NSPECIAL, DIM)
    text_frontend.weight.data = sd["text_frontend.weight"]
    final_proj = nn.Linear(DIM, VOCAB, bias=False)
    final_proj.weight.data = sd["final_proj.weight"]
    proj = nn.Linear(ENC, DIM)                          # == encoder_proj, TRAINABLE
    proj.weight.data = sd["encoder_proj.weight"]
    proj.bias.data = sd["encoder_proj.bias"]

    dec.requires_grad_(requires_grad=False)
    text_frontend.requires_grad_(requires_grad=False)
    final_proj.requires_grad_(requires_grad=False)
    for mod in (dec, text_frontend, final_proj, proj):
        mod.to(device=device, dtype=dtype)
    for m in dec.modules():                             # RoPE freqs must stay fp32
        f = getattr(m, "freqs", None)
        if isinstance(f, Tensor):
            m.freqs = f.float()

    # real LoRA on q/k/v/o + gate/inner/output of every layer (init B=0 -> starts as no-op)
    lora_params: list[nn.Parameter] = []
    for name, mod in dec.named_modules():
        w = getattr(mod, "weight", None)
        if name.split(".")[-1] in LORA_TARGETS and isinstance(w, Tensor) and w.dim() == 2:
            out_dim, in_dim = w.shape
            lora = LoRA(in_dim, out_dim, rank, dtype, device)
            lora_params += list(lora.parameters())
            mod.register_forward_hook(lambda _m, i, o, _l=lora: o + _l(i[0]))
    return dec, text_frontend, final_proj, proj, lora_params


def interleave(draft_ids: Tensor, device: str) -> Tensor:
    """epsilon-interleave a draft: (eps, x1, eps, x2, ..., xN, eps) -> length 2N+1."""
    n = draft_ids.numel()
    xt = torch.full((2 * n + 1,), EOS, dtype=torch.long, device=device)
    if n:
        xt[1::2] = draft_ids.to(device)
    return xt


def editor_forward(row: dict, dec, text_frontend, final_proj, proj,
                   dtype: torch.dtype, device: str) -> tuple[Tensor, Tensor]:
    """One bidirectional pass; return (logits over the 2N+1 edit positions, interleaved input)."""
    from fairseq2.nn import BatchLayout

    H = row["H"].to(device=device, dtype=dtype)          # [T', ENC]
    prefix = proj(H)                                     # [T', DIM]
    xt = interleave(row["draft_ids"], device)           # [2N+1]
    xt_emb = text_frontend(xt)                           # [2N+1, DIM]
    seqs = torch.cat([prefix, xt_emb], dim=0)[None]      # [1, L, DIM]
    seq_len, t0 = seqs.shape[1], prefix.shape[0]
    layout = BatchLayout((1, seq_len), seq_lens=[seq_len], device=torch.device(device))
    hid = dec(seqs.to(dtype), layout)                   # [1, L, DIM]
    logits = final_proj(hid[0, t0:])                    # [2N+1, VOCAB]
    return logits, xt


def losses(logits: Tensor, xt: Tensor, ref_ids: Tensor, device: str) -> tuple[Tensor, Tensor]:
    """Return (ctc, cr) tensors; caller combines per phase."""
    m = logits.shape[0]
    logp = logits.float().log_softmax(-1)               # [M, VOCAB]
    ctc = nn.functional.ctc_loss(
        logp[:, None, :], ref_ids.to(device)[None],
        torch.tensor([m]), torch.tensor([ref_ids.numel()]),
        blank=EOS, zero_infinity=True,
    )
    cr = nn.functional.cross_entropy(logits.float(), xt)  # copy: predict own input token
    return ctc, cr


def ctc_decode(logits: Tensor, sp: spm.SentencePieceProcessor) -> str:
    """Greedy CTC collapse over the edit positions: merge repeats, drop epsilon, detokenize."""
    ids, out, prev = logits.argmax(-1).tolist(), [], None
    for i in ids:
        if i not in (prev, EOS):
            out.append(i)
        prev = i
    return sp.decode(out)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4, help="peak LR (after linear warmup)")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--accum", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=12, help="copy-only (CR) epochs before CTC")
    ap.add_argument("--ramp", type=int, default=10, help="epochs to ramp CTC weight 0->1")
    ap.add_argument("--lam", type=float, default=0.02, help="copy-reg weight")
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip")
    ap.add_argument("--tie", action="store_true",
                    help="tie output head to input embeddings (residual-identity copy bias)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    import sentencepiece as spm
    from omni_curator.process import normalize
    from omni_finetune_core.metrics import compute_measures

    device, dtype, lang = "cuda", torch.bfloat16, "tgk_Cyrl"
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    assert sp.id_to_piece(EOS), "EOS id has no piece"
    assert EOS not in sp.encode("салом"), "EOS id leaks into normal text"

    data = torch.load(CACHE, weights_only=False)["rows"]
    train, heldout = data[: args.n_train], data[args.n_train:]

    def wer_of(rows: list[dict], hyps: list[str]) -> float:
        refs = [normalize(r["ref_text"], lang) for r in rows]
        return compute_measures(refs, [normalize(h, lang) for h in hyps]).wer

    draft_wer = wer_of(train, [r["draft_text"] for r in train])
    print(f"train rows {len(train)}  held-out {len(heldout)}  "
          f"CTC draft WER (train) = {draft_wer:.2f}%", flush=True)

    dec, tf, fp, proj, lora = build_editor(args.rank, dtype, device)
    if args.tie:  # head := input embeddings, so residual identity makes copy near-free
        fp.weight.data = tf.weight.data[:VOCAB].clone()
        print("TIED output head to input embeddings", flush=True)
    trainable = list(proj.parameters()) + lora
    print(f"trainable: projector {sum(p.numel() for p in proj.parameters()) / 1e6:.1f}M + "
          f"LoRA r{args.rank} {sum(p.numel() for p in lora) / 1e6:.1f}M = "
          f"{sum(p.numel() for p in trainable) / 1e6:.1f}M\n", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / args.accum)
    warmup_steps, total_steps = steps_per_epoch, args.epochs * steps_per_epoch

    def lr_at(step: int) -> float:  # linear warmup (1 epoch) then cosine to 0
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    @torch.no_grad()
    def eval_hyps(rows: list[dict]) -> list[str]:
        return [ctc_decode(editor_forward(r, dec, tf, fp, proj, dtype, device)[0], sp)
                for r in rows]

    print(f"  epoch 0 (untrained): train WER {wer_of(train, eval_hyps(train)):.2f}%   "
          f"(warmup={args.warmup} copy-only, ramp={args.ramp})", flush=True)
    for ep in range(1, args.epochs + 1):
        w_ctc = 0.0 if ep <= args.warmup else min(1.0, (ep - args.warmup) / max(1, args.ramp))
        tot_ctc = tot_cr = 0.0
        opt.zero_grad(set_to_none=True)
        for j, r in enumerate(train):
            logits, xt = editor_forward(r, dec, tf, fp, proj, dtype, device)
            ctc, cr = losses(logits, xt, r["ref_ids"], device)
            loss = w_ctc * ctc + args.lam * cr
            (loss / args.accum).backward()
            tot_ctc += ctc.item()
            tot_cr += cr.item()
            if (j + 1) % args.accum == 0 or (j + 1) == len(train):
                nn.utils.clip_grad_norm_(trainable, args.clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        if ep % 5 == 0 or ep in (1, args.warmup):
            hyps = eval_hyps(train)
            tw, hw = wer_of(train, hyps), (wer_of(heldout, eval_hyps(heldout))
                                          if heldout else float("nan"))
            tag = "warmup" if ep <= args.warmup else f"ctc{w_ctc:.2f}"
            print(f"  epoch {ep:3d} [{tag:>7}]  ctc {tot_ctc / len(train):.3f} "
                  f"cr {tot_cr / len(train):.3f}  lr {sched.get_last_lr()[0]:.1e}  "
                  f"train WER {tw:6.2f}%  held-out {hw:6.2f}%  | hyp0={hyps[0][:38]!r}",
                  flush=True)

    final = wer_of(train, eval_hyps(train))
    print(f"\nBASELINE CTC draft WER (train) {draft_wer:.2f}%  ->  editor train WER "
          f"{final:.2f}%   ({'BEAT draft' if final < draft_wer else 'did NOT beat draft'})",
          flush=True)
    print("sample (draft || editor || ref):", flush=True)
    for r in train[:4]:
        ed = ctc_decode(editor_forward(r, dec, tf, fp, proj, dtype, device)[0], sp)
        print(f"  {r['draft_text'][:55]!r}\n  {ed[:55]!r}\n  {r['ref_text'][:55]!r}\n",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
