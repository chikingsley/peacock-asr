"""Gate 4 — richer acoustic conditioning (IBM windowed Q-Former projector).

Gate 3b verdict: the single Linear projector over the final CTC layer is too weak — pure
CTC degenerates because the audio can't drive corrections. This swaps in IBM's actual NLE
projector (ported from `ibm-granite/granite-speech-4.1-2b-nar/modeling_granite_speech_nar.py`,
Apache-2.0): per-layer LayerNorm over 4 stacked encoder layers -> Linear -> window into
blocks -> mean-pool downsample -> cross-attention Q-Former with learned queries -> Linear to
LLM dim. Everything else (lifted+tied decoder, LoRA, ε-interleave, CTC+copy loss, recipe) is
reused from gate 3. (Deliberate deviation: IBM's BPE-posterior pooling is dropped — it's tied
to their BPE tokenizer; the Q-Former block-downsampling handles length reduction here.)

Gate: does richer conditioning let edits beat the 18.84% draft WER (where gate 3 couldn't)?

  uv run python experiments/nar/gate4_conditioning.py --tie --epochs 80
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from gate3_learnability_overfit import (
    EOS,
    VOCAB,
    build_editor,
    ctc_decode,
    interleave,
    losses,
)
from torch import Tensor, nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
CACHE = HERE / "gate4_cache_fleurs.pt"
TOKENIZER = HERE.parents[1] / "src/tajik_omnilingual_asr/models/omniASR_tokenizer_written_v2.model"


@dataclass
class ProjCfg:
    """IBM GraniteSpeechNarProjectorConfig defaults, with llm_dim retargeted to our 4096."""

    encoder_dim: int = 1024
    num_encoder_layers: int = 4
    hidden_size: int = 2048
    num_heads: int = 32
    num_layers: int = 1          # Q-Former depth (IBM base NLE = 1; NLE++ = 2)
    mlp_ratio: int = 2
    block_size: int = 15
    downsample_rate: int = 5
    llm_dim: int = 4096
    dropout_prob: float = 0.1
    layernorm_eps: float = 1e-6
    attn_bias: bool = True
    mlp_bias: bool = True


# ---- ported from modeling_granite_speech_nar.py (Apache-2.0) ----


class QFormerCrossAttention(nn.Module):
    def __init__(self, c: ProjCfg) -> None:
        super().__init__()
        self.num_heads, self.hidden_size = c.num_heads, c.hidden_size
        self.head_dim = c.hidden_size // c.num_heads
        self.q_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.attn_bias)
        self.k_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.attn_bias)
        self.v_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.attn_bias)
        self.o_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.attn_bias)

    def forward(self, x: Tensor, enc: Tensor) -> Tensor:
        b, q, _ = x.shape
        klen = enc.shape[1]
        qs = self.q_proj(x).view(b, q, self.num_heads, self.head_dim).transpose(1, 2)
        ks = self.k_proj(enc).view(b, klen, self.num_heads, self.head_dim).transpose(1, 2)
        vs = self.v_proj(enc).view(b, klen, self.num_heads, self.head_dim).transpose(1, 2)
        o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=False)
        return self.o_proj(o.transpose(1, 2).contiguous().view(b, q, self.hidden_size))


class QFormerMLP(nn.Module):
    def __init__(self, c: ProjCfg) -> None:
        super().__init__()
        h = int(c.hidden_size * c.mlp_ratio)
        self.fc1 = nn.Linear(c.hidden_size, h, bias=c.mlp_bias)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(h, c.hidden_size, bias=c.mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class QFormerLayer(nn.Module):
    def __init__(self, c: ProjCfg) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(c.hidden_size, eps=c.layernorm_eps)
        self.cross_attention = QFormerCrossAttention(c)
        self.mlp_norm = nn.LayerNorm(c.hidden_size, eps=c.layernorm_eps)
        self.mlp = QFormerMLP(c)

    def forward(self, x: Tensor, enc: Tensor) -> Tensor:
        x = x + self.cross_attention(self.attn_norm(x), enc)
        return x + self.mlp(self.mlp_norm(x))


class QFormerProjector(nn.Module):
    """Windowed Q-Former mapping [B, T, num_layers*enc_dim] -> [B, T*query_len/block, llm_dim]."""

    def __init__(self, c: ProjCfg) -> None:
        super().__init__()
        self.c = c
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(c.encoder_dim, eps=c.layernorm_eps) for _ in range(c.num_encoder_layers)])
        self.layer_projector = nn.Linear(c.encoder_dim * c.num_encoder_layers, c.hidden_size)
        self.dropout = nn.Dropout(c.dropout_prob)
        self.projector_act = nn.GELU()
        self.layers = nn.ModuleList([QFormerLayer(c) for _ in range(c.num_layers)])
        qlen = c.block_size // c.downsample_rate
        std = c.hidden_size**-0.5
        self.query = nn.Parameter(torch.randn(1, qlen, c.hidden_size) * std)
        self.window_positions = nn.Parameter(torch.randn(1, c.block_size, c.hidden_size) * std)
        self.out_norm = nn.LayerNorm(c.hidden_size, eps=c.layernorm_eps)
        self.out_linear = nn.Linear(c.hidden_size, c.llm_dim)

    def forward(self, hidden: Tensor) -> Tensor:
        c = self.c
        b, seq_len, _ = hidden.shape
        hidden = hidden.view(b, seq_len, c.num_encoder_layers, c.encoder_dim)
        hidden = torch.cat([ln(hidden[:, :, i]) for i, ln in enumerate(self.layer_norms)], dim=-1)
        hidden = self.projector_act(self.layer_projector(hidden))

        nblocks, rest = seq_len // c.block_size, seq_len % c.block_size
        if rest > 0:
            hidden = F.pad(hidden, (0, 0, 0, c.block_size - rest))
            nblocks += 1
        hidden = hidden.view(b * nblocks, c.block_size, c.hidden_size)
        qlen = self.query.shape[1]
        mean_pool = hidden.view(b * nblocks, qlen, c.downsample_rate, c.hidden_size).mean(dim=-2)

        x = self.dropout(self.query + mean_pool)
        enc = self.dropout(hidden + self.window_positions)
        for layer in self.layers:
            x = layer(x, enc)
        x = x.view(b, nblocks * qlen, c.hidden_size)
        return self.out_linear(self.dropout(self.out_norm(x)))


# ---- editor forward (gate3 harness + the Q-Former prefix) ----


def editor_forward(row: dict, dec, tf, fp, projector,
                   dtype: torch.dtype, device: str) -> tuple[Tensor, Tensor]:
    from fairseq2.nn import BatchLayout

    h = row["H"].to(device=device, dtype=dtype)              # [T', num_layers, enc_dim]
    prefix = projector(h.reshape(1, h.shape[0], -1))[0]      # [T_out, DIM]
    xt = interleave(row["draft_ids"], device)
    seqs = torch.cat([prefix, tf(xt)], dim=0)[None]          # [1, L, DIM]
    t0, seq_len = prefix.shape[0], prefix.shape[0] + xt.shape[0]
    layout = BatchLayout((1, seq_len), seq_lens=[seq_len], device=torch.device(device))
    hid = dec(seqs.to(dtype), layout)
    return fp(hid[0, t0:]), xt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--accum", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--ramp", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--qformer-layers", type=int, default=1)
    ap.add_argument("--downsample", type=int, default=5, help="acoustic downsample (IBM=5)")
    ap.add_argument("--block-size", type=int, default=15, help="Q-Former window (block %% ds == 0)")
    ap.add_argument("--tie", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    import sentencepiece as spm
    from omni_curator.process import normalize
    from omni_finetune_core.metrics import compute_measures

    device, dtype, lang = "cuda", torch.bfloat16, "tgk_Cyrl"
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    data = torch.load(CACHE, weights_only=False)["rows"]
    train, heldout = data[: args.n_train], data[args.n_train:]
    no_eos = all(EOS not in r["ref_ids"].tolist() for r in train + heldout)
    assert no_eos, "a ref contains the EOS/blank id"
    infeasible = sum(len(r["ref_ids"]) > 2 * len(r["draft_ids"]) + 1 for r in train)
    print(f"CTC-infeasible train rows (ref_len > 2N+1): {infeasible}/{len(train)}", flush=True)
    assert infeasible == 0, "infeasible rows get silently dropped by zero_infinity — filter first"

    def wer_of(rows: list[dict], hyps: list[str]) -> float:
        refs = [normalize(r["ref_text"], lang) for r in rows]
        return compute_measures(refs, [normalize(h, lang) for h in hyps]).wer

    draft_wer = wer_of(train, [r["draft_text"] for r in train])
    print(f"train {len(train)} held-out {len(heldout)}  draft WER {draft_wer:.2f}%", flush=True)

    dec, tf, fp, _linear_proj, lora = build_editor(args.rank, dtype, device)
    if args.tie:
        fp.weight.data = tf.weight.data[:VOCAB].clone()
        print("TIED output head to input embeddings", flush=True)
    assert args.block_size % args.downsample == 0, "block_size must be divisible by downsample"
    cfg = ProjCfg(num_layers=args.qformer_layers, downsample_rate=args.downsample,
                  block_size=args.block_size)
    projector = QFormerProjector(cfg).to(device=device, dtype=dtype)
    trainable = list(projector.parameters()) + lora
    print(f"trainable: Q-Former {sum(p.numel() for p in projector.parameters()) / 1e6:.1f}M "
          f"(layers={cfg.num_layers}, ds={cfg.downsample_rate}) + LoRA "
          f"{sum(p.numel() for p in lora) / 1e6:.1f}M\n", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    spe = math.ceil(len(train) / args.accum)
    warm_steps, total = spe, args.epochs * spe

    def lr_at(step: int) -> float:
        if step < warm_steps:
            return step / max(1, warm_steps)
        return 0.5 * (1 + math.cos(math.pi * (step - warm_steps) / max(1, total - warm_steps)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    @torch.no_grad()
    def eval_hyps(rows: list[dict]) -> list[str]:
        return [ctc_decode(editor_forward(r, dec, tf, fp, projector, dtype, device)[0], sp)
                for r in rows]

    print(f"  epoch 0 (untrained): train WER {wer_of(train, eval_hyps(train)):.2f}%", flush=True)
    for ep in range(1, args.epochs + 1):
        w_ctc = 0.0 if ep <= args.warmup else min(1.0, (ep - args.warmup) / max(1, args.ramp))
        lam_eff = args.lam * (1.0 - w_ctc)
        tot_ctc = tot_cr = 0.0
        opt.zero_grad(set_to_none=True)
        for j, r in enumerate(train):
            logits, xt = editor_forward(r, dec, tf, fp, projector, dtype, device)
            ctc, cr = losses(logits, xt, r["ref_ids"], device)
            loss = cr if ep <= args.warmup else w_ctc * ctc + lam_eff * cr
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
            tw = wer_of(train, hyps)
            hw = wer_of(heldout, eval_hyps(heldout)) if heldout else float("nan")
            tag = "warmup" if ep <= args.warmup else f"ctc{w_ctc:.2f}"
            print(f"  epoch {ep:3d} [{tag:>7}]  ctc {tot_ctc / len(train):.3f} "
                  f"cr {tot_cr / len(train):.3f}  train WER {tw:6.2f}%  held-out {hw:6.2f}%  "
                  f"| hyp0={hyps[0][:38]!r}", flush=True)

    final = wer_of(train, eval_hyps(train))
    verdict = "BEAT draft" if final < draft_wer else "did NOT beat draft"
    print(f"\nBASELINE draft WER {draft_wer:.2f}%  ->  editor WER {final:.2f}%   ({verdict})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
