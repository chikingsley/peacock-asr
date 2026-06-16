"""Gate 2 for the NAR editor (NLE): does one real training step fit the 12 GB card?

The editor decoder is FIXED in the omni v2 family: 300m/1b/7b only change the *audio
encoder* (which we replace with our own frozen CTC); the `llama_decoder` is the same
12-layer / d=4096 / 8-head / ffn-2816 / vocab-10288 model — **1.22B params** — in every
case. So there is exactly one editor to size.

This builds that decoder from config (random init — weights don't change memory), freezes
it, attaches **real LoRA** (forward-hooks on q/k/v/o + gate/inner/output projections, the
adapters IBM trains), adds a trainable acoustic projector and output head, then runs a real
forward + CTC-loss backward + Adam step at the worst-case sequence lengths gate 1 measured
(`2N+1` up to 1447 for conversational, 687 for FLEURS) plus the audio prefix. Reports peak
VRAM per (batch, length) so we know the feasible batch size before building anything real.

Memory note: bidirectional (IdentityBias) vs causal attention have ~identical peak memory
under SDPA (same activation buffers; only the additive mask values differ), so this measures
the stock decoder and the number transfers to the de-causalised editor.

  uv run python experiments/nar/gate2_memory_probe.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

CARD_GB = 12.0
MODEL_DIM = 4096
VOCAB = 10288
AUDIO_DIM = 1024  # our CTC encoder hidden -> projector input (size barely affects memory)
LORA_RANK = 128   # the spec's rank; we also report the analytic r=64 delta


def build_decoder(dtype: torch.dtype, device: str) -> nn.Module:
    from fairseq2.models.llama import LLaMAConfig
    from omnilingual_asr.models.wav2vec2_llama.factory import OmnilingualASRLLamaFactory

    cfg = LLaMAConfig(
        model_dim=MODEL_DIM, max_seq_len=8192, vocab_size=VOCAB, pad_idx=1,
        num_layers=12, num_attn_heads=8, num_key_value_heads=8,
        ffn_inner_dim=4096, rope_theta=10000.0, dropout_p=0.0,
    )
    dec = OmnilingualASRLLamaFactory(cfg).create_decoder()
    dec.requires_grad_(requires_grad=False)  # body frozen
    dec = dec.to(device=device, dtype=dtype)
    # RoPE freq buffers must stay fp32 (view_as_complex rejects bf16); q is cast to fp32
    # inside the rotary forward anyway, so this matches real bf16 training.
    for m in dec.modules():
        f = getattr(m, "freqs", None)
        if isinstance(f, torch.Tensor):
            m.freqs = f.float()
    return dec


class LoRA(nn.Module):
    """Low-rank adapter added to a wrapped Linear's output via forward-hook."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, dtype, device) -> None:
        super().__init__()
        self.a = nn.Linear(in_dim, rank, bias=False).to(device=device, dtype=dtype)
        self.b = nn.Linear(rank, out_dim, bias=False).to(device=device, dtype=dtype)
        nn.init.zeros_(self.b.weight)

    def __call__(self, x: Tensor) -> Tensor:
        return self.b(self.a(x))


def attach_lora(dec, rank: int, dtype, device) -> list[nn.Parameter]:
    """Hook every q/k/v/o + gate/inner/output projection with a real trainable LoRA."""
    targets = ("q_proj", "k_proj", "v_proj", "output_proj", "gate_proj", "inner_proj")
    adapters: list[LoRA] = []
    params: list[nn.Parameter] = []
    for name, m in dec.named_modules():
        leaf = name.split(".")[-1]
        if leaf in targets and hasattr(m, "weight") and m.weight.dim() == 2:
            out_dim, in_dim = m.weight.shape
            lora = LoRA(in_dim, out_dim, rank, dtype, device)
            adapters.append(lora)
            params += list(lora.parameters())

            def hook(_mod, inp, out, _lora=lora):
                return out + _lora(inp[0])

            m.register_forward_hook(hook)
    return params


def lora_param_count(rank: int) -> int:
    per_layer = (
        rank * (MODEL_DIM + MODEL_DIM) * 4          # q,k,v,o : 4096<->4096
        + rank * (MODEL_DIM + 2816) * 2             # gate,inner : 4096->2816
        + rank * (2816 + MODEL_DIM)                 # ffn output_proj : 2816->4096
    )
    return per_layer * 12


def run_step(dec, lora_params, B: int, T_audio: int, M_text: int, dtype, device) -> float:
    from fairseq2.nn import BatchLayout

    proj = nn.Linear(AUDIO_DIM, MODEL_DIM).to(device=device, dtype=dtype)        # trainable
    head = nn.Linear(MODEL_DIM, VOCAB, bias=False).to(device=device, dtype=dtype)  # trainable
    embed = nn.Embedding(VOCAB, MODEL_DIM).to(device=device, dtype=dtype)
    embed.requires_grad_(requires_grad=False)  # text frontend treated as frozen body
    opt = torch.optim.AdamW(lora_params + list(proj.parameters()) + list(head.parameters()),
                            lr=1e-4)

    L = T_audio + M_text
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    audio_feats = torch.randn(B, T_audio, AUDIO_DIM, device=device, dtype=dtype)
    text_ids = torch.randint(0, VOCAB, (B, M_text), device=device)
    prefix = proj(audio_feats)                       # [B, T_audio, d]  (grad path to projector)
    text_emb = embed(text_ids)                       # [B, M_text, d]
    seqs = torch.cat([prefix, text_emb], dim=1)      # [B, L, d]
    layout = BatchLayout((B, L), seq_lens=[L] * B, device=torch.device(device))

    hidden = dec(seqs, layout)                       # [B, L, d]  bidirectional in real editor
    logits = head(hidden[:, T_audio:, :])            # CTC over the M_text edit positions
    log_probs = logits.float().log_softmax(-1).transpose(0, 1)  # [M_text, B, vocab]
    tgt_len = max(1, M_text // 2)                    # ~N tokens of reference (R ~= N from gate 1)
    targets = torch.randint(1, VOCAB, (B, tgt_len), device=device)
    loss = nn.functional.ctc_loss(
        log_probs,
        targets,
        torch.full((B,), M_text, dtype=torch.long),
        torch.full((B,), tgt_len, dtype=torch.long),
        blank=0, zero_infinity=True,
    )
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    del proj, head, embed, opt, audio_feats, prefix, text_emb, seqs, hidden, logits, log_probs
    return peak


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA — gate 2 needs the GPU")
        return 1
    device, dtype = "cuda", torch.bfloat16
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"device: {name}  total {total:.1f} GB  dtype {dtype}", flush=True)

    dec = build_decoder(dtype, device)
    nparams = sum(p.numel() for p in dec.parameters())
    weights_gb = nparams * 2 / 1e9
    print(f"editor decoder: {nparams/1e6:.0f}M params (frozen), {weights_gb:.2f} GB bf16",
          flush=True)
    lp = attach_lora(dec, LORA_RANK, dtype, device)
    nl = sum(p.numel() for p in lp)
    print(f"LoRA rank {LORA_RANK}: {nl/1e6:.1f}M trainable  "
          f"(analytic r=64 -> {lora_param_count(64)/1e6:.1f}M, "
          f"r=128 -> {lora_param_count(128)/1e6:.1f}M)", flush=True)
    print(f"Adam overhead at r={LORA_RANK} ~ {nl*12/1e9:.2f} GB (grad+m+v); "
          f"already included in measured peaks below.\n", flush=True)

    # (label, batch, T_audio, M_text=2N+1) — audio prefix from frame-rate assumption.
    # 50 fps = no acoustic downsample; the editor could downsample (IBM uses 5x) to shrink T.
    scenarios = [
        ("FLEURS typical  (20s@50fps, M=259)", 1, 1000, 259),
        ("FLEURS worst    (30s@50fps, M=687)", 1, 1500, 687),
        ("conv typical    (30s@50fps, M=611)", 1, 1500, 611),
        ("conv worst NO-ds(40s@50fps, M=1447)", 1, 2000, 1447),
        ("conv worst 5x-ds(40s@10fps, M=1447)", 1, 400, 1447),
        ("conv worst NO-ds  batch 2", 2, 2000, 1447),
        ("conv worst NO-ds  batch 4", 4, 2000, 1447),
        ("conv worst 5x-ds  batch 4", 4, 400, 1447),
        ("conv worst 5x-ds  batch 8", 8, 400, 1447),
    ]
    print(f"{'scenario':<38} {'B':>2} {'T_aud':>6} {'M':>5} {'L':>5} {'peak GB':>8}  fit?",
          flush=True)
    for label, B, T, M in scenarios:
        try:
            peak = run_step(dec, lp, B, T, M, dtype, device)
            fit = "OK" if peak < CARD_GB else "OVER"
            print(f"{label:<38} {B:>2} {T:>6} {M:>5} {T+M:>5} {peak:>8.2f}  {fit}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{label:<38} {B:>2} {T:>6} {M:>5} {T+M:>5} {'OOM':>8}  OVER", flush=True)
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
