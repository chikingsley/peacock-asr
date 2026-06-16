"""Gate 2c: turn the "bidirectional ~= causal peak" assumption into a measurement.

The gate-2 probe ran the stock CAUSAL decoder and argued bidirectional has ~identical peak
under SDPA. De-causalising is the whole experiment, so this builds the REAL bidirectional
decoder (swap CausalAttentionBias -> IdentityBias, which is how the editor removes the mask)
and runs a real forward + CTC backward + step at the worst-case shapes, head-to-head with
causal. It also confirms the de-causalised forward/backward actually RUNS on the omni decoder
(plumbing de-risk), not just that memory matches.

  uv run python experiments/nar/gate2c_bidirectional_confirm.py
"""

from __future__ import annotations

import omnilingual_asr.models.wav2vec2_llama.factory as wllf
import torch
from gate2_memory_probe import LORA_RANK, attach_lora, build_decoder, run_step


def build_bidirectional(dtype, device):  # noqa: ANN201
    from fairseq2.models.transformer.attention_bias import IdentityBias

    orig = wllf.CausalAttentionBias
    wllf.CausalAttentionBias = IdentityBias  # factory reads this module global per layer
    try:
        dec = build_decoder(dtype, device)
    finally:
        wllf.CausalAttentionBias = orig
    return dec


def confirm_bias(dec, kind: str) -> None:
    biases = {
        type(getattr(lyr.self_attn.sdpa, "attn_bias",
                     getattr(lyr.self_attn.sdpa, "bias", None))).__name__
        for lyr in dec.layers
    }
    print(f"  {kind} decoder attn-bias across layers: {biases}", flush=True)


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 1
    device, dtype = "cuda", torch.bfloat16
    print(f"device: {torch.cuda.get_device_name(0)}  dtype {dtype}\n", flush=True)

    # worst-case shapes from gates 1 + 2b
    shapes = [
        ("conv worst B1 (T=2000, M=1447)", 1, 2000, 1447),
        ("conv worst B2 (T=2000, M=1447)", 2, 2000, 1447),
    ]

    results: dict[str, dict[str, float]] = {}
    for kind, builder in (("causal", build_decoder), ("bidirectional", build_bidirectional)):
        dec = builder(dtype, device)
        confirm_bias(dec, kind)
        lp = attach_lora(dec, LORA_RANK, dtype, device)
        results[kind] = {}
        for label, B, T, M in shapes:
            try:
                peak = run_step(dec, lp, B, T, M, dtype, device)
                results[kind][label] = peak
                print(f"  {kind:13s} {label}: {peak:.2f} GB", flush=True)
            except torch.cuda.OutOfMemoryError:
                results[kind][label] = float("nan")
                print(f"  {kind:13s} {label}: OOM", flush=True)
                torch.cuda.empty_cache()
        del dec, lp
        torch.cuda.empty_cache()
        print(flush=True)

    print(f"{'shape':<34} {'causal':>9} {'bidir':>9} {'Δ':>7}", flush=True)
    for label, *_ in shapes:
        c = results["causal"][label]
        b = results["bidirectional"][label]
        print(f"{label:<34} {c:>8.2f}G {b:>8.2f}G {b-c:>+6.2f}G", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
