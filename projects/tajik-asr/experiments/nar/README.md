# NAR editor on omni components — experiment spec

**Goal.** Get LLM-class accuracy at near-CTC speed by bolting a *non-autoregressive* LLM editor
onto our fine-tuned CTC. The editor fixes the CTC draft in **one parallel forward pass** (not
word-by-word), so we keep CTC throughput while gaining LLM accuracy. This is IBM's **NLE**
(Non-autoregressive LLM-based Editing, arXiv:2603.08397) reimplemented on Meta omni parts.

## Why (the numbers that motivate it)

Zero-shot bench, 2026-06-14 (Meta omni models, never trained on our data):

| | omni-LLM 300M (0-shot) | omni-LLM 1B (0-shot) | our FT CTC | our CTC+KenLM | ~RTFx |
|---|---|---|---|---|---|
| Tajik FLEURS | 12.6 | **10.9** | 16.9 | 14.5 | LLM ~6× / CTC ~hundreds× |
| Tajik conversational | 42.4 | 38.3 | 37.6 | **31.7** | |
| Farsi FLEURS | 15.8 | 14.3 | 8.5 | — | |
| Georgian test | 28.8 | 25.9 | 20.7 | — | |

Read: on Tajik read speech the **zero-shot** LLM already beats our fine-tuned CTC *and* CTC+KenLM.
The LLM ceiling is real and high — but autoregressive decoding is ~6× realtime vs CTC's hundreds×.
NAR is the only path to that accuracy without the latency. Target quality: single-digit WER on
read, sub-30 on conversational (the rest is a data lever, not architecture).

## Architecture — what we build

```
audio ─► [our fine-tuned omniASR-CTC]  ─► CTC draft (greedy)         ─┐  re-tokenize + interleave ε slots
              │  (FROZEN)                └► audio features ──► projector ─┤
              ▼                                                          ▼
        [omni Llama decoder, causal mask OFF, + LoRA]  ── one bidirectional pass ──► edit logits
              │  (body FROZEN, LoRA trains)                                            │
              ▼                                                                        ▼
        CTC-greedy over edit logits ───────────────────────────────────────────► transcript
```

All three pieces are ours / already pretrained:

- **Draft + audio features:** our `omni_ctc_300m_v2_tajik_v3_step_20000` (frozen). Char-level
  greedy draft, plus the encoder hidden states as the acoustic embeddings.
- **Editor:** the **`llama_decoder`** lifted straight out of `Wav2Vec2LlamaModel`
  (`omnilingual_asr/models/wav2vec2_llama/model.py`). It is a discrete `TransformerLMDecoder`,
  natively multilingual (the 10.9 Tajik 0-shot is *it*) — which sidesteps IBM's English-centric
  failure mode entirely. Body frozen; train LoRA (rank ~128, attn+MLP).
- **Projector:** start from omni's pretrained **`encoder_proj`** (already maps omni-encoder output
  → Llama dim); adapt/retrain it to take *our CTC's* features. ~few M params, trains.

Trainable params ≈ projector + LoRA ≈ **~14M** (per NLE). Frozen CTC + frozen Llama body. **Fits
the local 12 GB card** (inference of these models already uses ~6 GB; LoRA adds little).

## Input format (the NLE trick)

1. CTC greedy draft → token string `x₁…x_N` (re-tokenized into the Llama vocab).
2. Interleave insertion slots `ε` (reuse the Llama EOS id): `x̃ = (ε, x₁, ε, x₂, …, ε, x_N, ε)` —
   `N+1` slots. A K-token insertion only perturbs `2K-1` local positions (no sequence-wide shift).
3. Editor input = `[projected acoustic embeddings] ++ [embedded x̃]`, concatenated on the sequence
   axis. One **bidirectional** forward pass → logits at every interleaved position.
4. Edits are implicit: **copy** (residual identity + tied embeddings make copying the default),
   **replace** (different token), **delete** (predict `ε`), **insert** (fill an `ε` slot).
5. CTC-greedy decode (argmax, collapse, drop ε) over the output logits → final transcript.

## Training

- **Loss** `L = L_CTC + λ·L_CR`, λ=0.02.
  - `L_CTC`: standard CTC loss between the `(2N+1)` output logits and the reference — DP
    marginalizes all valid alignments, so **no labeled edit operations are ever needed** (this is
    the trick that makes it trainable on plain (audio, text) pairs).
  - `L_CR`: copying regularization — cross-entropy pushing each position to predict its own input
    token (reinforces the copy bias).
- **Frozen:** CTC encoder, Llama body. **Trains:** projector + LoRA.
- **Data:** our existing Tajik export (`data/datasets/v3` or v4 when ready). Same (audio, text)
  pairs the CTC trained on — no new labels.
- **Budget (NLE base):** ~3 epochs, AdamW, peak lr ~3e-5, SpecAugment + noise. Local, days not
  weeks at our data scale.

## Measurement plan

Compare **four** readouts on the *same* rows (FLEURS test + conversational held-out):

| readout | WER | RTFx |
|---|---|---|
| greedy CTC (production) | baseline | fastest |
| CTC + KenLM (proven −16% rel) | | ~hundreds× |
| **NAR editor (this)** | target ≤ CTC+KenLM | one extra parallel LLM pass |
| autoregressive omni-LLM (fine-tuned) | the ceiling | ~6× (the thing NAR beats on speed) |

Success = NAR matches/beats CTC+KenLM on WER **and** stays ≫ 6× realtime (clearly faster than the
autoregressive LLM). RTFx measured the same way as the lm_decoding experiment (decode-stage wall
time / audio seconds).

## Staged build

1. **Extract + sanity:** load `Wav2Vec2LlamaModel`, pull `llama_decoder` + `encoder_proj`; confirm
   vocab, **whether input/output embeddings are tied** (the copy trick depends on it), and the
   audio→decoder wiring (prefix vs cross-attn — `TransformerLMDecoder` suggests prefix, which is
   the NLE shape). **Open question to resolve first.**
2. **Forward path, no training:** wire frozen-CTC-draft → interleave → frozen Llama (mask off) →
   CTC-decode. Verify it runs and produces *something* (won't be good untrained).
3. **Train** projector + LoRA with `L_CTC + L_CR` on Tajik v3.
4. **Eval** the four-way table above; iterate (LoRA rank, λ, projector depth).
5. If it works: port to Farsi/Georgian; consider productizing in omni-finetune-core.

## Risks / open questions

- **Embedding tying** for the copy trick — verify in step 1; if untied, copy bias is weaker.
- **Vocab mismatch:** our CTC is char-level (10,288 pieces); the Llama has its own vocab. The draft
  must re-tokenize cleanly into Llama space and back. Char-level drafts may interleave awkwardly —
  test the tokenization round-trip early.
- **No training code from IBM** — only the paper + their Apache-2.0 inference reference
  (`modeling_granite_speech_nar.py`). We write the training loop.
- **Conversational ceiling:** even fine-tuned, conversational WER is data-bound (the v4 lever),
  not something NAR alone fixes.

## References

- IBM NLE paper: arXiv:2603.08397 (the recipe).
- IBM inference reference (Apache-2.0): `ibm-granite/granite-speech-4.1-2b-nar` →
  `modeling_granite_speech_nar.py`.
- Omni model def: `omnilingual_asr/models/wav2vec2_llama/{model,config,factory}.py`.
- Our CTC + KenLM baseline: `../lm_decoding/` and the EXPERIMENTS.md entry.
