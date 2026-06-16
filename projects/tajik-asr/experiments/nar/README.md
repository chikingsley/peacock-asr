# NAR editor on omni components — experiment spec

> **Status: risky feasibility test, not a sure thing.** See `CRITIQUE.md` (two independent
> source-verified reviews) — this spec was corrected against it. Honest framing of the question:
> *can a lifted, bidirectionalised omni `llama_decoder` edit a 300M-CTC draft to ≤ CTC+KenLM (14.5),
> clearly faster than the autoregressive LLM?* NOT "LLM accuracy at near-CTC speed" — one
> bidirectional LLM pass beats AR decoding but will not match CTC+KenLM throughput.

**Goal.** Bolt a *non-autoregressive* LLM editor onto our fine-tuned CTC: the editor fixes the CTC
draft in **one parallel forward pass** (not word-by-word), trading one extra LLM pass for accuracy
while staying far faster than autoregressive decoding. IBM's **NLE** (Non-autoregressive LLM-based
Editing, arXiv:2603.08397) reimplemented on Meta omni parts. **Realistic target: ≤ CTC+KenLM 14.5
on read; approaching the 10.9 AR number is a stretch goal, not the plan** (NLE *approximates* its AR
teacher and even loses to it on the harder multilingual case).

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
- **Editor:** the **`llama_decoder`** lifted out of `Wav2Vec2LlamaModel` — a 12-layer
  `StandardTransformerLMDecoder`, natively multilingual (omni's full speech-LLM scores 10.9 Tajik
  0-shot, so the language is *in these weights* — though that score is the *whole* omni system, not
  this lifted decoder + our CTC). Body frozen; train LoRA on attn+MLP. **De-causalising is a
  distribution shift, not just a mask flip** — the decoder was trained for causal next-token; the
  LoRA must absorb producing same-position edit logits with an untied head. Budget rank/warmup for it.
- **Projector:** omni's `encoder_proj` is a single **Linear** (audio_dim·stacking → llama_dim),
  trained on the *omni-LLM's* encoder features, not our CTC's. Initialise only if shape-compatible;
  **expect to retrain.** Projector depth + which CTC layer(s) feed it is a **central ablation**, not
  an afterthought (IBM used a Q-Former over stacked layers 4/8/12/16 — a Linear over final-layer
  features is weaker).

Trainable = projector + LoRA. **Do NOT inherit IBM's "14M / rank 128"** — those are for IBM's
architecture (Granite decoder + Q-Former). Recompute the exact LoRA param count for our chosen rank
over our decoder's attn+MLP, and report it **separately from activation memory**. **"Fits 12 GB" is
unproven and is a GATE, not a footnote:** LoRA still backprops through the full bidirectional decoder
activations over `audio_prefix + (2N+1)` positions with no train-time KV-cache savings. Measure one
real train step before committing.

## Input format (the NLE trick)

1. CTC greedy draft → token string `x₁…x_N` (re-tokenized into the Llama vocab).
2. Interleave insertion slots `ε` (reuse the Llama EOS id): `x̃ = (ε, x₁, ε, x₂, …, ε, x_N, ε)` —
   `N+1` slots. A K-token insertion only perturbs `2K-1` local positions (no sequence-wide shift).
3. Editor input = `[projected acoustic embeddings] ++ [embedded x̃]`, concatenated on the sequence
   axis. One **bidirectional** forward pass → logits at every interleaved position.
4. Edits are implicit: **copy** (residual identity + the `L_CR` regulariser — NOT tied embeddings;
   the omni decoder is **untied**, so the copy bias is weaker than IBM's and we mitigate with a copy
   warmup / tuned λ / explicit copy gate, *not* by naively tying post-pretraining), **replace**
   (different token), **delete** (predict `ε`), **insert** (fill an `ε` slot).
5. CTC-greedy decode (argmax, collapse, drop ε) over the output logits → final transcript.

**Draft-length bound (the sharpest risk):** a single pass can insert at most `N+1` tokens and the
CTC loss runs over `2N+1` positions — if `ref_len > 2N+1` (draft dropped words, common on
conversational), the loss is unreachable for that example. **Measure the `ref_len > 2N+1` rate on v3
before training.** Mitigation if it bites: NLE supports **iterative re-feeding** (multi-pass), which
extends the insertion budget at a speed cost.

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

## Settled (verified in our venv — do NOT re-litigate)

- **Embeddings untied** (`tied_embeddings=False`); **vocab shared by construction**
  (`target_vocab_size == llama_config.vocab_size` = 10288, our Tajik CTC vocab — so with matching v2
  variants the CTC draft ids **are** Llama ids; no re-tokenization round-trip); **audio fed as
  prefix** (not cross-attn); **attention hardcoded `CausalAttentionBias()`** (bidirectional = patch
  the vendored factory); **RoPE** positions (de-causalising is geometrically clean).
- Residual narrow risks from the shared vocab: **EOS-as-ε double duty** (`add_eos`/loss handling),
  the **CTC-blank convention**, and reference tokenisation/normalisation. Test these, not "vocab".

## Feasibility gates — run BEFORE full training (cheap, fail-fast)

1. **`ref_len ≤ 2N+1` coverage** on v3 (FLEURS + conversational separately).
2. **One real train-step memory** at the target rank — does it fit 12 GB?
3. **No-grad RTFx** of the forward path (is it actually faster than the AR LLM?).
4. **Identity-copy check** — does an untrained pass at least *copy* the draft?
5. **100-example overfit** — can it learn at all?

## Staged build

1. Extract `llama_decoder` + `encoder_proj`; **run the 5 gates above.** Use the **300M decoder** as
   the 12 GB feasibility model; 1B only after it works.
2. **Forward path, no training:** frozen-CTC-draft → interleave → frozen Llama (mask patched off) →
   CTC-decode. Verify it runs.
3. **Train** projector + LoRA with `L_CTC + L_CR` on Tajik v3.
4. **Eval** the four-way table + regression metrics below; ablate (LoRA rank, λ, projector
   depth/source-layer, text-only vs audio-conditioned to prove the acoustic prefix helps).
5. If it works: port to Farsi/Georgian; consider productizing.

## Metrics — not just WER

Unchanged-token accuracy, S/D/I breakdown, and the **"made a correct draft token wrong" rate** (the
failure mode that kills editors). Plus the four-way WER/RTFx table.

## Open risks (the genuinely unresolved ones)

Learnability (can de-causalised LoRA absorb the distribution shift?), **train-step memory** (the 12GB
gate), **realised speed** (one bidirectional pass, not CTC throughput), **draft-length coverage**
(`2N+1`), and **projector capacity** (Linear over CTC features vs IBM's Q-Former). Plus: no IBM
training code (we write the loop — crib ε-interleaving / mask construction / CTC-collapse shapes from
their Apache-2.0 `modeling_granite_speech_nar.py`); and conversational WER is **data-bound** (the v4
lever), not something NAR alone fixes. **Fallback** if de-causalising the omni decoder won't learn: a
clean small multilingual bidirectional LLM — but reuse omni first (shared tokenizer + speech path +
Tajik evidence are too valuable to discard).

## References

- IBM NLE paper: arXiv:2603.08397 (the recipe).
- IBM inference reference (Apache-2.0): `ibm-granite/granite-speech-4.1-2b-nar` →
  `modeling_granite_speech_nar.py`.
- Omni model def: `omnilingual_asr/models/wav2vec2_llama/{model,config,factory}.py`.
- Our CTC + KenLM baseline: `../lm_decoding/` and the EXPERIMENTS.md entry.
