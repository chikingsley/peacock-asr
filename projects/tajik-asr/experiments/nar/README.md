# NAR editor on omni components — plan, findings & status

A non-autoregressive (NAR) LLM **editor** that fixes our frozen CTC's greedy draft in **one bidirectional forward pass** — IBM's **NLE** (Non-autoregressive LLM-based Editing, arXiv:2603.08397) reimplemented on Meta OmniASR parts. Goal: approach LLM accuracy at far-below-AR latency. This file is the single source of truth: **status/TODO** at the top, the **design**, then a dated **gate-by-gate findings log**, the **corrections** to the original spec, and **references**. Update the status block each session; append to the findings log as gates run.

## Status & outstanding

**Where we are (2026-06-15).** Feasibility gates 1–2 **passed**. Gate 3 (learnability): **copy** works *only with a tied head*; the **edit** phase does not beat the draft. The recipe-vs-conditioning diagnostic (gate 3b) is **done — verdict: conditioning.** With the copy anchor annealed out, pure CTC degenerates (WER → 534%), proving the single Linear projector's acoustic signal is too weak to drive corrections. **A full run as-is would fail; do not start one.** Branch `nar-editor-feasibility`.

**Next, in priority order:**

1. **Richer acoustic conditioning (the blocker).** Replace the single reused Linear projector with a multi-layer / Q-Former adapter over stacked CTC encoder layers (crib shapes from IBM's Apache-2.0 `modeling_granite_speech_nar.py`). Keep a small λ throughout (IBM-style, do **not** anneal to 0 in production — the anneal was a diagnostic). Re-run the same overfit; **gate = edits beat the 18.84% draft.**
2. **Only then, staged real training** — overfit-100 → FLEURS-small → full v3. Always **tied head + copy warmup**. Precompute CTC drafts+features (frozen CTC → identical every epoch).
3. **Fallback if the omni editor still can't beat CTC+KenLM** — a clean small multilingual *bidirectionally-pretrained* LLM (the omni decoder is causal-pretrained; outputting its own input is off-distribution).

**Open risks:** projector capacity (now the #1, evidence-backed); whether a causal-pretrained decoder can edit at all; realised decode speed (faster than AR, *not* near-CTC).

**Resolved (don't re-litigate):** tied embeddings = **required** (untied can't copy); vocab is **shared** (no re-tokenisation); audio wiring is **prefix**; the editor decoder is a **fixed 1.22B** (only the discarded audio encoder differs by size); bidirectional ≈ causal memory; draft-length bound is a non-issue.

## Design — what we build & how

```text
audio ─► [our fine-tuned omniASR-CTC]  ─► CTC draft (greedy)         ─┐  interleave ε slots
              │  (FROZEN)                └► audio features ──► projector ─┤
              ▼                                                          ▼
        [omni Llama decoder, causal mask OFF, + LoRA]  ── one bidirectional pass ──► edit logits
              │  (body FROZEN, LoRA + projector train)                                │
              ▼                                                                        ▼
        CTC-greedy over edit logits ───────────────────────────────────────────► transcript
```

Three pieces, all ours / pretrained:

- **Draft + audio features:** our `omni_ctc_300m_v2_tajik_v3_step_20000` (frozen). Char-level greedy draft + the encoder hidden states (1024-d) as the acoustic embeddings.
- **Editor:** the `llama_decoder` lifted from `Wav2Vec2LlamaModel` — a `TransformerLMDecoder`, natively multilingual (this is what sidesteps NLE's English-centric failure mode). **Fixed at 1.22B** in the v2 family. Body frozen; train LoRA (rank ~128, attn+MLP). Causal mask removed (`IdentityBias`) → bidirectional.
- **Projector:** start from the pretrained `encoder_proj` — a Linear (4096,1024) that exactly matches our CTC's 1024-d encoder. Trains. *(Likely too weak alone — see findings; a multi-layer/Q-Former projector is the main upgrade lever.)*

**Input format (the NLE trick).** (1) CTC greedy draft → token ids `x₁…x_N` (already in Llama vocab — shared tokenizer). (2) Interleave insertion slots `ε` (reuse the EOS id): `x̃ = (ε, x₁, ε, …, ε, x_N, ε)`, `2N+1` positions. A K-token insertion perturbs only `2K-1` local positions. (3) Editor input = `[projected acoustic embeddings] ++ [embedded x̃]`, concatenated on the sequence axis; one bidirectional pass → logits at every position. (4) Edits are implicit: **copy** (residual identity + **tied** head — required), **replace** (different token), **delete** (predict ε), **insert** (fill an ε slot). (5) CTC-greedy decode (argmax, collapse repeats, drop ε) → transcript.

**Training.** `L = w_ctc·L_CTC + λ·L_CR`. `L_CTC`: CTC over the `2N+1` logits vs the reference (blank = ε); DP marginalises alignments, so **no edit labels** are needed — plain (audio, text) pairs. `L_CR`: copy regularizer (CE toward each position's own input token). Recipe: **copy-only warmup**, then **ramp `w_ctc` 0→1**, AdamW with cosine LR + grad clip. Frozen: CTC, Llama body, embeddings, head. Trains: projector + LoRA.

**Measurement plan.** Compare four readouts on the same rows (FLEURS test + conversational held-out):

| readout | WER (FLEURS / conv) | RTFx |
|---|---|---|
| greedy CTC (production) | 16.94 / 37.64 | fastest |
| CTC + KenLM | 14.50 / 31.66 | ~hundreds× |
| **NAR editor (this)** | target ≤ CTC+KenLM | one parallel LLM pass |
| autoregressive omni-LLM | ceiling (~10.9 read, 0-shot) | ~6× |

Success = NAR matches/beats CTC+KenLM on WER **and** stays clearly faster than the AR LLM (not "near-CTC speed" — that overclaims). "Single-digit read WER" is a **stretch goal**, not the target.

## Verdict

**The idea is sound and worth building — but the original spec overclaimed.** The premise is real (NLE is published, with Apache-2.0 reference code), and the key move (omni decoder instead of Granite) directly targets NLE's one documented failure mode. Cross-checked by two independent passes (source/web verification + codex gpt-5.5 xhigh) that converged on everything below.

## Findings (gate log — dated, append-only)

### Verified from primary sources

**Paper + HF (checked against the arXiv PDF and the live repo):**

- arXiv:2603.08397 is real: *"NLE: Non-autoregressive LLM-based ASR by Transcript Editing"*, Dekel/Thomas/Fukada/Saon (IBM), 9 Mar 2026. The mechanics (ε slots, 2N+1 positions, bidirectional single pass, implicit copy/replace/delete/insert, `L_CTC + λ·L_CR` with λ=0.02, CTC latent-alignment so no edit labels, ε = EOS) all match. **Not** hallucinated.
- `ibm-granite/granite-speech-4.1-2b-nar` exists, Apache-2.0, ships `modeling_granite_speech_nar.py` (50.7 kB) as the `trust_remote_code` inference reference. **No training code** is published (GitHub = notebooks only) — we write the loop.
- **14M trainable = base NLE** (1-layer Q-Former + LoRA rank 128); **NLE++ is ~280M** (rank 160). Both are IBM's numbers for IBM's architecture (see Correction 4).
- Multilingual weakness is in the paper: NLE **lost** to its AR baseline on multilingual CommonVoice (5.79 vs 5.18) — weak non-English CTC drafts + English-centric BPE. Exactly what the omni-decoder swap is meant to dodge.

**OmniASR source (`omnilingual_asr/models/wav2vec2_llama/`, read in our venv):**

- `Wav2Vec2LlamaModel` exposes `llama_decoder` (`StandardTransformerLMDecoder`) and `encoder_proj` — confirmed.
- `encoder_proj` is a single **Linear** `(audio_dim → llama_dim)` (`factory.py:205`), not a Q-Former.
- Audio is a **prefix** (concat on the sequence axis, `model.py concat_inputs`), not cross-attention.
- Attention is **hardcoded causal** (`CausalAttentionBias()`, `factory.py:75`); bidirectional = swap to `IdentityBias` (a module-global monkeypatch, confirmed working).
- Positional encoding is **RoPE** → dropping the causal mask is geometrically clean.
- **Embeddings are NOT tied** (`text_frontend` vs `final_proj` separate; `tied_embeddings=False`, `factory.py:215-229`).
- **Vocab is shared by construction**: `config.py:176-181` asserts CTC `target_vocab_size == llama vocab_size` (one omni tokenizer, 10288 = our Tajik CTC). Matching v2 variants → CTC draft ids **are** Llama ids.
- **The editor decoder is FIXED** — "300m/1b/7b" sizes the *audio encoder*, not the decoder. All v2 arches inherit `_7b_llama_v2`'s `LLaMAConfig`: 12 layers / d=4096 / 8 heads / ffn 2816 / RoPE / vocab 10288 = **1.22B** (`config.py:256-280`). The editor discards the omni encoder, so there is exactly one editor to train (~1.22B frozen + LoRA). Only choice: which checkpoint's (identical-arch) decoder weights to lift.

### Gate 1 — draft-length bound (2026-06-15): cleared

NLE can only insert ≤ N+1 tokens; CTC over `2N+1` positions needs `2N+1 ≥ ref_len`. Greedy CTC over both splits, draft vs reference tokenized in the shared vocab (`gate1_draft_length.py`): **FLEURS 0/599 fail (0.00%); conversational 2/1625 (0.12%)**. Draft and reference lengths track almost perfectly (FLEURS p50 N=129 vs R=129; conv 301 vs 305 — length-faithful, not deleting); `R/(2N+1) ≈ 0.50` (ε-interleaving ~doubles capacity → large headroom). The 2 fails are pathological audio, dropped by the existing length filter. Side output: worst-case editor text length `2N+1 = 1447` (conv) / 687 (FLEURS).

### Gate 2 — memory (2026-06-15): fits at batch 1–2

Real forward + CTC backward + AdamW step on the actual 1.22B decoder (frozen, bf16) with real rank-128 LoRA (82.2M trainable, ~0.99 GB Adam) on the RTX 5070 (`gate2_memory_probe.py`). Peak VRAM: FLEURS worst (L=2187) **5.5 GB**; conversational worst (40 s @ 50 fps, L=3447) **6.9 GB** at batch 1; **11.0 GB** at batch 2 (fits 12.4 GB); batch 4 OOMs. → **train at batch 1–2 + gradient accumulation**. Frozen base 2.44 GB bf16; acoustic downsampling (IBM uses 5×) buys batch size.

- **Prefix length measured** (`gate2b_audio_prefix_len.py`, pipeline tap): `T'` p50 680 / max 1844 (FLEURS), p50 1242 / **max 1995** (conversational) — pinned ~2000 by the 40 s cap.
- **Bidirectional measured, not assumed** (`gate2c_bidirectional_confirm.py`): the de-causalised decoder (`IdentityBias` on all 12 layers) builds and runs forward+backward+step; peak **identical** to causal (6.90 vs 6.89 GB B1; 11.03 vs 11.03 B2).
- **Scope caveats (non-blocking):** the probe sizes only the editor; the frozen CTC isn't resident → **precompute features** (frozen CTC → identical every epoch; cache pre-projection 768-d states, ~hundreds of GB at 1000 h; or recompute on-the-fly under `no_grad` for +~1.5–2 GB). Uses PyTorch SDPA's memory-efficient kernel, as real training does.

### Gate 3 — learnability (2026-06-15): copy needs tying; edits not yet shown

Built the real editor and overfit 100 FLEURS rows (draft WER 18.84%): lifted the pretrained omni-LLM-300M `llama_decoder` (109 tensors, zero missing/unexpected), de-causalised, reused `encoder_proj`, froze the body, trained projector + rank-128 LoRA with ε-interleaving + CTC-loss(blank=ε) + copy-reg. Artifacts: `gate3_extract_features.py`, `gate3_learnability_overfit.py`.

Result ladder (train WER on the 100):

- **Untied frozen head (omni default), λ=0.02:** stuck at **100%** — output collapses to blank/`"а"` repeats. *Copy is not learnable* — with `final_proj ≠ text_frontend` both frozen, rank-128 LoRA can't build the identity map (the decoder is a causal next-token predictor; outputting its own input is off-distribution).
- **Tie head → input embeddings:** untrained editor **36.85%**; **one** copy-only epoch → **18.84% = the draft, a perfect copy**. Residual-identity copy is near-free *once tied*.
- **CTC edit phase** (lr 3e-4 and 1e-4, copy-warmup then +CTC): **does not beat the draft.** WER pins ≈ draft (19–21%), CTC loss floors ~2–3 (doesn't →0 / doesn't memorise), with transient out-of-script garbage on CTC onset. Learns to **copy**, not **correct**.

**Read:** plumbing sound, copy works *with tying as a required recipe item*. Edit-beats-draft unshown in this minimal overfit — candidates were (1) conditioning too weak (single Linear projector), (2) recipe (abrupt CTC onset, no schedule), (3) data scale. The next sub-gate disambiguates.

### Gate 3b — recipe-vs-conditioning diagnostic (2026-06-15): verdict = **conditioning**

Re-ran the overfit with the full recipe **and** the key disambiguation fix (codex review): full-strength copy warmup, then ramp CTC weight 0→1 over 10 epochs while **annealing the copy regularizer `λ_eff = λ·(1−w_ctc) → 0`** — so a plateau-at-copy can't be blamed on a lingering copy anchor. (Also fixed a warmup bug where the copy phase had run at 2% strength, and added asserts: 0/100 CTC-infeasible rows, no ref contains the blank id.)

Trajectory as the copy anchor anneals out and CTC takes over:

| epoch | CTC weight | CTC loss | train WER | output |
|---|---|---|---|---|
| 1–12 (warmup) | 0 | — | **18.84%** | clean copy |
| 15 | 0.30 | 22.2 | 109% | dropping chars |
| 20 | 0.80 | 5.1 | 150% | mostly spaces |
| 25 | 1.00 | 8.7 | **534%** | every char space-separated |

**Pure CTC degenerates catastrophically.** As the anchor is removed, CTC loss falls while greedy WER *explodes* (insertions/blanks) — a classic degenerate CTC optimum. The crucial inference: **the copy regularizer was load-bearing** — it was holding the output *at* copy, not enabling corrections. Once unanchored, the single Linear projector's acoustic signal is **too weak to drive correct edits**, so CTC games the alignment instead. → **The bottleneck is acoustic conditioning, not the recipe.** (Contrast: IBM keeps a small λ throughout *and* uses a Q-Former projector — the anneal here was a diagnostic to expose the dependency, not the production recipe.)

**Consequence:** a full training run as-is would **not** work — it would either plateau at copy (anchor on) or degenerate (anchor off). **Do not "jump in the water."** The required next step is **richer acoustic conditioning** (multi-layer / Q-Former over stacked CTC encoder layers, λ kept small throughout), then re-run this same overfit; only if edits beat the 18.84% draft does staged real training make sense.

## Corrections to the original spec

1. **Tied embeddings — required, not optional (MEASURED, gate 3).** The spec said "residual identity + tied embeddings make copying the default," but the omni decoder is **untied** (inherited from the pre-pivot Granite plan). Untied → can't copy (100%); tied → copy in 1 epoch. **Tie the output head to the input embeddings.**
2. **Vocab mismatch is overstated.** Shared tokenizer is enforced; matching v2 variants → draft ids are Llama ids, no round-trip. Real residual risks: EOS-as-ε double duty, CTC-blank convention, normalisation.
3. **"Single-digit WER on read" is too optimistic.** NLE approximates (and sometimes loses to) its AR teacher; ceiling is bounded by the 300M CTC features + the 16.9 draft. Target ≤ CTC+KenLM 14.5; single-digit is a stretch.
4. **Param accounting is inherited, not recomputed.** rank-128 + 14M are both IBM's *base-NLE* numbers for IBM's architecture. Ours (measured, gate 2): rank-128 LoRA over q/k/v/o + gate/inner/output of 12 layers = **82.2M** (rank 64 = 41.1M).
5. **"The 10.9 zero-shot is the decoder" is misleading.** That score is the *full* omni speech-LLM (its own encoder + projector + decoder). The lifted-decoder + our-CTC stack is a different system; 10.9 proves the language is in the family, not that this stack reaches it.

## References

- Paper (definitive method/loss/params): <https://arxiv.org/pdf/2603.08397>
- Apache-2.0 inference reference — crib ε-interleaving, bidirectional-mask, and output-head / CTC-collapse shapes: <https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar/tree/main> (`modeling_granite_speech_nar.py`, `configuration_…`, `processing_…`)
- Closest open **training** reference for NAR ASR correction: FastCorrect (Microsoft NeuralSpeech), <https://github.com/microsoft/NeuralSpeech>; Levenshtein Transformer (in fairseq) for the insert/delete framing. Do **not** inherit Granite's tying/tokenizer/EOS assumptions.
- Our ground truth: `omnilingual_asr/models/wav2vec2_llama/{model,config,factory}.py` (vendored in `.venv`); `../lm_decoding/` + the top-level `EXPERIMENTS.md` for baselines.
