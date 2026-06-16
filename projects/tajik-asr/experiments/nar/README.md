# NAR editor on omni components — plan, findings & status

A non-autoregressive (NAR) LLM **editor** that fixes our frozen CTC's greedy draft in **one bidirectional forward pass** — IBM's **NLE** (Non-autoregressive LLM-based Editing, arXiv:2603.08397) reimplemented on Meta OmniASR parts. Goal: approach LLM accuracy at far-below-AR latency. This file is the single source of truth: **status/TODO** at the top, the **design**, then a dated **gate-by-gate findings log**, the **corrections** to the original spec, and **references**. Update the status block each session; append to the findings log as gates run.

## Status & outstanding

**Where we are (2026-06-15).** The feasibility study is **complete, with a negative result.** Ladder: gates 1–2 feasibility **passed**; gate 3 copy works *only tied*; gate 3b Linear+anneal **collapses** (conditioning is the axis); gate 4 IBM **Q-Former** fixes the collapse and *marginally* beats the overfit draft; gate 4b more projector = no gain; **gate 5 (first held-out run): corrections do NOT generalise** — held-out stays at copy (16.57% vs 16.53% draft), and gate 4's overfit "beat" was **memorisation** (it vanished at 480 rows). **Bottom line: the editor robustly learns to copy the CTC draft but not to correct it.** Branch `nar-editor-feasibility`.

**Decision point — three honest options:**

1. **Stop here (recommended).** The accumulated evidence (no generalising correction at 100→480 rows; trend going the wrong way) plus the *feature-ceiling* explanation says this is likely a real limit, not a tuning gap. **CTC + KenLM already buys the WER win we wanted** (FLEURS 16.9→14.5, conv 37.6→31.7) at hundreds× realtime, with none of this complexity. That is the production answer.
2. **One conclusive scale test, then decide.** Wire up the real **180k-row training export** (bigger job — v3's eval export only exposes test partitions) and run once. Only worth it if we want certainty; expected value is low given the trend.
3. **Different architecture** — a *bidirectionally-pretrained* small multilingual LLM instead of the causal-pretrained omni decoder (outputting its own input is off-distribution for a next-token model). Bigger pivot; only if the editor idea is strategically important.

**What we proved (kept regardless):** the NLE/omni editor is *buildable* and *runs* (gates 1–2), copy needs a tied head, and acoustic conditioning (Q-Former) is the axis that separates "collapses" from "stable." The open failure is **generalising correction**, most consistent with the editor's audio being the *same* features the CTC already used.

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

**Consequence:** a full training run as-is would **not** work — it would either plateau at copy (anchor on) or degenerate (anchor off). The required next step is **richer acoustic conditioning** (gate 4).

### Gate 4 — richer conditioning, IBM Q-Former projector (2026-06-15): collapse fixed, beats draft marginally

Swapped the single Linear projector for **IBM's actual NLE projector** (ported from `granite-speech-4.1-2b-nar`, Apache-2.0): per-layer LayerNorm over **4 stacked CTC encoder layers** → Linear → windowed mean-pool 5× downsample → 1-layer cross-attention **Q-Former** with learned queries. Everything else (lifted+tied decoder, LoRA, ε-interleave, CTC+copy loss, the *same* recipe incl. λ-anneal) is reused from gate 3 — so **gate 4 vs gate 3b is a clean A/B on the projector only**. Codex-reviewed (port faithful, A/B clean). Trainable: 50.4M Q-Former + 82.2M LoRA. Artifacts: `gate4_extract_features.py`, `gate4_conditioning.py`.

Same recipe, same λ-anneal as gate 3b — but the Q-Former changes the outcome completely:

| | gate 3b (Linear) | gate 4 (Q-Former) |
|---|---|---|
| CTC ramp onset (w≈0.3–0.8) | 109% → 150% | 87% → 100% (transient) |
| w_ctc = 1.0 (anchor gone) | **534%** (collapse) | **recovers** → 31% → 27% → 18.8% |
| final (epoch 80) | diverged | **17.92%** (CTC loss 4.3 → 0.29) |

**The conditioning hypothesis is confirmed.** Where the Linear projector catastrophically degenerated once unanchored, the Q-Former — same recipe — **recovers and converges to clean, audio-driven output** and **beats the 18.84% draft → 17.92%**. The richer multi-layer features are what let CTC make sensible (not degenerate) use of the audio.

**But the correction ability is weak.** The beat is only ~0.9 abs / ~5% rel, and CTC loss **floors at ~0.29 (not →0)** — so even on a memorizable 100-row *overfit* the editor learns "copy + a few fixes," not a real correction policy (for scale: KenLM already buys −2.4 WER on *held-out* FLEURS). So: **single Linear → collapse; 1-layer Q-Former → stable + marginal corrections; [still need] → strong corrections.** Conditioning was necessary and is now proven to help, but this config isn't yet sufficient.

### Gate 4b — scale the projector (2-layer Q-Former, 2× downsample) (2026-06-15): no gain — projector is not the lever

Same harness/recipe as gate 4, only the projector scaled up: **2 Q-Former layers** (IBM's NLE++ depth) + **gentler 2× downsample** (block 16, vs gate 4's 1 layer / 5×) → 84M trainable. Clean A/B vs gate 4.

- **CTC onset much smoother** (epoch 15: 21% vs gate 4's 87%) — the bigger/finer projector destabilises less while anchored.
- **But at full λ-anneal (λ→0) it degenerated *longer*** (epoch 25–30: 148% → 101%, recovered only by epoch 35 vs gate 4's epoch 25) — more capacity finds the degenerate no-anchor optimum more easily.
- **Final: 18.93% — did NOT beat the draft** (CTC floor ~0.47), *worse* than gate 4's 17.92% / 0.29.

**Verdict: more projector capacity/resolution is not the remaining lever.** Doubling Q-Former depth and halving the downsample produced no correction gain (slightly worse). Combined with gate 4, the overfit ladder has now answered what it can: **copy works (tie), conditioning matters (Q-Former ≫ Linear), but corrections stay ≈copy regardless of projector size.** The CTC floor (~0.3–0.5, never →0) on a memorisable 100-row set is the tell. That points the remaining bottleneck at **data** — tested next.

### Gate 5 — first train→held-out run (480 FLEURS train / 119 held-out) (2026-06-15): corrections do NOT generalise

The first non-overfit run: gate-4 config (1-layer Q-Former, 5×, tied), trained on 480 FLEURS rows with a **held-out 119** the editor never sees. Held-out draft WER = **16.53%** (train draft 17.04%) — the editor must beat *that* to show generalising corrections. Ran both λ regimes:

| run | λ | train WER (draft 17.04) | **held-out WER (draft 16.53)** |
|---|---|---|---|
| gate 5 | kept (IBM) | 17.06% | **16.57%** (= copy) |
| gate 5b | annealed | 17.10% | **16.76%** (slightly worse than copy) |

**Neither beats the draft — on train *or* held-out.** And the clincher: gate 4's 100-row train-beat (17.92 vs 18.84, −0.9) **vanished at 480 rows** (gate 5: 17.06 vs 17.04). That beat was **memorisation of the tiny overfit set, not a correction policy** — exactly what a held-out eval is for. With `keep-lam` it's pinned at copy (anchor holds it); with anneal it transiently collapses then returns to copy. Either way: **the editor robustly learns to *copy* the CTC draft and does not learn corrections that generalise** at reachable FLEURS scale (100–480 rows).

**Most consistent explanation:** the *feature-ceiling* hypothesis — the editor's acoustic input is the *same* CTC encoder features that already produced the wrong draft, so there's little extra signal to correct from; the bidirectional LLM prior alone isn't enough. (Data scale isn't ruled out — 480 ≪ IBM's 70k h — but the trend is the wrong direction: more data made even the memorisation-beat disappear, and held-out never moved off copy.)

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
