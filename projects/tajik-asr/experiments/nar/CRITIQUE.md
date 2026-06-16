# NAR editor spec — review & critique

Companion to `README.md`. This is a design review, not a rewrite of the plan. It records what was **verified against primary sources** (the paper PDF, the live HF repo, and the OmniASR source in our own venv), what the spec gets **right**, what it gets **wrong**, and the **risks it omits**. Cross-checked by two independent passes (source/web verification + codex gpt-5.5 xhigh) that converged on everything below.

## Verdict

**The idea is sound and worth building — but the spec currently overclaims.** Frame it as a *risky feasibility test* ("can a lifted, bidirectionalised omni decoder edit a 300M-CTC draft to ≤ CTC+KenLM, clearly faster than the AR LLM?"), not as "LLM accuracy at near-CTC speed". The premise is real (NLE is a published IBM method with Apache-2.0 reference code), and the key design move (omni decoder instead of Granite) is genuinely smart. The weak points are optimism in the headline numbers and three unstated technical risks.

## Verified (primary sources — do not re-litigate)

**Paper + HF (checked against the arXiv PDF and the live repo):**

- arXiv:2603.08397 is real: *"NLE: Non-autoregressive LLM-based ASR by Transcript Editing"*, Dekel/Thomas/Fukada/Saon (IBM), 9 Mar 2026. The method mechanics in the README (ε slots, N+1 slots / 2N+1 positions, bidirectional single pass, implicit copy/replace/delete/insert, `L = L_CTC + λ·L_CR` with λ=0.02, CTC latent-alignment so **no edit labels needed**, ε = EOS) all match the paper. The premise is **not** hallucinated.
- `ibm-granite/granite-speech-4.1-2b-nar` exists, Apache-2.0, ships `modeling_granite_speech_nar.py` (50.7 kB) as the `trust_remote_code` inference reference. **No training code** is published anywhere (GitHub = notebooks only). The README's "we write the training loop" is correct.
- **14M trainable = base NLE** (1-layer Q-Former + LoRA **rank 128**) — the rank and the 14M are the *same* (base) variant, consistent with each other. The leaderboard variant **NLE++ is ~280M** (bigger projector + LoRA **rank 160**). See Correction 4 — the issue isn't mixing, it's carrying IBM's numbers onto a different architecture.
- Multilingual weakness is in the paper: NLE **lost** to its AR baseline on multilingual CommonVoice (5.79 vs 5.18 WER) — weak non-English CTC drafts + English-centric BPE. This is exactly the failure the omni-decoder swap is meant to dodge.

**OmniASR source (`omnilingual_asr/models/wav2vec2_llama/`, read in our venv):**

- `Wav2Vec2LlamaModel` really exposes `llama_decoder` (a `StandardTransformerLMDecoder`) and `encoder_proj` — confirmed (`model.py`, `factory.py`).
- `encoder_proj` is a single **Linear** `(audio_dim*stacking → llama_dim)` (`factory.py:205`), **not** a Q-Former. Reusing it = a linear projector only.
- Audio is a **prefix** (concatenated on the sequence axis, `model.py concat_inputs`), **not** cross-attention. → The README's "prefix vs cross-attn" open question is **answered: prefix.**
- Attention is **hardcoded causal**: `CausalAttentionBias()` in `factory.py:75`, no config flag. Bidirectional = patching the vendored factory, not a kwarg.
- Positional encoding is **RoPE** (`factory.py`), so dropping the causal mask is geometrically clean (relative positions; no learned-absolute breakage).
- **Embeddings are NOT tied**: `text_frontend` (StandardEmbedding) and `final_proj` (Linear, bias=False) are separate; `tied_embeddings=False` (`factory.py:215-229`).
- **Vocab is shared by construction**: `config.py:176-181` asserts `wav2vec2_asr_config.target_vocab_size == llama_config.vocab_size` (one omni tokenizer, size 10288 = our Tajik CTC's vocab). Match the v2 LLM variant → CTC draft ids **are** Llama ids.
- **The editor decoder is a FIXED size — "300m/1b/7b" is the audio encoder, not the decoder.** All v2 arches (`300m_v2`/`1b_v2`/`3b_v2`) inherit `_7b_llama_v2`'s `LLaMAConfig` and only swap the `wav2vec2_asr_config` (audio encoder); the `llama_decoder` is always **12 layers / d=4096 / 8 heads / ffn 2816 / RoPE / vocab 10288 = 1.22B params** (`config.py:256-280`, confirmed by building it). Since the NAR editor discards the omni audio encoder and feeds it our own frozen CTC features, **there is exactly one editor decoder to train, ~1.22B frozen + LoRA** — the "which size model" question is moot. The only real choice is *which checkpoint's* (architecturally identical) decoder weights to lift — likely the 1B/7B card's, co-trained with a stronger encoder.

## What the spec gets right

1. **The core insight is the strongest part.** Swapping IBM's English-centric Granite for omni's natively-multilingual `llama_decoder` directly targets NLE's one documented failure mode. Omni has real Tajik (the 10.9 zero-shot is evidence the *weights* know the language).
2. **Reusing real components instead of coding from scratch.** `encoder_proj` exists, the prefix wiring exists, the shared tokenizer exists — much of the scaffolding is pretrained.
3. **Honest, well-grounded measurement plan.** The four-way table compares against baselines we already have real numbers for (greedy 16.9, KenLM 14.5, AR ~10.9), with RTFx measured the same way as the `lm_decoding` experiment.
4. **The training trick is correctly understood.** CTC marginalisation → plain (audio, text) pairs, no edit-op labels. This is what makes it trainable on our existing v3 export.

## Corrections (spec says X; the truth is Y)

1. **Tied embeddings — the body contradicts the code, and this is now MEASURED (gate 3).** README line 57-58 says "residual identity + tied embeddings make copying the default." The omni decoder is **untied** (`final_proj ≠ text_frontend`), a claim inherited from the pre-pivot Granite plan (Granite 1B *is* tied). Gate 3 makes this decisive: **untied → the editor cannot even learn to copy 100 examples (stuck at 100% WER); tying the head → copy is near-free (perfect draft copy in 1 epoch).** So this is not a soft "mitigate accordingly" — **tying the output head to the input embeddings is a required recipe item.** (Tying discards the pretrained final_proj, but the input embeddings make a fine tied head and LoRA adapts; copy-warmup on top.) See §Learnability.

2. **Vocab mismatch is overstated.** README risk #2 frames a char-vs-Llama re-tokenization round-trip. The omni family **enforces** equal vocab (shared tokenizer); with matching v2 variants the draft ids are already Llama ids. Downgrade this. The *real* residual risks here are narrower: EOS-as-ε double duty (`add_eos`/loss handling in `model.py`), the CTC-blank convention, and reference tokenisation/normalisation.

3. **"Single-digit WER on read" is too optimistic.** NLE *approximates* its AR teacher and even *loses* to it on the harder multilingual case. The realistic ceiling here is bounded by the 300M CTC's acoustic features and the 16.9 draft. Keep the table target (**≤ CTC+KenLM 14.5**); demote "single-digit" / approaching the 10.9 AR to an explicit **stretch goal**.

4. **Param accounting is inherited, not recomputed — and our number is now measured.** "LoRA rank ~128" and "~14M" are *both* IBM's **base-NLE** numbers (self-consistent — base NLE is rank 128 totalling 14M; NLE++ is rank 160 totalling ~280M), but they're for **IBM's architecture** (Granite decoder + Q-Former projector). Our build differs: the fixed 1.22B omni decoder + a reused **Linear** `encoder_proj` (no Q-Former). Measured (gate 2): **rank-128 LoRA over q/k/v/o + gate/inner/output of all 12 layers = 82.2M trainable** (rank 64 = 41.1M) — *not* 14M. Report this separately from activation memory (the latter dominates and is now measured: §"Risks", gate 2).

5. **"The 10.9 zero-shot is *it* (the decoder)" is misleading.** That score is produced by the **full omni speech-LLM** — its own encoder + projector + decoder, trained together. Lifting only the decoder and feeding it a *300M-CTC*-derived prefix is a different system. The 10.9 proves the language is in the family, not that the lifted-decoder + our-CTC stack reaches it.

## Risks the spec omits (the important new findings)

- **Draft-length bound — MEASURED, cleared (gate 1, 2026-06-15).** NLE can only **insert ≤ N+1** tokens; the CTC loss runs over `2N+1` positions, so a valid alignment needs `2N+1 ≥ ref_len`. Ran greedy CTC over both eval splits and tokenized draft vs reference in the shared omni vocab (`gate1_draft_length.py`): **FLEURS test 0/599 fail (0.00%); conversational held-out 2/1625 fail (0.12%)**. Draft and reference lengths track almost perfectly (FLEURS p50 N=129 vs R=129; conversational 301 vs 305 — the CTC is length-faithful, not silently deleting), and `R/(2N+1) ≈ 0.50` at p50/p95 (the ε-interleaving roughly doubles capacity, so there is large headroom). The 2 conversational fails are pathological audio (CTC badly under-transcribed overlapping/child speech), trivially handled by the length filter training already applies. **This was flagged as the sharpest unstated risk; it is empirically not a problem on our data.** Side output for gate 2: worst-case editor text length is `2N+1 = 1447` (conversational) / 687 (FLEURS) — within RoPE's 8192 range.
- **"Fits the 12 GB card" — MEASURED, fits at batch 1–2 (gate 2, 2026-06-15).** Ran a real forward + CTC-loss backward + AdamW step on the actual 1.22B editor decoder (frozen, bf16) with real rank-128 LoRA (82.2M trainable, ~0.99 GB Adam) on the RTX 5070 (`gate2_memory_probe.py`). Peak VRAM: every batch-1 case is comfortable — FLEURS worst (L=2187) **5.5 GB**, conversational worst with no acoustic downsampling (40 s @ 50 fps, L=3447) **6.9 GB**. Batch 2 at that worst case **11.0 GB** (fits the 12.4 GB card); batch 4 OOMs. So **train at batch 1–2 + gradient accumulation** — standard, and fine. The frozen base is 2.44 GB bf16; LoRA + activations are the rest. Acoustic downsampling (IBM uses 5×) is the lever that buys batch size: 40 s drops from 2000 to 400 prefix frames. The prefix length was then **measured directly** (`gate2b_audio_prefix_len.py`, tapping the production pipeline): CTC encoder output frames `T'` are p50 680 / max 1844 (FLEURS) and p50 1242 / **max 1995** (conversational) — pinned at ~2000 by the 40 s cap, confirming the `T'=2000` worst case used above. **Bidirectional is measured, not assumed** (`gate2c_bidirectional_confirm.py`): the real de-causalised decoder (`IdentityBias` on all 12 layers) builds and runs forward+backward+step, with peak **identical** to causal (6.90 vs 6.89 GB at B1; 11.03 vs 11.03 GB at B2) — confirming both the plumbing and the SDPA-equivalence argument.
- **Two caveats on the probe's scope (non-blocking).** (a) The probe sizes only the *editor* (decoder + LoRA + projector + head); the frozen **CTC encoder is not resident**. That's correct **if we precompute and cache the CTC drafts + features offline** — which is the right call anyway: the CTC is frozen, so its features are identical across all epochs, making caching a pure compute win (no redundant 13 h-of-audio forward × 3 epochs) on top of the memory win. Storage is the only cost: cache the pre-projection encoder hidden states (~768-dim) ≈ low-hundreds of GB at 1000 h, project on-the-fly. If instead features are computed on-the-fly, add ~1.5–2 GB for the CTC forward (under `no_grad`) — still fits 12 GB, just tighter. **Build plan should state: precompute features.** (b) The probe runs no flash-attn-specific path; it uses PyTorch SDPA's memory-efficient kernel, which is what real training uses.
- **"Near-CTC speed" overclaims.** One bidirectional 1B forward pass can beat AR decoding but will not resemble CTC+KenLM throughput. Reframe the headline as **"clearly faster than the AR LLM"** (the table already does this — make the prose match).
- **De-causalising is a distribution shift, not just a mask flip.** The decoder was trained for next-token causal prediction; producing same-position edit logits with an untied head is off the training distribution. The LoRA has to absorb that shift — budget for it (rank, warmup).
- **Projector downgrade.** A single Linear over (presumably) final-layer CTC features is poorer than IBM's Q-Former over stacked layers 4/8/12/16. And the omni `encoder_proj` was trained on the *omni-LLM's* encoder features, not our fine-tuned CTC's. Reword to "initialise if shape-compatible, **expect to retrain**", and make projector depth + source-layer selection a **central ablation**, not an afterthought.

## Learnability (gate 3, 2026-06-15) — copy CONFIRMED needs tying; edits not yet shown

Built the real editor and overfit it on 100 FLEURS rows (draft WER 18.84%): lifted the
pretrained omni-LLM-300M `llama_decoder` (109 tensors, zero missing/unexpected),
de-causalised to bidirectional (`IdentityBias`), reused the pretrained `encoder_proj`
(shape (4096,1024) — exact match to our 1024-dim CTC encoder), froze the body, trained only
projector + rank-128 LoRA with ε-interleaving + CTC-loss(blank=ε) + copy-regulariser.
Artifacts: `gate3_extract_features.py`, `gate3_learnability_overfit.py`.

**The result ladder (each line one config; train WER on the 100):**

- **Untied frozen head (the omni default), λ=0.02:** stuck at **100%** — loss plateaus ~3.0,
  greedy output collapses to blank/`"а"` repeats. *Copy is not learnable.* This is the
  decisive empirical confirmation of Correction 1: with `final_proj ≠ text_frontend` both
  frozen, rank-128 LoRA cannot build the identity/copy map (the omni decoder is a *causal
  next-token* predictor; outputting one's own input is off-distribution for it).
- **Tie the head to the input embeddings:** untrained editor jumps to **36.85%**, and **one**
  epoch of copy-only warmup reaches **18.84% = the draft, a perfect copy**. Tying makes
  residual-identity copy near-free, exactly as the README's mechanism claims — but only once
  tied, which the stock model is not.
- **CTC edit phase** (tried lr 3e-4 and 1e-4, copy-warmup then +CTC): **does not beat the
  draft.** WER pins at ≈ draft (19–21%), CTC loss floors at ~2–3 (does not →0, i.e. it does
  not memorise the corrections), and CTC onset transiently injects out-of-script garbage
  (CJK tokens) before relaxing back to copy. The editor learns to **copy**, not to **correct**.

**Read:** the architecture/plumbing is sound and the copy bias works — *with tying as a
required recipe item* (promote Correction 1 from "mitigate" to "do this"). But getting edits
to beat the draft was not demonstrated in this minimal overfit. Likely levers, in order: (1)
**richer acoustic conditioning** — a single reused Linear over final-layer CTC features (the
"projector downgrade" risk) is probably too weak to tell the editor *what* to fix; IBM uses a
Q-Former over stacked layers. (2) A **real training recipe** — grad clipping, LR schedule,
gentler CTC onset; the toy AdamW loop is CTC-unstable. (3) **Data scale** — 100 examples is
enough to prove copy, likely too few to learn a correction policy. So: feasibility (gates 1–2)
passed; learnability is **partial** — copy proven, edit-beats-draft is the open question that
defines the real training experiment, not a toy-loop gate.

## Suggested spec changes

- **Add feasibility gates before full training** (cheap, fail-fast): (a) `ref_len ≤ 2N+1` coverage rate on v3; (b) memory for one real train step at the target rank; (c) no-grad RTFx of the forward path; (d) identity-copy overfit (does an untrained pass at least copy?); (e) 100-example overfit (can it learn at all?).
- **Add ablations/baselines that isolate the mechanism**: text-only editor (no audio) vs audio-conditioned — proves the acoustic prefix actually helps; 300M decoder **first** (12 GB feasibility model), 1B second; projector depth and source-layer sweep.
- **Add regression metrics**, not just WER: unchanged-token accuracy, S/D/I breakdown, and the **"made a correct draft token wrong"** rate (the failure mode that kills editors).
- **Rewrite the "open questions"**: tied embeddings / vocab / prefix-wiring are **resolved** (untied / shared / prefix). The genuinely open risks are **learnability, train-step memory, realised speed, draft-length coverage, and projector capacity.**

## Build order (revised after gates 1–2)

Reuse omni first — shared tokenizer + known speech path + Tajik evidence are too valuable to discard. Note there is **no "300M vs 1B decoder" choice**: the editor decoder is fixed at 1.22B; lift its weights from the strongest available v2 checkpoint (1B or 7B card). Status after gates 1–3: feasibility **passed** (draft-length coverage ~100%; one real train step fits at batch 1–2; forward path built and runs), copy **learns** (with the required head-tie), and **edit-beats-draft is the open question**. Next concrete steps, in priority order: (1) **richer acoustic conditioning** — replace the single reused Linear projector with a Q-Former / multi-layer adapter over stacked CTC encoder layers (gate 3 suggests the linear projector can't tell the editor what to fix); (2) a **real training recipe** — grad clipping, LR schedule, gentle CTC onset, larger batch (gate 2 says batch 1–2 + accum); (3) train on the **full v3** (100 examples proved copy, not correction); always with **tied head + copy warmup**. A clean small multilingual bidirectional LLM is the **fallback** only if a properly-conditioned, properly-trained omni editor still can't beat the CTC+KenLM baseline.

## References to pull official code from (lower the from-scratch risk)

- Paper (definitive method/loss/params): <https://arxiv.org/pdf/2603.08397>
- Apache-2.0 inference reference — crib ε-interleaving, bidirectional-mask construction, and the output-head / CTC-collapse shapes from here rather than re-deriving: <https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar/tree/main> (`modeling_granite_speech_nar.py`, `configuration_…`, `processing_…`)
- Closest open **training** reference for NAR ASR correction: FastCorrect (Microsoft NeuralSpeech), <https://github.com/microsoft/NeuralSpeech> — plus Levenshtein Transformer (in fairseq) for the insert/delete framing. Do **not** inherit Granite's tying/tokenizer/EOS assumptions from these.
- Our own ground truth: `omnilingual_asr/models/wav2vec2_llama/{model,config,factory}.py` (vendored in `.venv`), and `../lm_decoding/` + the EXPERIMENTS.md entry for the baselines.
