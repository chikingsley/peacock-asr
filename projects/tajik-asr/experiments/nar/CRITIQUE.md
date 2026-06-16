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

## What the spec gets right

1. **The core insight is the strongest part.** Swapping IBM's English-centric Granite for omni's natively-multilingual `llama_decoder` directly targets NLE's one documented failure mode. Omni has real Tajik (the 10.9 zero-shot is evidence the *weights* know the language).
2. **Reusing real components instead of coding from scratch.** `encoder_proj` exists, the prefix wiring exists, the shared tokenizer exists — much of the scaffolding is pretrained.
3. **Honest, well-grounded measurement plan.** The four-way table compares against baselines we already have real numbers for (greedy 16.9, KenLM 14.5, AR ~10.9), with RTFx measured the same way as the `lm_decoding` experiment.
4. **The training trick is correctly understood.** CTC marginalisation → plain (audio, text) pairs, no edit-op labels. This is what makes it trainable on our existing v3 export.

## Corrections (spec says X; the truth is Y)

1. **Tied embeddings — the body contradicts the code.** README line 57-58 says "residual identity + tied embeddings make copying the default." The omni decoder is **untied**. That claim is inherited from the pre-pivot Granite plan (Granite 1B *is* tied). So the copy bias rests on **residual identity + the L_CR regulariser alone** — weaker than implied. Don't blindly tie the weights after pretraining (it can break the learned output head); prefer a copy warmup, a tuned/larger λ, or an explicit copy gate. Move this from "open question" to "settled finding: untied, mitigate accordingly."

2. **Vocab mismatch is overstated.** README risk #2 frames a char-vs-Llama re-tokenization round-trip. The omni family **enforces** equal vocab (shared tokenizer); with matching v2 variants the draft ids are already Llama ids. Downgrade this. The *real* residual risks here are narrower: EOS-as-ε double duty (`add_eos`/loss handling in `model.py`), the CTC-blank convention, and reference tokenisation/normalisation.

3. **"Single-digit WER on read" is too optimistic.** NLE *approximates* its AR teacher and even *loses* to it on the harder multilingual case. The realistic ceiling here is bounded by the 300M CTC's acoustic features and the 16.9 draft. Keep the table target (**≤ CTC+KenLM 14.5**); demote "single-digit" / approaching the 10.9 AR to an explicit **stretch goal**.

4. **Param accounting is inherited, not recomputed.** "LoRA rank ~128" and "~14M" are *both* IBM's **base-NLE** numbers (they're self-consistent — base NLE is rank 128 totalling 14M; NLE++ is rank 160 totalling ~280M). The problem isn't a mismatch between the two — it's that they're IBM's numbers for **IBM's architecture** (Granite decoder + Q-Former projector), and this build differs: a 12-layer omni decoder and a reused **Linear** `encoder_proj` (no Q-Former). So 14M almost certainly won't be our number. Recompute the exact LoRA param count for the chosen rank over our decoder's attn+MLP layers, and report it **separately** from activation memory.

5. **"The 10.9 zero-shot is *it* (the decoder)" is misleading.** That score is produced by the **full omni speech-LLM** — its own encoder + projector + decoder, trained together. Lifting only the decoder and feeding it a *300M-CTC*-derived prefix is a different system. The 10.9 proves the language is in the family, not that the lifted-decoder + our-CTC stack reaches it.

## Risks the spec omits (the important new findings)

- **Draft-length bound — the sharpest unstated risk.** NLE can only **insert ≤ N+1** tokens; the CTC loss runs over `2N+1` positions. If the reference token length exceeds `2N+1` (heavy deletions in the draft), the loss is unreachable / pathological for that example. **Measure the `ref_len > 2N+1` rate on v3 before training** — likely fine for a char-level read-speech draft, but conversational (where the draft drops words) is where it could bite.
- **"Fits the 12 GB card" is unproven.** Freezing the body removes optimiser state, but LoRA still backprops through the **full bidirectional decoder activations** over `audio_prefix + (2N+1)` positions, with no KV-cache savings at train time. Memory feasibility is a **gate**, not a footnote — measure one real train step.
- **"Near-CTC speed" overclaims.** One bidirectional 1B forward pass can beat AR decoding but will not resemble CTC+KenLM throughput. Reframe the headline as **"clearly faster than the AR LLM"** (the table already does this — make the prose match).
- **De-causalising is a distribution shift, not just a mask flip.** The decoder was trained for next-token causal prediction; producing same-position edit logits with an untied head is off the training distribution. The LoRA has to absorb that shift — budget for it (rank, warmup).
- **Projector downgrade.** A single Linear over (presumably) final-layer CTC features is poorer than IBM's Q-Former over stacked layers 4/8/12/16. And the omni `encoder_proj` was trained on the *omni-LLM's* encoder features, not our fine-tuned CTC's. Reword to "initialise if shape-compatible, **expect to retrain**", and make projector depth + source-layer selection a **central ablation**, not an afterthought.

## Suggested spec changes

- **Add feasibility gates before full training** (cheap, fail-fast): (a) `ref_len ≤ 2N+1` coverage rate on v3; (b) memory for one real train step at the target rank; (c) no-grad RTFx of the forward path; (d) identity-copy overfit (does an untrained pass at least copy?); (e) 100-example overfit (can it learn at all?).
- **Add ablations/baselines that isolate the mechanism**: text-only editor (no audio) vs audio-conditioned — proves the acoustic prefix actually helps; 300M decoder **first** (12 GB feasibility model), 1B second; projector depth and source-layer sweep.
- **Add regression metrics**, not just WER: unchanged-token accuracy, S/D/I breakdown, and the **"made a correct draft token wrong"** rate (the failure mode that kills editors).
- **Rewrite the "open questions"**: tied embeddings / vocab / prefix-wiring are **resolved** (untied / shared / prefix). The genuinely open risks are **learnability, train-step memory, realised speed, draft-length coverage, and projector capacity.**

## Build order (unchanged recommendation, sharpened)

Reuse omni first — shared tokenizer + known speech path + Tajik evidence are too valuable to discard. Start with the **300M decoder** as the 12 GB feasibility model; go to 1B only after. A clean small multilingual bidirectional LLM is the **fallback** if de-causalising the omni decoder fails to learn, not the first move.

## References to pull official code from (lower the from-scratch risk)

- Paper (definitive method/loss/params): <https://arxiv.org/pdf/2603.08397>
- Apache-2.0 inference reference — crib ε-interleaving, bidirectional-mask construction, and the output-head / CTC-collapse shapes from here rather than re-deriving: <https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar/tree/main> (`modeling_granite_speech_nar.py`, `configuration_…`, `processing_…`)
- Closest open **training** reference for NAR ASR correction: FastCorrect (Microsoft NeuralSpeech), <https://github.com/microsoft/NeuralSpeech> — plus Levenshtein Transformer (in fairseq) for the insert/delete framing. Do **not** inherit Granite's tying/tokenizer/EOS assumptions from these.
- Our own ground truth: `omnilingual_asr/models/wav2vec2_llama/{model,config,factory}.py` (vendored in `.venv`), and `../lm_decoding/` + the EXPERIMENTS.md entry for the baselines.
