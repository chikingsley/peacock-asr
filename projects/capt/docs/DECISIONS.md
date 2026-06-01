# Decisions & idea log

Append-only record of what we chose and why (and what we rejected), so we don't re-litigate.

## 2026-06-01 — Architecture: one funnel, two target sources
Score pronunciation as `text → G2P → canonical IPA` vs `audio → ZIPA universal phone recognizer`,
by panphon **PFER** (phonological-feature edit distance). Two paths share the funnel and differ
only in where the target text comes from: **read-aloud** (known reference, no ASR) and
**free-form** (ASR = ElevenLabs Scribe v2). ZIPA is the recognizer; it's solid across languages —
**G2P is the bottleneck.** Removed Gradio + Qwen.

## 2026-06-01 — Per-language G2P routing
No single G2P wins; it's per-language. `TargetG2P(backend="routed")` reads `g2p_routing.json`,
populated by `scripts/g2p_ablation.py` (recognize each FLEURS clip once with ZIPA, score the
reference against candidate G2Ps by PFER, pick the min). Candidates: **espeak** (universal floor),
**Epitran** (~70 langs), **CharsiuG2P** (~60, complementary — covers Epitran gaps + CJK). All 102
FLEURS languages routed (espeak ~33 / epitran ~37 / charsiu ~23). Surprise: Charsiu beats
espeak+epitran on de/fr.

## 2026-06-01 — Trained G2P for the 9 no-G2P gap languages
9 languages (ig_ng, wo_sn, ast_es, kam_ke, kea_cv, ln_cd, luo_ke, nso_za, umb_ao) have no viable
rule-based G2P. Fix: **distill a G2P from ZIPA itself** — run ZIPA on FLEURS audio, align to the
reference words, train a G2P on the (word→IPA) pairs (matches ZIPA's convention by construction).
Trial: Igbo went from *no target (1.0)* → **0.235**. **Phonetisaurus** (WFST, CPU, trivial install)
is the proof-of-concept trainer; **byT5** (neural) scored slightly better on identical data
(0.179/0.230 vs 0.209/0.235) and is the planned scale-up (small/base, on a free GPU). Distillation
does NOT beat existing routes on well-covered languages, so it's a gap-filler, not a replacement —
unless the improved (G2P-scaffolded monotonic) aligner closes that gap. Wired as the `trained`
backend; the 9 gaps route to it.

## 2026-06-01 — REJECTED (for now): MixGoP / reference-free scoring
**What:** score pronunciation without any G2P/answer-key by modeling the density of native (L1)
speech in SSL-feature space (per-phone GMMs) and scoring a clip's phone log-likelihood (Choi et al.
2025, arXiv:2502.07029). Prototype built + validated that the density model fits native Russian
(held-out native clips score tightly). Parked in `experiments/mixgop/` (isolated, never wired in).
**Why not pursued:** a typicality score isn't proven to measure *pronunciation quality* until it's
shown to **discriminate good vs bad** — and that can only be validated against **human L2
pronunciation labels**, which exist only for a few languages (English speechocean762). The
discrimination test was never run. Given the project's whole premise is *no L2 data*, validating a
second, harder-to-trust lane wasn't worth it versus improving the G2P lane that already works.
**Revisit if:** we want a fluency/holistic signal for free-form speech (where segmental G2P scoring
is weakest), or we decide to validate both lanes head-to-head on speechocean762 (PCC vs human
scores). The prototype + how-to-validate notes are in `experiments/mixgop/README.md`.

## 2026-06-01 — REJECTED: distilled-G2P as the universal backend (stays gap-only)
Tested whether scaffold-distilled Phonetisaurus should replace rule-G2P routing on covered
languages (the de_de=0.109 lead). Re-test across 10 covered langs (300-utt train, same held-out
eval): distilled wins only 2/10 (de_de, es_419 — both ties within 6-clip noise) and loses 8/10,
badly on ja (0.60 vs 0.39), ko (0.42 vs 0.13), hi (0.37 vs 0.19). Mean PFER routed 0.135 vs
distilled 0.220. **Decision: keep `trained` as a GAP-FILLER ONLY** — not universal, not even an
ablation co-candidate for covered languages. Rule G2Ps generalize better on covered langs;
distillation is limited by low eval-word reuse + noisy targets. byT5 (deferred to free GPU) would
improve distilled numbers but the high-value byT5 experiment is the **9 gap languages**, not
covered ones. Study: experiments/g2p_train/RESULTS_UNIVERSAL.md.
