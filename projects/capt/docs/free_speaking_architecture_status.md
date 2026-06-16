# P016 Status — Two-Path Pronunciation Scoring

Last updated: 2026-06-01

## Goal

No-training, generalist pronunciation scoring that works for **most languages** — including the
majority that have **no L2 (learner) labeled speech database**, so a supervised scorer
(GOPT / MuFFIN / HiPPO, all trained on English SpeechOcean762) is not an option.

## The one funnel, two target sources

There is a single scoring funnel:

```text
text -> G2P -> canonical IPA   (the "answer key")
audio -> phone recognizer -> produced IPA
align the two -> PER / PFER
```

The **only** thing that changes between the two paths is where the answer-key text comes from:

- **read-aloud** — the target sentence is known (a prompt). G2P *that*. No ASR. The answer key
  is guaranteed correct, so this is the clean ceiling of the funnel.
- **free-form** — the target is unknown, so it is recovered by ASR (ElevenLabs Scribe v2) and
  G2P'd. The score is only as good as the ASR's word recovery; the residual limit is intent
  ambiguity (ASR can't tell "mispronounced the intended word" from "tried a different word").

We have ground truth (the reference text) for both, so running both on the same dataset shows
the funnel's ceiling (read-aloud) and what deriving the target via ASR costs (free-form).

## What's been tried (chronological)

1. **GOP / SpeechOcean762 detour.** Tried Goodness-of-Pronunciation scoring. ZIPA scored
   PCC ≈ 0.067 (near zero) — its 127-token char-level IPA vocab can't do per-phoneme GOP, and
   GOP itself needs L2-style per-phone calibration, the resource most languages lack. Dead end
   for the generalist goal. (See `docs/DECISIONS.md`.)
2. **Pivot to phone-recognition + feature distance.** Dropped GOP; scored recognized IPA vs
   canonical IPA by phonological-**feature** edit distance (PFER, via panphon). **ZIPA became
   the best lane by far** — on the FLEURS gate-100 check, RU PFER 0.062 vs XLSR-eSpeak 0.138,
   and RU PFER actually *below* EN. A universal multilingual phone recognizer is the right
   backbone, and it isn't English-biased. This matches the literature (TextPA, PRISM/PFER,
   Allosaurus, ZIPA — see `docs/DECISIONS.md`).
3. **Free-form was noisy in the Qwen era.** The old ASR was Qwen3-ASR-1.7B; free-form Russian
   looked diagnostic-only, dominated by ASR drift + target-G2P errors on function words /
   numerals — not pronunciation. A strong ASR is exactly the missing piece.

## Current state (this rebuild)

- **Removed (locked in):** the Gradio app, the Qwen ASR lane, the XLSR-eSpeak recognizer + all
  its lanes, the Charsiu G2P backend, the diagnostic lanes, and the single-file `analyze`/
  `cli.py` machinery. The project is now exactly the two-path dataset eval. Visual comes later.
- **ASR = ElevenLabs Scribe v2** (`capt.asr.ScribeAsrTranscriber`) via the
  `superwhisper-api` realtime stream. Free-form only; read-aloud needs no ASR.
- **Recognizer = ZIPA only** (universal IPA, ONNX), run **in-process** via onnxruntime (vendored in
  `capt.recognize`, no subprocess / no `third_party/` clone). Recognition depends only on the audio,
  so the eval recognizes **once** per sample and scores both target sources against it
  (`PronunciationComparePipeline.recognize()` + `.score_text()`).
- **G2P:** per-language **routed** backends (espeak-ng universal floor / Epitran / CharsiuG2P), with
  ZIPA-distilled FSTs for the no-G2P gap languages — see the routing decisions in `docs/DECISIONS.md`
  and the trainable-G2P record in `experiments/g2p_train/`. (The earlier Russian MFA lane is gone.)

## How to run

```bash
# 1. Build a multilingual FLEURS manifest (any FLEURS config codes; defaults to a diverse 10).
uv run capt-manifest --out-dir runs/two_paths --per-language 20 \
  --languages en_us ru_ru fr_fr de_de es_419 it_it fa_ir hi_in tr_tr ja_jp

# 2a. Read-aloud (local, no ASR/auth):
uv run capt-eval --manifest runs/two_paths/manifest.jsonl --out-dir runs/two_paths/eval

# 2b. Add free-form (needs superwhisper-api auth + network on this host):
uv run capt-eval --manifest runs/two_paths/manifest.jsonl --out-dir runs/two_paths/eval --free-form
```

Outputs: `summary.csv` (PER/PFER per id × mode × lane), `words.csv` (per-word
target_phones vs recognized_phones + sub/del/ins detail — the G2P diagnostic), `results.jsonl`,
and `report.md` (avg PER/PFER by language × mode × lane).

### Reading the G2P diagnostics

The point of the multilingual run is to see **where G2P is the bottleneck**, not the recognizer.
Per language, check `summary.csv` `g2p_warnings` and the `words.csv` `target_phones` column: if a
language's read-aloud PFER is high *and* its targets look wrong/empty, that's a G2P gap (missing
espeak voice, bad normalization), not a pronunciation signal. Read-aloud isolates this because its
target is the known-correct text.

## Open work

- **G2P internals:** the ruff complexity hotspots are resolved (`g2p/routing.py:_spoken_word_parts`
  split into `_russian_spoken_parts`/`_russian_decimal_parts`;
  `g2p/text_normalization.py:_russian_thousands_penalty` split into `_thousands_digit_penalty`/
  `_iz_thousand_adjustment`). `ruff check src tests` is clean. Per-language G2P correctness is what
  the read-aloud eval surfaces next.
- **Per-language G2P coverage:** espeak covers most FLEURS languages but quality varies (e.g. ja);
  the eval will rank where targets are trustworthy.
- **Phone-equivalence tiering & allophony:** exact / small-feature-distance / real-error tiers, and
  not penalizing legitimate allophonic variation (MixGoP idea) — the one robustness gap ZIPA/PRISM flag.
- **Visual:** rebuild after the funnels are trusted.
