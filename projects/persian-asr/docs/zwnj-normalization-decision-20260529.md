# Decision: ZWNJ Normalization for Persian ASR (Omni CTC)

- **Status:** Accepted
- **Date:** 2026-05-29
- **Scope:** Persian (Farsi) ASR training labels, evaluation scoring, and the Omni CTC model family (`omniASR_CTC_*_v2`, char tokenizer `omniASR_tokenizer_written_v2`).

## TL;DR

The acoustic model is trained and scored on a **ZWNJ-free canonical surface** (ZWNJ → space).
We do **not** add ZWNJ to the tokenizer vocabulary. ZWNJ is, if ever needed, restored
as a separate **text post-processing** step, never by the acoustic model.

This is applied in exactly one place — the shared `maybe_normalize()` normalizer — so it
flows consistently through training labels, evaluation references, and evaluation hypotheses.

## Context

The v3 Omni CTC run (`persian_asr_scribe_v3_max_20260527`) emits the unknown-token glyph
`⁇` (U+2047) heavily on messy datasets. Measured from the benchmark comparison DB:

| Model | Benchmark rows with `⁇` | Share |
|---|---:|---:|
| omni-v3 | 10,376 / 31,121 | ~33% |
| omni-exact | 0 | 0% |
| all other models | 0–6 | ~0% |

Root cause (verified locally, not inferred):

```
v3 training text contains ZWNJ (U+200C)
  -> Omni char tokenizer has no ZWNJ piece; ZWNJ PieceToId == unk_id (3)
  -> training teaches <unk> at Persian morpheme boundaries
  -> CTC decoder renders <unk> as " ⁇ "
```

Direct audit of the exported training parquet:

- exact-match export: 219,646 rows, **0** tokenizer-unknown rows
- v3 export: 563,749 rows, **107,742** tokenizer-unknown rows
- top unknown piece: ZWNJ, **209,194** occurrences (plus small counts of bidi/control chars)

So both exports were already normalized via `maybe_normalize()`, but the NVIDIA Farsi
normalizer leaves ZWNJ (and a few bidi/control chars) intact, and the Omni tokenizer
cannot represent them.

### What ZWNJ is

ZWNJ (U+200C, "zero-width non-joiner" / half-space) is an **invisible** character. Persian
letters join cursively; ZWNJ breaks that join without inserting a visible space. It is real,
meaningful orthography (plural `کتاب‌ها`, verb prefix `می‌خوام`, compounds `علاقه‌مند`).

The decisive property: **ZWNJ is inaudible.** `می‌خوام`, `می خوام`, and `میخوام` sound
identical. There is no acoustic signal for ZWNJ placement — it is a text/orthography
decision, not a speech decision.

## Decision

1. **Acoustic model trains and is scored on a ZWNJ-free surface.** Map ZWNJ → space (default),
   plus map the related bidi/control characters out. Keep the existing Omni tokenizer unchanged.
2. **Evaluation normalizes reference and hypothesis identically** (already true via the shared
   normalizer), so WER/CER measure acoustic accuracy rather than orthographic luck.
3. **ZWNJ restoration, if needed, is a separate text post-processing step** (rule-based or a
   small text model), applied to final output only. Out of scope for the acoustic model.

### Why not "just add ZWNJ to the tokenizer"

Mechanically, adding one class to a CTC char vocab is easy (resize the final linear layer,
init the new logit, fine-tune) and would not harm the other ~1600 languages. The objection is
not difficulty. It is that ZWNJ has **no acoustic correlate**: the audio does not tell the
model where ZWNJ belongs. A CTC model (near frame-independent) is especially weak at this kind
of context-only orthographic call, so a ZWNJ output token would be placed inconsistently —
no CER benefit, persistently noisy WER. It spends a vocab divergence and a retrain to make the
problem fuzzier, not cleaner.

### Why "restore as text post-processing" is the right home

ZWNJ placement is **recoverable from text alone** (~92% F1 with a Persian BERT sequence
labeler) and is **not** recoverable from audio. That asymmetry is the whole argument: it
belongs in a text model, not the ASR head. This also matches Meta's own NLLB SentencePiece
pipeline, which normalizes U+200C to a regular space during preprocessing.

### ZWNJ → space vs ZWNJ → remove (glue)

We use **ZWNJ → space**. It aligns morphemes to whitespace word units, matches the dominant
informal Persian writing, and matches the NLLB precedent. (Removing ZWNJ — gluing into
`میخوام` — is also tokenizer-safe but produces non-standard glued forms; not chosen.)
The hard requirement is **consistency**: the same transform applies to training labels,
eval references, and eval hypotheses. Which variant matters less than applying one variant
everywhere.

## Implementation

Single change in the shared normalizer:

`src/persian_asr_dataset/vendor/nvidia_stt_fa_fastconformer_hybrid_large.py`

Add ZWNJ and the related bidi/control characters to `REPLACEMENTS` (mapped to a space),
*before* the existing `NFKC` + whitespace-collapse steps. NFKC does not strip ZWNJ, and
`str.split()` does not treat ZWNJ as whitespace — which is exactly why it currently survives
into the parquet. Mapping it to a space first lets the final `" ".join(text.split())` separate
the morphemes and collapse any doubled spaces.

Characters to map to space (from the local audit): `U+200C` (ZWNJ), plus `U+200D` (ZWJ),
`U+200E` (LRM), `U+200F` (RLM), `U+FEFF` (BOM/ZWNBSP), `U+FFFD` (replacement char).

Because `maybe_normalize()` is the single chokepoint, this propagates automatically to:

- **training labels** — `src/persian_omnilingual_asr/dataset_prep/curated.py:210`
- **eval reference** — `src/persian_omnilingual_asr/benchmarks/omni.py:88`, `benchmarks/asr.py:364`
- **eval hypothesis** — `src/persian_omnilingual_asr/benchmarks/omni.py:89`, `benchmarks/asr.py:365`
  (then `jiwer.wer/cer` on the normalized pair)

### Hard gate before training

Add a preflight that fails any Omni export/train when training `text` produces tokenizer
unknowns. Require `unk_rows == 0`. The audit logic already exists:

```
uv run persian-omni-text-audit <export-name>
```

Wire it as a gate in the export path rather than a manual step.

### Roll-out

- **Training fix:** change `maybe_normalize` → re-export the v3 parquet → confirm `unk_rows == 0`
  → short probe train + benchmark, require near-zero `⁇` on youtube/mana/neyshekar/worldspeech
  → full run. The current v3 checkpoint stays out of the next parent-checkpoint choice until
  this passes.
- **Eval fix is cheap:** the benchmark DB stores raw `reference` and `hypothesis`, so corrected
  WER/CER can be recomputed by re-running `maybe_normalize` + `jiwer` over existing rows — no
  GPU, no model rerun. This deflates the `high_wer_low_cer` scoring inflation on the current
  scoreboard.

## Open item (downstream-dependent)

Whether to build the ZWNJ-restoration post-processor depends on the end use:

- **WER benchmark / rough transcripts** → ZWNJ → space everywhere, no restorer needed.
- **Publication-quality Persian text** → add a restoration pass (rule-based or BERT-style)
  as a final text step. The acoustic model still stays ZWNJ-free.

## Sources

- Joint Persian Word Segmentation Correction and ZWNJ Recognition Using BERT (COLING 2020) — ZWNJ recoverable from text at ~92% F1: https://arxiv.org/abs/2010.00287
- Correcting Space and ZWNJ Errors in Persian Text (RANLP 2025): https://acl-bg.org/proceedings/2025/RANLP%202025/pdf/2025.ranlp-1.40.pdf
- PSRB: A Comprehensive Benchmark for Evaluating Persian ASR Systems — ZWNJ inflates Persian WER, calls for linguistic normalization: https://www.themoonlight.io/en/review/psrb-a-comprehensive-benchmark-for-evaluating-persian-asr-systems
- Omnilingual ASR (Meta, 2025): https://arxiv.org/html/2511.09690v1
- Hugging Face ASR evaluation (WER vs CER behavior): https://huggingface.co/learn/audio-course/en/chapter5/evaluation
- JiWER usage: https://jitsi.github.io/jiwer/usage/
- CER for multilingual ASR evaluation: https://arxiv.org/abs/2410.07400
