# Parakeet TDT → Tajik — plan, findings & status

Fine-tune NVIDIA **Parakeet TDT-0.6B-v3** (FastConformer + Token-and-Duration Transducer) to Tajik, to get **CTC-class accuracy at transducer speed** (~thousands× RTFx vs our CTC's ~hundreds×). The twist: TDT new-language fine-tuning is a **known, unresolved failure** (the TDT head stalls at ~0.8 WER while CTC trains) — so this experiment is as much about **validating/fixing the community recipe** as about Tajik. Single source of truth: **status/TODO** top, **why**, the **problem + recipe** (with provenance flags), **design**, a **gated ablation plan**, **risks**, **references**. Same method as `../nar/` (sources verified, ablate end-to-end, cheap gates first).

## Status & outstanding

**Where we are (2026-06-16).** Planning only — nothing run. Prior art verified against primary sources (NeMo issues/discussions, source, HF card). Our Farsi `finetune_parakeet` package is largely reusable; one real gap (parquet→manifest). Branch TBD.

**The bet:** v3's encoder already knows **Cyrillic acoustics** (it's trained on Russian/Ukrainian/Bulgarian — though *not* Tajik), so a Tajik fine-tune could **beat our omni CTC 300M (FLEURS 16.9) and approach CTC+KenLM (14.5)** at **~10× the decode speed** — *if* we can get the TDT head to actually train.

**The risk in one line:** multiple users (incl. on v3) report the TDT head stuck ~0.8 WER while CTC learns; a maintainer acknowledged a real `change_vocabulary` loss-init bug but the fix never merged (issue #14140 open). **This experiment's gate 0 is to reproduce that failure, then beat it.**

**Outstanding, in order:** (0a) reproduce the TDT-stagnation + loss-init bug on the cheap 110M-hybrid; (0b) v3 inspection preflight (tokenizer coverage on Tajik, blank/duration indices, aux-CTC-head?); (1) parquet→NeMo-manifest builder; (2) Tajik tokenizer; (3) clean-controls ablation (loss-fix in all trainable arms; extend vs restore split); (4) scale the winning arm to v3; (5) eval WER (TDT head) + **RTFx** vs the omni+KenLM baseline.

## Why (the numbers that motivate it)

| model (ours unless noted) | FLEURS WER | RTFx | notes |
|---|---|---|---|
| omni CTC 300M Tajik v3 (production) | 16.9 / **14.5** +KenLM | ~450× | multilingual base; current best |
| our Parakeet CTC 110M (Farsi) | 17.3 | ~590× | English base, fine-tuned; ~1.3× faster than omni |
| **Parakeet TDT 0.6B v2/v3 (NVIDIA, English/EU)** | ~6.3 avg | **~3,300×** | the speed prize — transducer + FastConformer |

TDT is the architecture behind the "ridiculous RTFx" (≈3,300× batched, NVIDIA's #1 on the Open ASR leaderboard). Our CTCs are already fast (hundreds×) but TDT is ~6–10× faster *and* v3 is a stronger, multilingual 600M encoder. **Target: Tajik TDT that beats omni CTC on WER and runs at transducer speed.** (Realistic ceiling per prior art: match/beat CTC+KenLM; not a miracle.)

## The problem + the community recipe (provenance-flagged)

**TDT new-language fine-tuning is broken in a specific, documented way.** Stock NeMo `change_vocabulary()` on a transducer **fully reinitializes the decoder + joint** (verified in `rnnt_models.py`: it rebuilds both from config and discards pretrained weights) — and for **TDT** it mis-sizes the loss: `RNNTLoss(num_classes = joint.num_classes_with_blank - 1)` **omits `- num_extra_outputs`** (the duration outputs). Symptom (issue **#14140**, OPEN, reproduced by ≥3 users incl. on **v3**): the TDT head's WER stalls ~0.8 while the CTC head trains to ~0.1. A NeMo team member (hainan-xv): *"that is indeed a bug"* — but the fix PR **#14155 was closed, NOT merged**. **This is the central thing to fix/validate.**

**The community recipe** (discussion **#14728**, *single community author `lee-onidas`, no maintainer reply, no published WER — treat as hypothesis*): instead of letting `change_vocabulary` discard everything, **extend** the tokenizer (keep pretrained IDs `0..N-1`, append new-language IDs `N..`), then **manually restore** decoder/joint/CTC weights, leaving only the *new* vocab rows randomly-init near-zero; and **freeze the encoder during warmup**, unfreeze after (he reports rapid convergence + code-switching by ~5k steps on the 0.6b; the 110M needed the freeze, the 0.6b mostly didn't). Caveat he hit: over-training degrades numbers/acronyms.

**Maintainer-backed (issue #6895, titu1994 = NeMo ASR lead — authoritative):** for the tokenizer, **SPE-BPE is fine** (unigram not worth the cost), you can train it on **more text than the audio corpus**, and at vocab 1024 *"tokenizer tuning has no effect on WER at that scale."* (v3's unified vocab is **8,192**, so "append above ID 8191".)

So the experiment = **{stock change_vocab} → {+ loss-init fix} → {+ extend-tokenizer & weight-restore} → {+ encoder-freeze warmup}**, measuring at each step whether the TDT head escapes the ~0.8 stall. That's a clean ablation that both *validates* the community recipe and *isolates* which piece (or the acknowledged bug fix) actually matters.

## Design — what we build & reuse

```text
omni-parquet (v3 Tajik)  ──►  [parquet→NeMo manifest]  ──►  train/dev/test JSONL (audio_filepath/text/duration)
transcripts ─► [train_tokenizer]                       ──►  Tajik tokenizer (extend v3 8192, or fresh BPE)
parakeet-tdt-0.6b-v3 (download) ─► [extend vocab + RESTORE decoder/joint weights + loss-init fix]
                                 ─► [encoder-freeze warmup → unfreeze] ─► fine-tuned Tajik TDT
eval: WER (TDT head vs CTC/ref) + RTFx  vs  omni CTC+KenLM baseline
```

**Reuse from `projects/farsi-asr/src/finetune_parakeet/` (verified language-agnostic):** `finetune_parakeet.py` (wraps NeMo `speech_to_text_finetune.py`, `+init_from_pretrained_model=...`), `train_tokenizer.py` (wraps vendored `process_asr_text_tokenizer.py`), the vendored NeMo recipes + `convert_nemo_asr_hybrid_to_ctc.py`, the Lhotse bucketing config (`batch_duration=700, num_buckets=30` — proven stable), and `speech_to_text_finetune.yaml`. The 110M **hybrid** + **CTC** `.nemo` bases are on disk (good for the cheap gate-0 dry run); **v3 0.6b is not — download it** (`from_pretrained("nvidia/parakeet-tdt-0.6b-v3")`).

**Build new (the gaps):** (1) **omni-parquet → NeMo manifest** for Tajik — Farsi's `export_nemo_manifest.py` needs a SQLite ledger we don't have for Tajik; write a lightweight converter over `omni_finetune_core.parquet.iter_split()` (v3 `corpus=*/split=*/language=tgk_Cyrl`) emitting FLAC + JSONL. (2) Tajik text normalizer (or reuse omni-curator `normalize`). (3) The **vocab-extend + weight-restore + loss-init-fix** code (the recipe is a forum post, not a script — this is the real engineering, cribbed from #14728 and the #14155 diff). (4) A **TDT** training config (NeMo `conf/.../conformer_tdt_bpe.yaml` as template; Farsi only had CTC configs).

**Target sequencing:** validate the recipe on the **cheap 110M-hybrid first** (we have it; and hybrid has a CTC head as a built-in sanity baseline) → then the **v3 0.6b** (the prize; bigger, Cyrillic prior). Note **open question:** is v3 transducer-*only* (no aux CTC head)? If so, the "CTC fallback is free" safety net exists only on the 110M-hybrid / parakeet-ctc-0.6b, not within v3 — confirm at gate 0.

## Plan — gated ablation (cheapest-first, kill-gates)

**Question we can actually answer** (narrowed per review — we can't *reproduce* the unbenchmarked single-author claim, only test our own build): *does our implementation of {loss-init fix + extend-tokenizer + decoder/joint restore + optional encoder-freeze} make v3's TDT head train on Tajik and beat stock under our eval?*

- **Gate 0a — reproduce the failure (cheap, 110M-hybrid we own).** Tiny Tajik manifest (~100–500 rows), stock `change_vocabulary` + fine-tune; confirm the TDT head stalls ~0.8 while CTC trains, and that the loss-init bug (`num_extra_outputs`) is present in our vendored NeMo. **Monitor `val_wer` (TDT) AND `val_wer_ctc` separately** — the stall was historically masked by watching the wrong head. Kill-criterion: if stock already trains fine in our NeMo version, the bug's fixed upstream → skip to plain fine-tune.
- **Gate 0b — v3 inspection preflight (before *any* v3 training).** Load `parakeet-tdt-0.6b-v3`: audit its 8,192 SP tokenizer's character coverage on Tajik text, read the blank-id + TDT duration-output indices (the restore code depends on these), and confirm **whether v3 has an aux CTC head** (decides if the CTC fallback lives inside v3 or is a separate model).
- **Gate 1 — data.** Build the parquet→manifest converter; Tajik train/dev/test JSONL from v3 (FLEURS + conversational). Verify durations/text.
- **Gate 2 — tokenizer.** Extract Tajik transcripts; build two ways (extend v3-8192 keeping 0..8191 + append Tajik; vs fresh BPE-1024 char-coverage 0.9995). A/B later.
- **Gate 3 — clean-controls ablation (the crux).** Each arm a separate run; **the loss-init fix is in every *trainable* arm** (stock is just broken), and tokenizer-extend is split from weight-restore so each lever is isolated:
  - **A0** stock `change_vocabulary` — *broken negative control* (expect stall).
  - **B** + loss-init fix — minimal-correct baseline (fresh decoder/joint, fixed loss).
  - **C1** B + **extend-tokenizer only** (keep pretrained rows, no weight restore).
  - **C2** C1 + **decoder/joint/ctc weight restore** (the recipe's core claim).
  - **D** C2 + **encoder-freeze warmup → unfreeze**.
  - *Control tokenizer variance:* B and C1 use the **same** Tajik tokenizer-training text + settings; the only delta is fresh-vocab (B) vs extended-onto-pretrained-rows (C1), so the lever is clean.
  - Run on **110M-hybrid first** (cheap; CTC head = sanity floor), then the winning arm(s) on **v3**. Metric: does the **TDT-head** val WER break below ~0.8 toward CTC-competitive, and which single lever flips it. **Always read TDT-head and CTC-head WER separately** (the historical failure is "CTC learns while TDT stalls" — aggregate val_wer hides it).
- **Gate 4 — eval + the payoff.** Best config: WER (TDT head) on FLEURS test (599) + conversational held-out (1,625) vs omni CTC (16.9) and CTC+KenLM (14.5); and **measure RTFx** the same way as the lm_decoding baseline (is it really ~thousands×?). Plus the #14728 regression checks (numbers/acronyms after bilingual fine-tune).
- **Separate CTC track (not a "v3 fallback"):** if the TDT head won't train despite B–D, a **Parakeet-CTC** model (110M-hybrid's CTC head — proven on Farsi — and/or `parakeet-ctc-0.6b`) reuses all the gate-1/2 data+tokenizer work to still ship a fast CTC Tajik model. Frame as a parallel track, *not* a result of the v3-TDT experiment (v3 may be TDT-only).

## Risks / open questions

- **TDT stagnation may be fundamental, not a tuning gap.** It's reproduced on v3 by others and the upstream fix didn't merge. If B–D don't break the stall, that's the (valuable, publishable) finding — and the fallback is CTC.
- **v3 tokenizer is 8,192 (unified), not 1,024.** The recipe's "append above 1023" becomes "above 8191"; the restore code must handle blank-id + the TDT duration outputs correctly (this is exactly where the #14155 bug lives).
- **v3 may be transducer-only** → no free CTC fallback inside it; confirm at gate 0.
- **Cyrillic prior is partial** — v3 knows Ru/Uk/Bg acoustics, not Tajik phonology; still a far better start than the English-only 110M, but not a guarantee.
- **Data → manifest is new code** (no Tajik ledger); and **RTFx must be measured the same way** as the lm_decoding/omni baselines for a fair speed comparison.
- **"Improve the community thing"** = implement the loss-init fix properly + monitor both heads + (if it works) write it up / contribute back — but only after we've validated it beats stock.

## References (provenance-flagged)

- **Community recipe (hypothesis, single author, unbenchmarked, built on 110M/v2):** NeMo discussion #14728 — <https://github.com/NVIDIA-NeMo/NeMo/discussions/14728> (extend tokenizer + restore decoder/joint + encoder-freeze warmup).
- **The TDT bug (OPEN, maintainer-acknowledged, fix unmerged):** issue #14140 — <https://github.com/NVIDIA-NeMo/NeMo/issues/14140>; closed-unmerged fix PR #14155 (the `RNNTLoss num_classes - num_extra_outputs` correction).
- **Tokenizer guidance (authoritative — NeMo ASR lead):** issue #6895 — <https://github.com/NVIDIA-NeMo/NeMo/issues/6895> (SPE-BPE fine; train on extra text; no WER effect at vocab 1024).
- **Tokenizer script (Apache-2.0, vendored):** `scripts/tokenizers/process_asr_text_tokenizer.py` — <https://github.com/NVIDIA-NeMo/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py>.
- **NeMo fine-tune docs / configs:** <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html> (`init_from_pretrained_model`, `model.tokenizer.update_tokenizer`, decoding strategies; TDT specifics live in source, not the docs prose).
- **Model card (official):** parakeet-tdt-0.6b-v3 — <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3> (FastConformer+TDT, 8,192 unified SP, 25 EU langs incl. ru/uk/bg, CC-BY-4.0). TDT paper: arXiv 2304.06795.
- **Our prior art:** `projects/farsi-asr/src/finetune_parakeet/` (scripts, vendored NeMo, 110M bases, Lhotse config) + `projects/farsi-asr/EXPERIMENTS.md` (Parakeet CTC 844h Farsi run, the TDT-drop decision); `projects/farsi-asr/src/farsi_asr_dataset/cli/export_nemo_manifest.py` (manifest builder to adapt). Our baselines: `../lm_decoding/` + the tajik `EXPERIMENTS.md`.
