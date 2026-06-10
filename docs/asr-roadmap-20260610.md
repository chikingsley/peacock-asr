# ASR Roadmap, 2026-06-10

Review of the full fine-tuning history in this repo (Persian, Tajik, Georgian) against the
published literature, answering one question per language: is the model maxed out, and if not,
what are the highest-value next experiments. All repo numbers cite files by path; all external
numbers cite URLs. Hard constraints respected throughout: single ~12 GB GPU, machine labels
(Scribe ensemble + LLM fusion) for scraped data, no native-speaker annotators, free unlimited
Scribe/SuperWhisper API.

## Verdict summary

| Language | Current best | Verdict | Highest-value next lever |
|---|---|---|---|
| Persian | omni CTC 300M scribe-v4-rewarm: FLEURS 8.51 / CV 19.41 / YouTube 20.19 WER (`projects/persian-asr/benchmarks/suites/canonical-tests-scribe-v4-rewarm-20260530/summary.tsv`) | Maxed on the training side for this data; NOT maxed on the decoding side | n-gram LM fusion + beam decoding (untried, standard, ~5 to 30 percent relative in the literature) |
| Tajik | omni CTC 300M v3: FLEURS 17.2 / conversational held-out 37.65 WER (`projects/tajik-asr/EXPERIMENTS.md`) | Far from maxed; ~20 WER points of relative headroom to the teacher-agreement floor | More conversational data from NEW channels (diversity, not just hours), plus a label-quality iteration |
| Georgian | Untrained; 145.3 h gold v0 export ready (`projects/georgian-asr/data/datasets/v0/export_summary.json`) | Not started; literature says the existing gold data is already enough for a strong model | Run the v0 fine-tune now (gpu_max preset); YouTube scraping only pays if conversational Georgian is a target |

The owner's suspicion on Persian is mostly right but needs one qualifier: every TRAINING lever
on the existing data is exhausted (the re-warm experiment returned noise, the 1B lost), but the
project has never decoded with anything except greedy CTC, never averaged checkpoints, and never
scraped Persian conversational audio with the newer Tajik-grade pipeline. Those three are real,
untried levers.

---

## 1. Persian

### Where it stands

Production model: omni CTC 300M `scribe-v4-rewarm` (step_7000, dev WER 11.15 per `TODO.md`).
Six-split canonical WER (`projects/persian-asr/benchmarks/suites/canonical-tests-scribe-v4-rewarm-20260530/summary.tsv`):

| split | WER | CER |
|---|---|---|
| common_voice_25 | 19.41 | 4.71 |
| fleurs | 8.51 | 2.36 |
| mana_tts | 6.61 | 1.79 |
| neyshekar | 8.22 | 1.86 |
| worldspeech | 27.61 | 17.30 |
| youtube | 20.19 | 9.66 |

Macro WER 15.1. The commercial teacher, Scribe v2, scores macro 21.9 on the same harness
(CV 31.28, FLEURS 9.90, mana 14.54, neyshekar 15.17, worldspeech 31.30, youtube 29.23;
`projects/persian-asr/benchmarks/suites/canonical-tests-scribe-v2-20260525/summary.tsv`).
The student beats its teacher on all six splits. That is the single most important fact for the
"maxed" question: agreement-filtered training data produced a model better than any single pass
of the labeler, so iterating the labeler (a "Scribe v5") on the SAME audio cannot add much.

### Against published Persian numbers

- PSRB, the most thorough Persian benchmark (10.4 h, 58.6 percent spontaneous speech, expert
  double-annotated): best commercial system Avanegar WER 19.30 / CER 8.75; Google Chirp v2
  19.92; best open model Faster-Whisper 33.93; whisper-large-v3 41.49; Seamless 38.85
  (Table 4, <https://arxiv.org/pdf/2505.2123>0).
- Whisper-large on Persian YouTube podcast audio (oracle segmentation): mean WER 32.5; on a
  noisy entertainment channel far worse (<https://aclanthology.org/2026.silkroadnlp-1.13.pdf>).
- Meta's own 7B LLM-ASR scores CER 3.4 for `fas_Arab` on their benchmark with 214.7 training
  hours (<https://raw.githubusercontent.com/facebookresearch/omnilingual-asr/main/per_language_results_table_7B_llm_asr.csv>);
  our 300M CTC sits at FLEURS CER 2.36 (different test set, but same order).
- Qwen3-Omni-30B, the strongest model on PARSA-Bench, reports mean WER 0.358 on its Persian
  ASR tasks (<https://arxiv.org/pdf/2603.14456>).

Read: a 300M model scoring 20.2 WER on real YouTube Persian and 8.5 on FLEURS is at or above
the best published commercial level for real-world Persian, and far above every open-source
system PSRB measured. There is no published Persian system this repo is visibly chasing.

### The 1B post-mortem (verified)

The 1B lost to the 300M on every split: CV 24.27 vs 19.41, FLEURS 10.60 vs 8.51, mana 10.80 vs
6.61, neyshekar 13.12 vs 8.22, worldspeech 30.07 vs 27.61, youtube 26.21 vs 20.19
(`canonical-tests-omni-ctc-1b-v2-scribe-v4-20260602/summary.tsv`). Best dev WER 16.86 vs the
300M's 11.27. Three confounded handicaps, all visible in
`projects/persian-asr/src/persian_omnilingual_asr/training/configs/persian-asr-scribe-v4-ctc-1b-v2.yaml`:

1. Pure-bf16 AdamW (mixed_precision mode "off", no fp32 master copy) to fit 12 GB. bf16
   optimizer state costs accuracy; this is the known price of the memory fit.
2. Half the effective batch: 960k elements x grad-accum 4 = 3.84M elements per update vs the
   300M's 3.84M x 2 = 7.68M, plus a 30 s audio cap.
3. Fresh start from the upstream base, while the 300M v4 continued from the exact-match parent
   (an extra 34k-step curriculum the 1B never got).

The literature says capacity should win: OWLS finds WER scales as a power law in model size for
pretrained models and that 1B to 9B cut low-resource WER from 59 to 45
(<https://arxiv.org/pdf/2502.10373>). So "1B is worse" is not a law; "1B under pure-bf16 with half
the batch and no curriculum is worse" is what was measured. A retry is only worth it if the
optimizer constraint is removed: 8-bit optimizer states or LoRA adapters (frozen bf16 base, tiny
optimizer) would restore an fp32-quality update within 12 GB. Expected gain over the 300M even
then: 1 to 2 macro WER points at best, at 3 to 4x the inference cost. Verdict: dead unless the
300M plateaus on a future, larger corpus AND fairseq2 LoRA/8-bit integration is cheap. Do not
re-run the same recipe.

### Remaining levers, ranked

1. **LM fusion + beam decoding. Expected: 5 to 15 percent relative on CV/YouTube; cost: days,
   CPU only; confidence: high that it helps, medium on size.** The eval path is greedy argmax
   (`packages/omni-finetune-core/src/omni_finetune_core/recipe/wer_calculator.py`; beam search
   exists only for the LLama decoder variants). The omni CTC tokenizer is character-level, which
   is exactly what pyctcdecode + KenLM supports
   (<https://github.com/kensho-technologies/pyctcdecode>). A Swedish XLS-R + 5-gram showed ~30
   percent relative improvement on Common Voice (<https://huggingface.co/blog/wav2vec2-with-ngram>);
   gains shrink for stronger acoustic models, so expect the low end. Persian text for the n-gram
   is abundant (e.g. the naab corpus, <https://huggingface.co/datasets/SLPL/naa>b). Bonus: PSRB
   identifies word-boundary/ZWNJ placement as the top Persian error class
   (<https://arxiv.org/pdf/2505.21230>, section 5.1.1), and a word-level LM directly attacks
   word-boundary errors, which is also where the model loses to itself on CER vs WER.
2. **Checkpoint averaging. Expected: 1 to 2 percent relative; cost: hours; confidence: medium.**
   The regime already keeps best-3 checkpoints (`packages/omni-finetune-core/src/omni_finetune_core/presets.py`,
   `keep_best_n_checkpoints=3`) and ships a single one. Averaging best-N is standard practice for
   transformer ASR with roughly 1 to 2 percent relative WER reduction
   (<https://caiman-asr.myrtle.ai/training/checkpoint_averaging.html>). Free to test on the
   existing run directory.
3. **Conversational-scale scrape with the Tajik-grade pipeline. Expected: 1 to 3 points on the
   youtube split, little on the clean splits; cost: weeks of wall-clock, zero dollars;
   confidence: medium.** The v4 corpus is 1,032 h of the OLD Persian scrape; the rebuilt
   omni-curator channel pipeline (41 channels, 1,826 h for Tajik per `TODO.md`) was never run
   for Persian. Tajik proved the conversational lever is invisible on read-speech benchmarks and
   large on conversational ones. Persian's weak splits (worldspeech 27.6, youtube 20.2) are the
   conversational ones. Prerequisite per `TODO.md` phase 2: freeze a Persian held-out manifest
   first.
4. **Longer training / re-warm: dead, measured twice.** v4 re-warm at lr 2e-6 for 10k steps
   produced sub-0.2-point mixed deltas (`projects/persian-asr/EXPERIMENTS.md`). Dev WER was
   flat. Do not spend GPU here.
5. **Label iteration (Scribe v5) on existing audio: low value.** The student already beats the
   teacher on all six gold splits. New labels only matter for NEW audio (lever 3).
6. **Whisper-large-v3 fine-tune on 12 GB: feasible via LoRA/PEFT, not recommended.** Whisper
   base Persian is weak (41.5 WER on PSRB) and hallucination-prone on Persian (PSRB section
   5.3); published CV-17 fine-tunes reach roughly 13 to 14 percent on Common Voice test
   (<https://huggingface.co/MohammadGholizadeh/whisper-large-v3-persian-common-voice-17>), which
   is on clean read speech with seen-domain training and does not beat the current model
   broadly. 15x the inference cost of the 300M CTC. Skip.
7. **omniASR-LLM variants: not trainable here.** The LLM decoder adds 1.2B parameters on top of
   the encoder (<https://arxiv.org/html/2511.09690v1>); even the smallest LLM variant exceeds what
   pure-bf16 tricks fit for TRAINING on 12 GB, and the 1B CTC experiment already showed what the
   memory-starved regime does. Inference-only use as a second verifier is the one cheap use.

**Persian verdict: maxed for the model class on the current data and training recipe. Two cheap
non-training levers (LM fusion, checkpoint averaging) and one expensive data lever
(conversational scrape) remain. Everything else is dead or negative-expected-value.**

---

## 2. Tajik

### Where it stands

v3 (300M, 1,070 h, held-out-safe): conversational held-out 37.65 WER / 14.04 CER, FLEURS test
17.2; v0 (5.8 h) scored 49.89 on the same held-out; base 57.87
(`projects/tajik-asr/EXPERIMENTS.md`). The 1,070 h bought 24.5 percent relative, and v3 matching
the contaminated v2 (37.40) proves real generalization, not memorization.

### The label-quality ceiling, quantified

The conversational references are machine labels (Scribe ensemble + compile-down), and the test
set is WER-gated at 0.35 against a Scribe verify pass. Computed from the curation store
(`projects/tajik-asr/data/curator.sqlite`, gated clips of the 157 frozen held-out videos): Scribe
verify-pass self-agreement is 12.9 percent WER clip-mean, 15.2 percent pooled
(duration-weighted 14.9); a fresh independent Scribe pass scores ~17.6 percent (owner
measurement; higher than the verify pass because the gate conditions on that pass). What this
means:

- The benchmark floor is NOT zero. A model that perfectly reproduced single-pass Scribe would
  still score 15 to 18 on these references, because the references are the fused ensemble, not
  any single pass.
- v3 at 37.65 is therefore roughly 20 WER points, or about 50 percent relative, above the
  practical agreement floor. The headroom claim "far from maxed" is solid; the model is nowhere
  near the teacher-agreement ceiling, unlike Persian where the student already beats the teacher
  on gold sets.
- The ceiling starts to BIND when model WER approaches the low-to-mid 20s on this benchmark.
  Below that, score changes will be dominated by reference noise and a native-speaker check (or
  a gold-labeled slice) becomes the only way to see real progress. `TODO.md` already flags this
  under "Someday".

### What scaling predicts for doubling the hours

Fitting WER = A * D^-beta to the repo's own two points (5.8 h at 49.89, 1,070 h at 37.65) gives
beta = 0.054, so doubling to ~2,100 h predicts ~3.7 percent relative, about 1.4 points (37.7 to
36.3). That fit understates the conversational-data effect (v0's 5.8 h was read speech, wrong
domain), so treat 1.5 to 3 points as the honest band for a same-mix doubling. The literature
warning is sharper: OWLS found data scaling saturates quickly WITHOUT added diversity, and that
adding a new data distribution restored gains where same-distribution scaling had plateaued
(<https://arxiv.org/pdf/2502.10373>). Whisper similarly saw gains slow past the first tens of
thousands of hours (<https://arxiv.org/pdf/2212.04356>). Practical translation: queue NEW channels
(new speakers, registers, topics) before queueing more videos from the 41 existing channels. The
per-channel spread in the v3 eval supports this: held-out WER ranges from 28.5
(alifbo_podcast) to 61.8 (aziya_khujand) (`projects/tajik-asr/data/logs/eval_heldout_v3_20260608_2109.log`),
so the corpus is far from homogeneous and the worst channels mark the domains the mix
underrepresents.

### Is a label-improvement iteration worth more than more hours?

Roughly equal expected value, and they compose. Three label levers, all free in dollars:

1. **Agreement-filtered self-training with v3 in the loop.** Run v3 over the gated-out clips
   (the WER > 0.35 rejects plus the 37k language-gate drops that were actually Tajik) and over
   fresh audio; keep clips where v3 and Scribe agree. This is standard iterative pseudo-labeling
   / noisy-student practice, with consistent gains reported across iterations in low-resource
   settings (<https://arxiv.org/pdf/2408.14026>, <https://www.researchgate.net/publication/354140808_Improved_Noisy_Student_Training_for_Automatic_Speech_Recognition>);
   Google reported pseudo-labels outperforming human labels at scale
   (<https://arxiv.org/pdf/2203.12668>). v3 is now a second, independent-ish teacher; two-teacher
   agreement recovers data a single-teacher gate threw away and cleans data the gate let
   through. Expected: 1 to 3 points, partially additive with more hours.
2. **More Scribe passes per clip in the ensemble + better fusion prompts.** Cheapest variant of
   the same idea; the compile-down already fuses en+tgk passes
   (`projects/tajik-asr/EXPERIMENTS.md`, 2026-05-30 entry). Adding a third pass and a
   disagreement-aware fusion prompt lowers reference noise on TRAIN, which matters because at
   WER 0.35 the gate admits up to one error in three words. Expected: smaller, maybe 0.5 to 1.5
   points, but it also raises the benchmark's resolution.
3. **A 1,000-clip gold slice.** No native speaker is available, but a Tajik-literate freelancer
   correcting machine labels is a different (cheaper) ask than annotation from scratch, and even
   500 gold clips would calibrate how far the relative benchmark is from truth. Until then every
   Tajik number is relative to Scribe.

### Recipe notes

The v2 run-1 drift (FLEURS sampled at 2 percent under sqrt tempering, dev WER rising while UER
fell) and its fix (hand-weighted TSV, FLEURS to 12 percent, lr 5e-6) are the template for v4;
the preset default beta_corpus 0.5 will do the same thing again on any corpus this lopsided
(`projects/tajik-asr/EXPERIMENTS.md`, `packages/omni-finetune-core/src/omni_finetune_core/presets.py`).
The vocabulary-gate fix (+1,563 clips, commit 6259a355 per `TODO.md`) is already queued for the
v4 export. LM fusion applies to Tajik too: Tajik text is scarcer than Persian but the corpus's
own 180k fused transcripts (~10M words) plus tg Wikipedia is enough for a 5-gram; on a model at
37 WER the expected relative gain is larger than on Persian's 8 to 20.

**Tajik verdict: not maxed. Next 2 to 4 points come from (a) new-channel conversational data
and (b) a v3-in-the-loop label iteration, in either order, ideally both feeding one v4 export.
The Scribe-agreement floor (~15 to 18 on this benchmark) is still ~20 points away and does not
bind yet.**

---

## 3. Georgian

### What exists

145.3 h gold-labeled v0 export, never trained: Common Voice scripted 135.8 h + FLEURS 9.4 h +
CV spontaneous 0.15 h; 98.6 h train / 22.3 dev / 24.4 test; 0 unk rows
(`projects/georgian-asr/data/datasets/v0/export_summary.json`). Train/eval CLI is template-ready
(`georgian-train --regime gpu_max`, per `TODO.md`).

### External anchors

- Base-model expectation: Meta's omni training saw 327 h of `kat_Geor`, more than the 125.9 h of
  `tgk_Cyrl` and bigger relative coverage than Persian; the 7B-LLM scores CER 1.9 on their
  Georgian benchmark (<https://raw.githubusercontent.com/facebookresearch/omnilingual-asr/main/per_language_results_table_7B_llm_asr.csv>).
  The Tajik base 300M scored 19.74 WER on FLEURS Tajik (`projects/tajik-asr/EXPERIMENTS.md`), so
  expect the Georgian base 300M somewhere in the 15 to 25 WER band on FLEURS Georgian, with WER
  inflated relative to CER by Georgian's agglutinative morphology. Measure it first; the eval is
  local and free.
- The closest published project (arXiv 2501.14788) trained on almost exactly this data
  inventory: MCV 76.4 h + FLEURS 3 h. Baseline 13.41 WER on the MCV test, 12.81 with YouTube
  pseudo-labels (4 percent relative), final 5.73 WER with an RNN-T head and punctuation
  handling (<https://arxiv.org/html/2501.14788>). Their conclusion was that the existing Georgian
  gold data "is enough to train a reasonably good ASR model" without major augmentation.
- Older fine-tunes for calibration: wav2vec2-xls-r-1b-ka 15.3 WER on MCV; whisper-large-v2-ka
  31.9 (same source).

### The ablation roadmap implied by Persian and Tajik

1. **v0 run, this week.** `gpu_max_finetune` preset, lr 1e-5, bf16, grad-accum 2, layerwise
   checkpointing; step budget from TRUE export hours: `recommend_num_steps(98.6, target_epochs
   in 20-30)` gives roughly 40k to 60k steps, so cap at ~30k and early-stop on dev plateau the
   way Persian v4 did at 34k on 1,032 h. Eval base model on FLEURS/CV test BEFORE training (two
   numbers, one command, free) so the delta is anchored, the lesson of the Tajik base row.
2. **Expected v0 result.** Persian got CV 21.9 / FLEURS 9.8 from 224 h of clean data on this
   exact recipe (`projects/persian-asr/EXPERIMENTS.md`, scribe-exact-match run); Georgian has
   99 h gold train and a stronger base. Expect roughly 10 to 16 WER on FLEURS Georgian and a
   large drop on CV test; the published 5.73 MCV number used a different architecture and
   punctuation normalization, so do not treat it as the bar, but sub-15 CV WER is the level the
   data supports.
3. **When YouTube pays.** Both prior languages say: gold read-speech training leaves
   conversational WER roughly untouched (Tajik v0 was 49.9 conversational while scoring 17.3 on
   FLEURS; Persian FLEURS-only collapsed off-domain, `projects/*/EXPERIMENTS.md`). The published
   Georgian YouTube pseudo-labeling gain on the READ benchmark was only 4 percent relative
   (<https://arxiv.org/html/2501.14788>). So: scrape YouTube only when conversational Georgian
   becomes a target, and if so, carve the frozen held-out video manifest BEFORE the first
   YouTube export (the v3 lesson; the template already has `heldout` support per `TODO.md`).
   The curator pipeline makes this mechanical: enqueue channels, segment, label, verify
   script-aware, gate at WER 0.35.
4. **Skip entirely:** transliteration-style augmentation (Tajik v1 proved a wash, dead in
   `projects/tajik-asr/docs/persian-augmentation-experiment-20260530.md`), the 1B, warm
   restarts.

**Georgian verdict: the cheapest large win in the repo. One preset-driven run on data that
already exists, with gold labels, no label-ceiling caveats, and a published near-replica saying
it works.**

---

## 4. Cross-cutting

1. **Greedy-only decoding is the standing gap.** Every benchmark number in this repo is greedy
   CTC argmax (`packages/omni-finetune-core/src/omni_finetune_core/recipe/wer_calculator.py`).
   Beam search with a KenLM n-gram via pyctcdecode is the single standard technique the project
   has never tried, it applies to all three languages, needs no GPU and no new audio, and the
   tokenizer is char-level so it drops in. This is the clearest thing being left on the table.
2. **No checkpoint averaging.** The regime keeps best-3 and ships best-1
   (`presets.py:_best_wer_regime`). Averaging them is free and worth 1 to 2 percent relative
   (<https://caiman-asr.myrtle.ai/training/checkpoint_averaging.html>).
3. **Warm restarts measured useless twice, still in the toolbox.** Persian re-warm: noise.
   The preset docstring already says the gain is usually small; the evidence says zero. Treat
   `warm_restart` as deprecated in practice and spend GPU on data ablations.
4. **Mixture tempering will repeat the v2 run-1 drift.** beta_corpus 0.5 let a 99 percent
   YouTube corpus drown FLEURS to 2 percent of batches; the fix was a hand-weighted TSV
   (`projects/tajik-asr/EXPERIMENTS.md`). The fix lives in a TSV, not in the preset; any new
   lopsided export (Persian v5, Georgian+YouTube) re-derives it by hand or repeats the drift.
   Worth a `mixture-weights` default that floors benchmark-domain corpora at ~10 percent.
5. **The worldspeech split is anomalous and unexamined.** WER 27.6 with CER 17.3 (every other
   Persian split has CER 4 to 10) and even Scribe scores 31.3/19.5 there. A CER that high on a
   trained model usually means reference normalization or transliteration problems in the split
   itself, not acoustics. Audit before ever optimizing toward it.
6. **Eval bias is mostly handled, with one hole.** Persian has six splits, Tajik has the
   conversational held-out (the single best methodological move in the repo; FLEURS compressed a
   20-point spread to 3, `projects/tajik-asr/EXPERIMENTS.md`). The hole: no absolute,
   human-verified conversational number exists for any language, so all conversational claims
   are relative to Scribe. One small gold slice per language is the cheapest fix and converts
   every relative claim into an absolute one.
7. **The teacher-student pattern is now proven and reusable.** Persian: agreement-filtered
   student beats teacher on all six splits. Tajik: student still 20 points from the agreement
   floor. The general rule for this repo: when the student beats the teacher on gold splits,
   stop iterating labels and go get new audio; when the student is far from the agreement floor,
   labels and hours both still pay.

---

## Next 5 experiments, ranked

| # | Experiment | Language | Expected gain | Effort |
|---|---|---|---|---|
| 1 | Georgian v0 fine-tune (gpu_max preset, ~30k steps, base-model eval first) | Georgian | New language at usable quality; ~10-16 FLEURS WER vs an untrained baseline | 1 day setup + ~2-3 GPU-days, all assets exist |
| 2 | KenLM + pyctcdecode beam decoding, wired into the shared eval; Persian first (naab text), then Tajik (own transcripts + Wikipedia) | All | Persian: 5-15 percent relative on CV/YouTube; Tajik: likely more at 37 WER | 2-4 days CPU work, no GPU, no new data |
| 3 | Tajik v4: new-channel scrape (prioritize underrepresented registers; worst held-out channels show the gap) + vocab-gate fix + same hand-weighted recipe | Tajik | 1.5-3 points conversational WER (scaling fit: ~1.4 for a same-mix doubling; diversity should beat the fit) | Weeks wall-clock, zero dollars, pipeline proven |
| 4 | Tajik label iteration: v3-as-second-teacher agreement filtering over rejects + fresh audio, +1 Scribe pass in the ensemble; feeds the same v4 export | Tajik | 1-3 points, partially additive with #3 | ~1 week, free API, mostly orchestration |
| 5 | Checkpoint averaging (average best-3) evaluated on Persian rewarm and Tajik v3 run dirs | Persian, Tajik | 1-2 percent relative, free | Hours |

Not on the list and why: Persian Scribe-v5 relabeling (student already beats teacher on gold),
1B retry (dead without an fp32-quality optimizer path; revisit only after #3 saturates the
300M), whisper-large-v3 fine-tune (weaker starting point on Persian, 15x inference cost),
transliteration augmentation (measured dead), warm restarts (measured dead, twice).
