# Scribe Curation SQLite Organization, 2026-05-27

## Goal

Build the next Persian ASR training manifest from the canonical Scribe SQLite store with a small set of human-readable row decisions.

The working store is:

```text
data/curation/scribe_jobs/scribe-canonical-all-20260516T192536Z/scribev2.full-20260523.sqlite
```

## Ground Rules

- SQLite is the working curation store.
- Training text for accepted Persian rows comes from `normalized_reference`.
- Scribe output is evidence for whether the reference/audio pair is usable.
- WER and CER are computed on the same normalized strings used for classification.
- Rows whose normalized reference is empty are dropped before Scribe or LLM calls.
- The Persian normalizer is `persian_asr_dataset.vendor.nvidia_stt_fa_fastconformer_hybrid_large.maybe_normalize`.

Official references:

- NVIDIA NeMo Curator audio curation pipeline: ingestion, optional ASR inference, WER/CER quality metrics, filtering, export. <https://docs.nvidia.com/nemo/curator/latest/about/concepts/audio/curation-pipeline.html>
- NVIDIA NeMo Curator WER filtering: calculate WER, inspect distributions, apply thresholds, tune thresholds by domain. <https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-assessment/wer-filtering>
- NVIDIA Speech Data Explorer: inspect alphabet, vocabulary/OOV words, zero-accuracy words, and high-CER utterances. <https://docs.nvidia.com/nemo-framework/user-guide/25.02/nemotoolkit/tools/speech_data_explorer.html>
- Meta self-training for end-to-end ASR: strong teacher models, pseudo-label filtering, and ensemble/agreement improve student training. <https://ai.meta.com/research/publications/self-training-for-end-to-end-speech-recognition/>
- Hugging Face ASR fine-tuning guidance: train on a transcript field, normalize text consistently, and skip empty normalized references during metric computation. <https://huggingface.co/docs/transformers/tasks/asr>

Operational interpretation:

- Transcript-difference classification says what differs between reference and model text.
- Export policy says whether `normalized_reference` is safe as the training label.
- Scribe and 300M outputs are evidence. They do not replace the export label unless a separate modified-label workflow is created.
- WER/CER bands select review queues. They do not automatically make a row trainable.

## Tables

`scribe_curation`

- One row per dataset item.
- Best source for canonical row identity, audio path, raw reference, raw Scribe, current normalized reference, current normalized Scribe, current WER/CER.
- `difference_category` is populated only for current exact matches plus legacy carryover in a small subset; treat blank values here as missing projection, not missing history.

`scribe_audit`

- Prior full classification table, 704,451 rows.
- Useful as historical evidence: `difference_category`, `difference_description`, `likely_cause`, `suggested_action`.
- Its normalized strings differ from the current curation normalization for 474,787 joined rows, so categories from this table require a current normalized-text check before export.

`scribe_results`

- Raw Scribe response store.
- Use for provenance and endpoint metadata.

`scribe_rerun_results`

- Latest rerun output for rows that previously normalized to empty.
- Current status: 17,859 rows, 2,651 normalized nonempty, 15,208 normalized empty.

`script_equivalence`

- Currently has 17,859 rows with empty decision fields.
- Keep it out of human-facing exports until it contains actual decisions.

## Current Counts

Current resolved text means:

- use latest `scribe_rerun_results.normalized_transcript` when present
- otherwise use `scribe_curation.normalized_scribe`

Decision-sized buckets:

| decision | rows | hours | avg WER | avg CER |
|---|---:|---:|---:|---:|
| use_reference_same_normalized | 249,025 | 268.75 | 0.0000 | 0.0000 |
| use_reference_tiny_delta | 33,726 | 74.11 | 0.1104 | 0.0149 |
| candidate_close_text | 67,236 | 143.71 | 0.1363 | 0.0341 |
| use_reference_audit_safe_current_gate | 67,210 | 138.85 | 0.2524 | 0.0521 |
| candidate_medium_text | 72,000 | 198.80 | 0.1984 | 0.0713 |
| needs_same_speech_script_check | 15,052 | 45.69 | 1.0000 | 1.0000 |
| review_content | 103,955 | 195.65 | 0.4173 | 0.1228 |
| reject_boundary_or_length | 70,587 | 191.23 | 0.3652 | 0.2034 |
| reject_wrong_audio_or_language | 5,495 | 10.09 | 0.9691 | 0.6811 |
| reject_annotation | 1,115 | 2.88 | 0.6792 | 0.5856 |
| reject_large_mismatch | 18,762 | 27.97 | 0.8707 | 0.3278 |
| drop_scribe_empty | 160 | 0.58 | 1.0000 | 1.0000 |
| drop_empty_reference | 132 | 0.27 |  |  |

## Runnable Manifest Surfaces

These are experiment surfaces. They make the quality/quantity tradeoff explicit.

| surface | rows | hours | avg WER | avg CER |
|---|---:|---:|---:|---:|
| E0 exact current | 249,025 | 268.75 | 0.0000 | 0.0000 |
| E1 exact plus CER <= 2% | 282,751 | 342.86 | 0.0132 | 0.0018 |
| E2 close, CER <= 5% and WER <= 20% | 349,987 | 486.57 | 0.0368 | 0.0080 |
| E3 broad, CER <= 10% and WER <= 35% | 421,987 | 685.36 | 0.0644 | 0.0188 |
| max_train_now | 417,197 | 625.41 | 0.0716 | 0.0151 |
| max_train_metric_audit | 488,992 | 823.67 | 0.0902 | 0.0233 |
| max_train_supported_script | 423,203 | 643.06 | 0.0847 | 0.0291 |
| max_train_metric_supported_script | 494,998 | 841.32 | 0.1012 | 0.0352 |
| max_train_metric_supported_script_rescue | 495,644 | 843.53 | 0.1024 | 0.0365 |
| max_train_metric_supported_script_rescue_text | 533,692 | 898.78 | 0.1245 | 0.0390 |
| max_train_metric_supported_script_rescue_text_boundary_asr | 555,930 | 957.63 | 0.1308 | 0.0428 |
| max_train_metric_supported_script_rescue_text_asr_recovery | 576,934 | 1,008.05 | 0.1381 | 0.0456 |
| max_train_metric_supported_script_rescue_text_asr_recovery_content20 | 578,160 | 1,009.83 | 0.1385 | 0.0459 |
| max_train_metric_supported_script_rescue_text_asr_recovery_content30 | 578,850 | 1,010.54 | 0.1388 | 0.0461 |
| max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer | 586,109 | 1,026.02 | 0.1429 | 0.0475 |
| max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script | 586,405 | 1,026.50 | 0.1434 | 0.0480 |
| max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold | 586,443 | 1,026.56 | 0.1434 | 0.0480 |
| max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold_content_low_v2 | 586,656 | 1,026.99 | 0.1434 | 0.0480 |
| max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold_content_low_boundary_v2 | 586,771 | 1,027.34 | 0.1435 | 0.0480 |
| max_train_metric_supported_script_rescue_text_asr_recovery_held_strong_v2 | 586,923 | 1,027.64 | 0.1435 | 0.0480 |
| max_train_metric_supported_script_rescue_text_asr_recovery_held_possible_v3 | 588,696 | 1,032.41 | 0.1441 | 0.0483 |
| script_queue | 15,052 | 45.69 | 1.0000 | 1.0000 |
| max_reviewable | 608,204 | 1,065.54 | 0.1687 | 0.0645 |

Readout:

- E0 matches the finished 300M exact run surface.
- E1 and E2 are clean metric expansions.
- E3 is the broad metric expansion closest to the old `<=35% WER` idea, with CER added.
- `max_train_metric_supported_script_rescue_text_asr_recovery_held_possible_v3` is the current maximal defended export.
- `script_queue` is a classification queue. It cannot be judged by current normalized WER/CER because the Scribe side normalizes to empty.

## The Latin/Script Problem

The current script queue has 15,052 rows and 45.69 hours. These rows have a nonempty normalized reference, raw Scribe text, and empty normalized Scribe, usually because Scribe produced Latin/script-mixed text.

Examples from the queue:

| job_order | old category | reference | Scribe |
|---:|---|---|---|
| 3146 | script_mismatch | می خواهم بروم ایستگاه چرینگ کراس | می خواهم بروم ایستگاه Charing Cross |
| 3221 | script_mismatch | توی پیوی تست کیس ها و یا جوابی مخصوص کد شما داده نمیشود | توی PV تست کیسها و یا جوابی مخصوص کد شما داده نمیشود |
| 4189 | script_mismatch | من از لیبره آفیس استفاده میکنم | من از LibreOffice استفاده میکنم |
| 9295 | script_mismatch | دراپ باکس | Dropbox |
| 9513 | script_mismatch | دی ان ا شکل حلزونی دارد | DNA شکل حلزونی دارد |
| 9711 | named_entity_mismatch | مشهورترین پول دهنده در جهان گوگل و سرویس گوگل ادز است | مشهورترین پولدهنده در جهان گوگل و سرویس Google Ads است |
| 9778 | script_mismatch | وان درایو مایکروسافت | OneDrive مایکروسافت |
| 6989 | language_mismatch | چای ساز | Soy sauce |
| 12481 | language_mismatch | گابرون | God room |
| 13897 | script_mismatch | افزارههایمان ابزار واپایش برهمکنشمان با رسانههای دیجیتالند | افزارهایی‌مان، ابزار warships بر هم‌کنش‌مان با رسانه‌های دیجیتال‌اند |

Manual sample decisions:

| job_order | decision | reason |
|---:|---|---|
| 3146 | `same_speech_wrong_script` | `Charing Cross` matches `چرینگ کراس`. |
| 3221 | `same_speech_wrong_script` | `PV` matches `پیوی` in the reference. |
| 4189 | `same_speech_wrong_script` | `LibreOffice` matches `لیبره آفیس`. |
| 9295 | `same_speech_wrong_script` | `Dropbox` matches `دراپ باکس`. |
| 9513 | `same_speech_wrong_script` | `DNA` matches `دی ان ا`. |
| 9711 | `same_speech_wrong_script` | `Google Ads` matches `گوگل ادز`. |
| 6989 | `different_speech` | `Soy sauce` is not `چای ساز`. |
| 12481 | `different_speech` | `God room` is not `گابرون`. |
| 521 | `same_speech_minor_mixed_script` | Sentence matches except Scribe wrote `shut` for `چفت`. |
| 13897 | `same_speech_minor_mixed_script` | Sentence mostly matches, with `warships` substituted for `واپایش`. |

The useful split is:

- `same_speech_wrong_script`: Scribe uses Latin script for the same word or named entity. Export can use `normalized_reference`.
- `different_speech`: Scribe says a different word, phrase, or language. Drop or review audio.
- `mixed_error`: Scribe contains a mostly matching sentence with one or more English substitutions. Send to review or accept only after the same-speech check.
- `annotation_or_noise`: bracketed labels, non-speech, or endpoint artifacts. Drop from training unless a separate cleanup rule exists.

Current SQLite check:

- all 15,052 script-queue rows have Latin text only on the Scribe side, not in
  the normalized reference
- the current maximal export includes the 6,006 LLM-accepted script rows
  supported by 300M plus the 296 strict low-CER script rescue rows below
- strict 300M CER <= 0.10 script rescue adds 296 rows and 0.48 hours
- most excluded LLM-accepted script rows have high 300M CER, so they need audio
  review or a stronger teacher signal before inclusion

Strict script rescue candidates:

| 300M flag | rows | hours | avg 300M WER | avg 300M CER |
|---|---:|---:|---:|---:|
| `llm_accept_unconfirmed_by_300m`, CER <= 0.10 | 225 | 0.39 | 0.602 | 0.075 |
| `accepted_needs_audio_review`, CER <= 0.10 | 71 | 0.09 | 0.871 | 0.080 |

## Transcript Difference Vocabulary

Use this vocabulary when classifying what changed between reference text and a
model transcript. Keep these labels separate from export decisions.

| transcript category | meaning |
|---|---|
| `exact_match` | normalized strings match |
| `surface_rendering` | same spoken content with spelling, spacing, punctuation, diacritic, script, acronym, name, or digit-vs-word rendering differences |
| `minor_wording_variant` | likely same utterance with small morphology, function-word, or colloquial variant |
| `content_substitution` | content word, name, number, polarity, verb/person, or entity changes |
| `speech_inserted` | transcript adds speech absent from the reference |
| `speech_omitted` | transcript omits speech present in the reference |
| `boundary_overlap` | adjacent start/end spill or partial next/previous segment |
| `wrong_utterance` | different segment, speaker, or utterance |
| `different_language` | mainly another spoken language |
| `annotation_or_noise` | bracketed non-speech, speaker tags, endpoint artifacts |
| `unclear` | text evidence cannot decide |

Current stored labels map imperfectly into this vocabulary. `script_mismatch`
and `language_mismatch` are overloaded historical labels; they need same-speech
or ASR cross-check evidence before export. `row_decision`, `candidate_*`,
`use_reference_*`, `reject_*`, and `hold_for_audio_review` are policy/review
states, not transcript categories.

## Row Decisions

Use these names in exports and reports:

| row decision | meaning | export text |
|---|---|---|
| `use_reference_same_normalized` | normalized reference equals normalized Scribe | `normalized_reference` |
| `use_reference_tiny_delta` | tiny spelling/spacing/character delta, CER <= 2% | `normalized_reference` |
| `candidate_close_text` | close normalized pair, CER <= 5% and WER <= 20% | `normalized_reference` after sample gate |
| `candidate_medium_text` | broader normalized pair, CER <= 10% and WER <= 35% | `normalized_reference` after category gate |
| `needs_same_speech_script_check` | Scribe raw text exists but normalized Scribe is empty, often Latin/script output | run same-speech classifier |
| `review_content` | current metrics show real content differences | inspect or classify before export |
| `reject_boundary_or_length` | extra speech, omitted speech, or boundary text | drop for training manifest |
| `reject_wrong_audio_or_language` | wrong utterance or wrong spoken language | drop for training manifest |
| `reject_annotation` | non-speech annotation or speaker/label text | drop for training manifest |
| `reject_large_mismatch` | high metric mismatch | drop for training manifest |
| `drop_scribe_empty` | no usable Scribe evidence | drop from pseudo-label export |
| `drop_empty_reference` | empty normalized reference | drop before API/classification |

## Same-Speech Script Classifier Prompt

Use only for rows where normalized reference is nonempty and normalized Scribe is empty or script-mixed.

```text
You compare Persian ASR dataset rows for curation.

The training label, if accepted, will be normalized_reference.
Your job is only to decide whether raw_scribe_text appears to represent the same spoken utterance as normalized_reference.

Return JSON with exactly:
sample_id, job_order, decision, reason, evidence

decision must be one of:
same_speech_wrong_script,
same_speech_minor_mixed_script,
different_speech,
different_language,
annotation_or_noise,
unclear

Rules:
- Choose same_speech_wrong_script when Latin words are names, acronyms, products, places, or loanwords matching the Persian-script reference.
- Choose same_speech_minor_mixed_script when the sentence mostly matches but Scribe used an English spelling or translation for a small part.
- Choose different_speech when the words refer to a different utterance or meaning.
- Choose different_language when Scribe output is mainly another spoken language.
- Choose annotation_or_noise for bracketed non-speech labels, speaker labels, or endpoint artifacts.
- Choose unclear when text alone cannot decide.
- Do not choose a training label.

Input:
{
  "sample_id": "...",
  "job_order": 0,
  "normalized_reference": "...",
  "raw_reference_text": "...",
  "raw_scribe_text": "...",
  "old_difference_category": "..."
}
```

## Text Recovery Classifier Prompt

Use only for bounded `review_content` rows where current normalized WER/CER keeps the pair close enough for text review.

```text
You review Persian ASR dataset rows for text recovery.

The training label, if accepted, will be normalized_reference.
Scribe output is evidence about whether the reference/audio pair is usable.
Decide whether the reference text is safe enough for ASR training.

Return JSON with exactly:
sample_id, job_order, decision, reason, evidence

decision must be one of:
use_reference_text_recovery,
hold_for_audio_review,
reject_reference_mismatch

Rules:
- Choose use_reference_text_recovery when differences are bounded spelling, spacing, normalization, number/name rendering, filler, short boundary drift, or obvious Scribe error while the same Persian utterance remains clear.
- Choose reject_reference_mismatch when Scribe evidence indicates the audio likely says a different utterance or another spoken language than the reference.
- Choose hold_for_audio_review when a content word changes, meaningful speech is added/omitted, or text evidence cannot prove the reference is safe.
- Use normalized_reference as the only possible training text.
- Do not invent a corrected label.

Input:
{
  "sample_id": "...",
  "job_order": 0,
  "normalized_reference": "...",
  "normalized_scribe": "...",
  "raw_reference_text": "...",
  "raw_scribe_text": "...",
  "old_difference_category": "...",
  "current_wer": 0.0,
  "current_cer": 0.0
}
```

## Completed Work

Completed:

1. Created SQLite views `scribe_resolved_pairs`, `scribe_training_decisions`, and `scribe_asr_review_flags`.
2. Ran the full `needs_same_speech_script_check` review with `same_speech_script_v1` and `gpt-5.4-mini`.
3. Ran 300M Omni ASR cross-checks over the reviewed script rows that fit the Omni 40 second inference cap.
4. Exported the supported training surface from SQLite.

## Full Script Review Result

The script queue contained 15,052 rows and 45.69 hours. Every row has a review record.

| decision | rows | hours |
|---|---:|---:|
| `same_speech_minor_mixed_script` | 6,847 | 22.04 |
| `same_speech_wrong_script` | 2,177 | 4.30 |
| `different_speech` | 3,804 | 13.32 |
| `different_language` | 2,082 | 5.42 |
| `annotation_or_noise` | 108 | 0.51 |
| `unclear` | 34 | 0.10 |

Text-only LLM review accepted 9,024 rows and 26.35 hours. It rejected or held 6,028 rows and 19.34 hours.

## 300M Cross-Check Result

The 300M check used:

```text
omni_ctc_300m_v2_persian_scribe_exact_match_20260525_best
```

One LLM-accepted row had duration 40.26 seconds, above the Omni inference cap, so the 300M accepted-side check covered 9,023 of 9,024 LLM-accepted rows.

| side | rows | aggregate WER | aggregate CER | RTFx |
|---|---:|---:|---:|---:|
| LLM accepted script rows | 9,023 | 33.65% | 12.62% | 416.54x |
| LLM rejected or unclear script rows | 6,028 | 55.48% | 33.52% | 447.17x |

The cross-check flags are stored in `scribe_asr_review_flags`.

| flag | rows | hours | meaning |
|---|---:|---:|---|
| `llm_accept_supported_by_300m` | 6,006 | 17.64 | LLM accepted and 300M is within WER <= 0.50 and CER <= 0.15 |
| `llm_accept_unconfirmed_by_300m` | 1,856 | 5.61 | LLM accepted, 300M gives mixed evidence |
| `accepted_needs_audio_review` | 1,161 | 3.08 | LLM accepted, 300M strongly disagrees |
| `strong_rescue_review` | 152 | 0.15 | LLM rejected, 300M exact-matches the reference |
| `weak_rescue_review` | 844 | 3.26 | LLM rejected, 300M is within WER <= 0.35 and CER <= 0.10 |
| `reject_unconfirmed_by_300m` | 1,728 | 6.04 | LLM rejected, 300M gives mixed evidence |
| `reject_supported_by_300m` | 3,304 | 9.89 | LLM rejected, 300M strongly disagrees with the reference |

The important failure mode was semantic translation. Some rows looked acceptable to text-only review because the English Scribe text conveyed the same meaning as the Persian reference, but the 300M output showed the audio was English speech, not Persian speech. Those rows stay out of the supported training export.

## Exported Training Surface

The supported script export is:

```text
data/training/scribe-verified/max-train-supported-script-20260527
```

It contains base `max_train_now` rows plus script rows accepted by LLM review and supported by the 300M cross-check.

| split | rows | hours |
|---|---:|---:|
| train | 388,341 | 584.28 |
| dev | 17,860 | 29.02 |
| test | 17,002 | 29.75 |
| total | 423,203 | 643.06 |

The first ASR recovery export was:

```text
data/training/scribe-verified/max-train-metric-supported-script-rescue-text-asr-recovery-20260527
```

It contains:

- base metric/audit-safe rows
- `candidate_medium_text` rows except the stale high-risk old labels `wrong_segment`, `language_mismatch`, `script_mismatch`, and `low_confidence_unclear`
- script rows accepted by LLM and supported by the 300M cross-check
- rows recovered from `strong_rescue_review` and `weak_rescue_review` after `asr_rescue_v1`
- `review_content` safe-category rows accepted by `text_recovery_v1`
- accepted normalized-reference rows from the ASR recovery queues: boundary extra/omitted, boundary mismatch, content mismatch low-CER, and large mismatch low-CER

| split | rows | hours |
|---|---:|---:|
| train | 507,263 | 863.85 |
| dev | 23,775 | 42.34 |
| test | 23,100 | 42.90 |
| total | 554,138 | 949.08 |

The previous maximal export before the additional ASR recovery queues is:

```text
data/training/scribe-verified/max-train-metric-supported-script-rescue-text-20260527
```

It contains:

- base metric/audit-safe rows
- `candidate_medium_text` rows except the stale high-risk old labels `wrong_segment`, `language_mismatch`, `script_mismatch`, and `low_confidence_unclear`
- script rows accepted by LLM and supported by the 300M cross-check
- rows recovered from `strong_rescue_review` and `weak_rescue_review` after `asr_rescue_v1`
- `review_content` safe-category rows accepted by `text_recovery_v1`

| split | rows | hours |
|---|---:|---:|
| train | 488,525 | 817.35 |
| dev | 22,885 | 40.33 |
| test | 22,282 | 41.10 |
| total | 533,692 | 898.78 |

The prior maximal export before text recovery remains:

```text
data/training/scribe-verified/max-train-metric-supported-script-rescue-20260527
```

It contains 495,644 rows and 843.53 hours.

The earlier base export remains:

```text
data/training/scribe-verified/max-train-now-20260527
```

It contains 417,197 rows and 625.41 hours, before script-row recovery.

## Candidate Medium Decision

`candidate_medium_text` has 72,000 rows and 198.80 hours. Current metrics are bounded at WER <= 0.35 and CER <= 0.10.

The maximal export includes 71,795 rows and 198.26 hours from this block. It holds out 205 rows and 0.53 hours whose old audit categories are stale high-risk labels:

```text
wrong_segment
language_mismatch
script_mismatch
low_confidence_unclear
```

The included `candidate_medium_text` rows are dominated by old `content_mismatch`, `extra_speech`, `boundary_mismatch`, `omitted_speech`, `exact_match`, and `non_speech_annotation` labels under the current normalized comparison. Their examples show local substitutions, fillers, boundary fragments, and orthographic drift rather than wholesale wrong audio.

## ASR Rescue Review

The rescue review task is:

```text
asr_rescue_v1
```

It reviewed all 996 rows from `strong_rescue_review` and `weak_rescue_review`.

| decision | rows | hours |
|---|---:|---:|
| `use_reference_asr_rescue` | 646 | 2.21 |
| `hold_for_audio_review` | 310 | 1.09 |
| `reject_reference_mismatch` | 40 | 0.11 |

The maximal export includes only `use_reference_asr_rescue`.

## Text Recovery Review

The text recovery task is:

```text
text_recovery_v1
```

It reviewed the `review_content_safe` queue:

```sql
row_decision = 'review_content'
AND audit_difference_category IN (
  'punctuation_or_orthography_only',
  'near_match',
  'number_or_symbol_mismatch',
  'named_entity_mismatch',
  'exact_match'
)
AND resolved_cer <= 0.15
AND resolved_wer <= 0.50
```

Final result:

| decision | rows | hours |
|---|---:|---:|
| `use_reference_text_recovery` | 38,048 | 55.26 |
| `hold_for_audio_review` | 4,013 | 10.72 |
| `reject_reference_mismatch` | 120 | 0.22 |

The text-recovery export includes only `use_reference_text_recovery`.

## Next Review Queues

Use SQLite, not new dataset folders:

```sql
SELECT *
FROM scribe_asr_review_flags
WHERE asr_model_label IN (
  'omni300_exact_best_script_accepted_full',
  'omni300_exact_best_script_rejected_full'
)
AND asr_review_flag IN (
  'accepted_needs_audio_review',
  'llm_accept_unconfirmed_by_300m',
  'strong_rescue_review',
  'weak_rescue_review'
);
```

The highest-value manual/audio review queues are:

| queue | rows | hours |
|---|---:|---:|
| `accepted_needs_audio_review` | 1,161 | 3.08 |
| `llm_accept_unconfirmed_by_300m` | 1,856 | 5.61 |
| `hold_for_audio_review` from `asr_rescue_v1` | 310 | 1.09 |
| `reject_reference_mismatch` from `asr_rescue_v1` | 40 | 0.11 |

The completed broad recovery queue is:

| queue | rows | hours | result |
|---|---:|---:|---|
| `review_content_safe` | 42,181 | 66.19 | 38,048 rows accepted into the maximal export |
| `reject_boundary_or_length` extra/omitted, CER <= 0.20, WER <= 0.50 | 31,715 | 87.48 | 9,700 rows accepted into the boundary-ASR export |

## Boundary Extra/Omitted ASR Recovery

The boundary recovery queue was:

```sql
row_decision = 'reject_boundary_or_length'
AND audit_difference_category IN ('extra_speech', 'omitted_speech')
AND resolved_cer <= 0.20
AND resolved_wer <= 0.50
```

Queue size:

| category | rows | hours |
|---|---:|---:|
| `extra_speech` | 23,954 | 62.35 |
| `omitted_speech` | 7,761 | 25.13 |
| total | 31,715 | 87.48 |

The 300M check used:

```text
omni_ctc_300m_v2_persian_scribe_exact_match_20260525_best
```

Benchmark result:

| rows | hours | WER | CER | RTFx |
|---:|---:|---:|---:|---:|
| 31,715 | 87.48 | 29.68% | 11.94% | 371.47 |

ASR support flags:

| flag | rows | hours | avg ASR WER | avg ASR CER |
|---|---:|---:|---:|---:|
| `asr_exact_reference` | 695 | 1.30 | 0.0000 | 0.0000 |
| `asr_strong_reference` | 3,015 | 7.75 | 0.1177 | 0.0288 |
| `asr_possible_reference` | 7,742 | 22.40 | 0.2093 | 0.0724 |
| `asr_mixed_reference` | 19,044 | 53.26 | 0.3625 | 0.1451 |
| `asr_reject_reference` | 1,219 | 2.76 | 0.6519 | 0.2612 |

LLM review was first run on `asr_exact_reference`, `asr_strong_reference`, and
`asr_possible_reference`. The later mixed pass also reviewed
`asr_mixed_reference` rows after a 100-row gate returned 100 reviewed and 0
failures.

Final review result:

| decision | rows | hours |
|---|---:|---:|
| `use_reference_asr_recovery` | 22,238 | 58.84 |
| `hold_for_audio_review` | 7,973 | 25.02 |
| `reject_reference_mismatch` | 285 | 0.86 |

The boundary-only ASR export surface is:

```text
max_train_metric_supported_script_rescue_text_boundary_asr
```

Export path:

```text
data/training/scribe-verified/max-train-metric-supported-script-rescue-text-boundary-asr-20260527
```

Export result:

| split | rows | hours |
|---|---:|---:|
| train | 497,332 | 841.04 |
| dev | 23,342 | 41.41 |
| test | 22,718 | 42.09 |
| total | 543,392 | 924.54 |

## Content, Boundary, And Large ASR Recovery

The next ASR recovery pass used the same 300M checkpoint:

```text
omni_ctc_300m_v2_persian_scribe_exact_match_20260525_best
```

It covered three additional queues from SQLite.

| queue | rows | hours | WER | CER | RTFx |
|---|---:|---:|---:|---:|---:|
| `content_mismatch_low_cer` | 20,880 | 48.42 | 32.60% | 11.61% | 355.07x |
| `boundary_mismatch` | 14,069 | 35.75 | 28.33% | 10.47% | 403.47x |
| `large_mismatch_low_cer` | 2,564 | 2.32 | 79.54% | 8.35% | 430.24x |

The content queue had one row over the 40 second Omni inference cap, so the ASR run used the 20,880-row `<=40s` manifest.

ASR support flags:

| queue | flag | rows | hours | avg ASR WER | avg ASR CER |
|---|---|---:|---:|---:|---:|
| `content_mismatch_low_cer` | `asr_exact_reference` | 1,287 | 1.24 | 0.0000 | 0.0000 |
| `content_mismatch_low_cer` | `asr_strong_reference` | 1,801 | 3.67 | 0.1386 | 0.0331 |
| `content_mismatch_low_cer` | `asr_possible_reference` | 4,771 | 12.57 | 0.2323 | 0.0706 |
| `content_mismatch_low_cer` | `asr_mixed_reference` | 11,927 | 29.17 | 0.4106 | 0.1342 |
| `content_mismatch_low_cer` | `asr_reject_reference` | 1,094 | 1.77 | 0.7525 | 0.2463 |
| `boundary_mismatch` | `asr_exact_reference` | 318 | 0.62 | 0.0000 | 0.0000 |
| `boundary_mismatch` | `asr_strong_reference` | 1,712 | 4.33 | 0.1135 | 0.0290 |
| `boundary_mismatch` | `asr_possible_reference` | 4,092 | 10.96 | 0.2102 | 0.0719 |
| `boundary_mismatch` | `asr_mixed_reference` | 7,611 | 19.11 | 0.3714 | 0.1377 |
| `boundary_mismatch` | `asr_reject_reference` | 336 | 0.73 | 0.6910 | 0.2421 |
| `large_mismatch_low_cer` | `asr_exact_reference` | 433 | 0.35 | 0.0000 | 0.0000 |
| `large_mismatch_low_cer` | `asr_strong_reference` | 11 | 0.01 | 0.1887 | 0.0298 |
| `large_mismatch_low_cer` | `asr_possible_reference` | 37 | 0.04 | 0.2750 | 0.0499 |
| `large_mismatch_low_cer` | `asr_mixed_reference` | 375 | 0.37 | 0.5283 | 0.0729 |
| `large_mismatch_low_cer` | `asr_reject_reference` | 1,708 | 1.54 | 1.0728 | 0.1068 |

LLM review was first run on exact, strong, and possible ASR-reference bands.
The later mixed passes also reviewed the `boundary_mismatch` mixed band and
the `content_mismatch_low_cer` mixed band after 100-row gates returned 0
failures.

Final review results:

| task | decision | rows | hours |
|---|---|---:|---:|
| `content_mismatch_low_cer_asr_recovery_v1` | `use_reference_asr_recovery` | 9,303 | 21.97 |
| `content_mismatch_low_cer_asr_recovery_v1` | `hold_for_audio_review` | 9,426 | 22.97 |
| `content_mismatch_low_cer_asr_recovery_v1` | `reject_reference_mismatch` | 1,057 | 1.71 |
| `boundary_mismatch_asr_recovery_v1` | `use_reference_asr_recovery` | 11,225 | 28.05 |
| `boundary_mismatch_asr_recovery_v1` | `hold_for_audio_review` | 2,454 | 6.80 |
| `boundary_mismatch_asr_recovery_v1` | `reject_reference_mismatch` | 53 | 0.16 |
| `large_mismatch_low_cer_asr_recovery_v1` | `use_reference_asr_recovery` | 476 | 0.40 |
| `large_mismatch_low_cer_asr_recovery_v1` | `hold_for_audio_review` | 4 | 0.01 |
| `large_mismatch_low_cer_asr_recovery_v1` | `reject_reference_mismatch` | 1 | 0.00 |

The final export surface is:

```text
max_train_metric_supported_script_rescue_text_asr_recovery
```

Export path:

```text
data/training/scribe-verified/max-train-metric-supported-script-rescue-text-asr-recovery-20260527
```

Export result:

| split | rows | hours |
|---|---:|---:|
| train | 507,263 | 863.85 |
| dev | 23,775 | 42.34 |
| test | 23,100 | 42.90 |
| total | 554,138 | 949.08 |

## Additional Content ASR Recovery

The next content recovery queues used the same 300M checkpoint:

```text
omni_ctc_300m_v2_persian_scribe_exact_match_20260525_best
```

They target `review_content` rows where Scribe and reference still have bounded normalized disagreement. The export label remains `normalized_reference`; Scribe and the 300M output act as agreement evidence.

| queue | rows | hours | benchmark WER | benchmark CER | RTFx |
|---|---:|---:|---:|---:|---:|
| `content_mismatch_mid_cer`, 0.15 < CER <= 0.20 and WER <= 0.50 | 9,862 | 24.11 | 39.42% | 17.31% | 330.95x |
| `content_mismatch_high_cer_20_30`, 0.20 < CER <= 0.30 and WER <= 0.50 | 7,980 | 19.93 | 46.59% | 24.39% | 310.79x |

ASR support flags:

| queue | flag | rows | hours |
|---|---|---:|---:|
| `content_mismatch_mid_cer` | `asr_exact_reference` | 488 | 0.41 |
| `content_mismatch_mid_cer` | `asr_strong_reference` | 375 | 0.57 |
| `content_mismatch_mid_cer` | `asr_possible_reference` | 1,060 | 1.99 |
| `content_mismatch_mid_cer` | `asr_mixed_reference` | 6,759 | 18.60 |
| `content_mismatch_mid_cer` | `asr_reject_reference` | 1,180 | 2.54 |
| `content_mismatch_high_cer_20_30` | `asr_exact_reference` | 376 | 0.29 |
| `content_mismatch_high_cer_20_30` | `asr_strong_reference` | 169 | 0.20 |
| `content_mismatch_high_cer_20_30` | `asr_possible_reference` | 422 | 0.56 |
| `content_mismatch_high_cer_20_30` | `asr_mixed_reference` | 3,724 | 9.81 |
| `content_mismatch_high_cer_20_30` | `asr_reject_reference` | 3,289 | 9.06 |

LLM review first covered exact, strong, and possible ASR-reference bands. Later
mixed passes reviewed the `asr_mixed_reference` rows for both queues after
100-row gates completed without failures.

Final LLM review results:

| task | decision | rows | hours |
|---|---|---:|---:|
| `content_mismatch_mid_cer_asr_recovery_v1` | `use_reference_asr_recovery` | 3,466 | 8.35 |
| `content_mismatch_mid_cer_asr_recovery_v1` | `hold_for_audio_review` | 4,621 | 12.14 |
| `content_mismatch_mid_cer_asr_recovery_v1` | `reject_reference_mismatch` | 595 | 1.08 |
| `content_mismatch_high_cer_20_30_asr_recovery_v1` | `use_reference_asr_recovery` | 1,748 | 3.68 |
| `content_mismatch_high_cer_20_30_asr_recovery_v1` | `hold_for_audio_review` | 2,564 | 6.51 |
| `content_mismatch_high_cer_20_30_asr_recovery_v1` | `reject_reference_mismatch` | 379 | 0.68 |

The original content export surfaces below were produced before the later mixed
passes. The current export surface at the end of this document supersedes them.

| surface | rows | hours | train | dev | test |
|---|---:|---:|---:|---:|---:|
| `max_train_metric_supported_script_rescue_text_asr_recovery_content20` | 555,364 | 950.87 | 508,412 | 23,809 | 23,143 |
| `max_train_metric_supported_script_rescue_text_asr_recovery_content30` | 556,054 | 951.57 | 509,073 | 23,823 | 23,158 |

## High-WER ASR Recovery

The next pass checked rows where WER is high but CER keeps the strings close enough to inspect. This catches short utterances, names, numbers, orthography, and boundary effects where word-level scoring over-penalizes the pair.

Queue and benchmark results:

| queue | rows | hours | benchmark WER | benchmark CER | RTFx |
|---|---:|---:|---:|---:|---:|
| `orthography_near_low_cer_high_wer_50_75` | 8,618 | 10.03 | 51.95% | 8.64% | 428.20x |
| `entity_symbol_mid_cer_15_25` | 2,814 | 6.83 | 29.75% | 13.57% | 352.61x |
| `content_mismatch_low_cer_high_wer_50_75` | 3,465 | 5.13 | 48.48% | 13.19% | 359.34x |
| `boundary_low_cer_high_wer_50_75` | 1,815 | 4.16 | 38.48% | 11.31% | 427.67x |

ASR support flags:

| queue | exact | strong | possible | mixed | reject |
|---|---:|---:|---:|---:|---:|
| `orthography_near_low_cer_high_wer_50_75` | 871 | 328 | 952 | 4,763 | 1,704 |
| `entity_symbol_mid_cer_15_25` | 167 | 313 | 488 | 1,677 | 169 |
| `content_mismatch_low_cer_high_wer_50_75` | 222 | 158 | 414 | 2,004 | 667 |
| `boundary_low_cer_high_wer_50_75` | 34 | 162 | 393 | 1,015 | 211 |

LLM review results:

| task | decision | rows | hours |
|---|---|---:|---:|
| `orthography_near_low_cer_high_wer_50_75_asr_recovery_v1` | `use_reference_asr_recovery` | 2,060 | 2.33 |
| `orthography_near_low_cer_high_wer_50_75_asr_recovery_v1` | `hold_for_audio_review` | 85 | 0.13 |
| `orthography_near_low_cer_high_wer_50_75_asr_recovery_v1` | `reject_reference_mismatch` | 6 | 0.01 |
| `entity_symbol_mid_cer_15_25_asr_recovery_v1` | `use_reference_asr_recovery` | 855 | 1.77 |
| `entity_symbol_mid_cer_15_25_asr_recovery_v1` | `hold_for_audio_review` | 96 | 0.26 |
| `entity_symbol_mid_cer_15_25_asr_recovery_v1` | `reject_reference_mismatch` | 17 | 0.03 |
| `content_mismatch_low_cer_high_wer_50_75_asr_recovery_v1` | `use_reference_asr_recovery` | 532 | 0.67 |
| `content_mismatch_low_cer_high_wer_50_75_asr_recovery_v1` | `hold_for_audio_review` | 229 | 0.31 |
| `content_mismatch_low_cer_high_wer_50_75_asr_recovery_v1` | `reject_reference_mismatch` | 33 | 0.04 |
| `boundary_low_cer_high_wer_50_75_asr_recovery_v1` | `use_reference_asr_recovery` | 514 | 1.17 |
| `boundary_low_cer_high_wer_50_75_asr_recovery_v1` | `hold_for_audio_review` | 71 | 0.17 |
| `boundary_low_cer_high_wer_50_75_asr_recovery_v1` | `reject_reference_mismatch` | 4 | 0.01 |

## Current Final Export

Surface:

```text
max_train_metric_supported_script_rescue_text_asr_recovery_held_possible_v3
```

Current export path:

```text
data/training/scribe-verified/max-train-metric-supported-script-rescue-text-asr-recovery-held-possible-v3-full-20260527
```

Export result:

| split | rows | hours |
|---|---:|---:|
| train | 538,117 | 938.80 |
| dev | 25,632 | 46.61 |
| test | 24,947 | 47.00 |
| total | 588,696 | 1,032.41 |

Accepted ASR recovery rows in the final manifest:

| task | rows |
|---|---:|
| `boundary_extra_omitted_asr_recovery_v1` | 22,238 |
| `boundary_mismatch_asr_recovery_v1` | 11,225 |
| `content_mismatch_low_cer_asr_recovery_v1` | 9,303 |
| `content_mismatch_mid_cer_asr_recovery_v1` | 3,466 |
| `orthography_near_low_cer_high_wer_50_75_asr_recovery_v1` | 2,060 |
| `content_mismatch_high_cer_20_30_asr_recovery_v1` | 1,748 |
| `entity_symbol_mid_cer_15_25_asr_recovery_v1` | 855 |
| `content_mismatch_low_cer_high_wer_50_75_asr_recovery_v1` | 532 |
| `boundary_low_cer_high_wer_50_75_asr_recovery_v1` | 514 |
| `large_mismatch_low_cer_asr_recovery_v1` | 476 |
| strict low-CER script rescue rows | 296 |
| held rows where 300M exact-matches the reference | 38 |
| `content_mismatch_low_cer_held_strong_asr_recovery_v2` accepted rows | 213 |
| `boundary_extra_omitted_held_strong_asr_recovery_v2` accepted rows | 115 |
| `boundary_mismatch_held_strong_asr_recovery_v2` accepted rows | 48 |
| `content_mismatch_mid_cer_held_strong_asr_recovery_v2` accepted rows | 35 |
| `entity_symbol_mid_cer_15_25_held_strong_asr_recovery_v2` accepted rows | 21 |
| `content_mismatch_high_cer_20_30_held_strong_asr_recovery_v2` accepted rows | 18 |
| `content_mismatch_low_cer_high_wer_50_75_held_strong_asr_recovery_v2` accepted rows | 14 |
| `orthography_near_low_cer_high_wer_50_75_held_strong_asr_recovery_v2` accepted rows | 9 |
| `boundary_low_cer_high_wer_50_75_held_strong_asr_recovery_v2` accepted rows | 7 |
| `boundary_extra_omitted_held_possible_asr_recovery_v3` accepted rows | 641 |
| `content_mismatch_low_cer_held_possible_asr_recovery_v3` accepted rows | 556 |
| `boundary_mismatch_held_possible_asr_recovery_v3` accepted rows | 265 |
| `content_mismatch_mid_cer_held_possible_asr_recovery_v3` accepted rows | 134 |
| `content_mismatch_high_cer_20_30_held_possible_asr_recovery_v3` accepted rows | 49 |
| `content_mismatch_low_cer_high_wer_50_75_held_possible_asr_recovery_v3` accepted rows | 48 |
| `orthography_near_low_cer_high_wer_50_75_held_possible_asr_recovery_v3` accepted rows | 32 |
| `boundary_low_cer_high_wer_50_75_held_possible_asr_recovery_v3` accepted rows | 26 |
| `entity_symbol_mid_cer_15_25_held_possible_asr_recovery_v3` accepted rows | 20 |
| `large_mismatch_low_cer_held_possible_asr_recovery_v3` accepted rows | 2 |

Validation checks:

| check | result |
|---|---|
| manifest row count | 588,696 rows |
| added identities over prior held-strong-v2 export | 1,773 |
| removed identities versus prior held-strong-v2 export | 0 |
| duplicate `sample_id` plus `job_order` identities | 0 |
| empty `text` fields | 0 |
| average manifest WER / CER | 0.1441 / 0.0483 |

The exact-held rescue rule is intentionally tiny. It only admits rows where a
previous ASR recovery review held the item for audio review, then the matching
300M queue hypothesis exact-matched `normalized_reference` with WER = 0 and
CER = 0.

The `content_mismatch_low_cer_held_strong_asr_recovery_v2` pass reviewed held
`content_mismatch_low_cer` rows with 300M strong-reference evidence
(WER <= 0.20 and CER <= 0.05), excluding exact rows already covered by the
exact-held rule.

| decision | rows | hours |
|---|---:|---:|
| `use_reference_asr_recovery` | 213 | 0.434 |
| `hold_for_audio_review` | 341 | 0.677 |
| `reject_reference_mismatch` | 41 | 0.058 |

The `boundary_extra_omitted_held_strong_asr_recovery_v2` pass used the same
second-pass rule on held boundary extra/omitted rows with 300M strong-reference
evidence, excluding exact rows already covered by the exact-held rule.

| decision | rows | hours |
|---|---:|---:|
| `use_reference_asr_recovery` | 115 | 0.345 |
| `hold_for_audio_review` | 83 | 0.243 |
| `reject_reference_mismatch` | 6 | 0.018 |

Additional strong-held second-pass results:

| task | accepted | held | rejected | accepted hours |
|---|---:|---:|---:|---:|
| `boundary_mismatch_held_strong_asr_recovery_v2` | 48 | 29 | 3 | 0.133 |
| `content_mismatch_mid_cer_held_strong_asr_recovery_v2` | 35 | 57 | 11 | 0.048 |
| `entity_symbol_mid_cer_15_25_held_strong_asr_recovery_v2` | 21 | 10 | 1 | 0.049 |
| `content_mismatch_high_cer_20_30_held_strong_asr_recovery_v2` | 18 | 24 | 5 | 0.022 |
| `content_mismatch_low_cer_high_wer_50_75_held_strong_asr_recovery_v2` | 14 | 27 | 7 | 0.015 |
| `orthography_near_low_cer_high_wer_50_75_held_strong_asr_recovery_v2` | 9 | 6 | 0 | 0.018 |
| `boundary_low_cer_high_wer_50_75_held_strong_asr_recovery_v2` | 7 | 4 | 0 | 0.016 |
| `large_mismatch_low_cer_held_strong_asr_recovery_v2` | 0 | 1 | 0 | 0.000 |

The possible-reference pass reviewed all remaining held rows where the 300M
crosscheck was in the `asr_possible_reference` band (`WER <= 0.35` and
`CER <= 0.10`). Accepted rows are included in the current export; held and
rejected rows remain outside the manifest.

| task | accepted | held | rejected | accepted hours |
|---|---:|---:|---:|---:|
| `boundary_extra_omitted_held_possible_asr_recovery_v3` | 641 | 820 | 18 | 1.998 |
| `content_mismatch_low_cer_held_possible_asr_recovery_v3` | 556 | 1,466 | 133 | 1.476 |
| `boundary_mismatch_held_possible_asr_recovery_v3` | 265 | 277 | 6 | 0.761 |
| `content_mismatch_mid_cer_held_possible_asr_recovery_v3` | 134 | 329 | 31 | 0.230 |
| `content_mismatch_high_cer_20_30_held_possible_asr_recovery_v3` | 49 | 106 | 29 | 0.069 |
| `content_mismatch_low_cer_high_wer_50_75_held_possible_asr_recovery_v3` | 48 | 116 | 15 | 0.068 |
| `orthography_near_low_cer_high_wer_50_75_held_possible_asr_recovery_v3` | 32 | 30 | 8 | 0.051 |
| `boundary_low_cer_high_wer_50_75_held_possible_asr_recovery_v3` | 26 | 33 | 0 | 0.063 |
| `entity_symbol_mid_cer_15_25_held_possible_asr_recovery_v3` | 20 | 38 | 4 | 0.052 |
| `large_mismatch_low_cer_held_possible_asr_recovery_v3` | 2 | 1 | 0 | 0.002 |

Remaining high-value queues:

| held-row slice | rows | hours | next action |
|---|---:|---:|---|
| remaining ASR mixed held rows | 21,116 | 57.91 | audio review or stronger teacher |
| Scribe normalized text exactly equals 300M hypothesis | 492 | 0.66 | modified-label/reference-error review queue |

The modified-label queue is real but separate from this export. It should not
reuse `normalized_reference` as the label without review. Examples include
reference text such as `اتهام اقدام علیه امنیت کشور` where Scribe and 300M agree
on `اتهام تبلیغ علیه امنیت کشور`, and short Common Voice rows where both teacher
outputs agree against the reference.

Script rows outside strict rescue remain separate:

| slice | rows | hours | next action |
|---|---:|---:|---|
| `llm_accept_unconfirmed_by_300m` script rows outside strict rescue | 1,631 | 5.22 | audio review or stronger teacher |
| `accepted_needs_audio_review` script rows outside strict rescue | 1,090 | 2.99 | audio review or stronger teacher |
