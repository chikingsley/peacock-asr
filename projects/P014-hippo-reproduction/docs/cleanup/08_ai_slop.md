# 08: AI Slop, Stubs, LARP, and Useless Comments

## 1. Executive summary

The P014 codebase is **unusually clean** for slop. No emoji, no section-divider
banners, no `TODO/FIXME/HACK/XXX/NOTE` tags, no `# NEW/CHANGED/UPDATED/WAS/
PREVIOUSLY` scar-tissue markers, no `pass`/`...` stub bodies, no unjustified
`NotImplementedError`. A repository-wide `rg '^\s*#' src/ tests/` returns 103
comment lines across ~5,200 LOC of `.py` files — roughly one inline comment per
50 LOC.

The real slop lives in **docstrings**, not comments:

- Heavy module-level docstrings that overlap with `docs/` and paper references.
- Function docstrings that restate type annotations in prose.
- Several `Args:` / `Returns:` blocks on purely internal helpers.

No file is "almost entirely slop". The worst offenders (by ratio) are
`features/ssl_utterance.py`, `freespeak/asr.py`, and `freespeak/align.py`,
which carry multi-paragraph module headers that could collapse to 1-2 lines
plus a paper citation. The CTC-GOP module header is the single largest block of
prose (30 lines), but most of it is justified (explains the LPP/LPR math and
why `vocab_size == 40 → 41-dim features`).

Nothing dangerous found; nothing required rewriting for correctness. This is a
polish pass.

## 2. Quantitative

- `.py` files: 18 under `src/`, 8 under `tests/`
- Total LOC: 5,235
- Inline `#` comment lines: **103**
- `NOTE/FIXME/HACK/XXX/TODO`: **0**
- `NEW/CHANGED/UPDATED/REPLACED/WAS/PREVIOUSLY`: **0**
- Emoji: **0**
- Rough split of the 103 inline comments:
  - Justified (hidden invariant / paper eq / edge case / algorithm choice): ~70
  - Redundant-with-code (what, not why): ~20
  - Paraphrased restatement of the next line / unclear: ~13
- Docstrings total: ~40 module + class + function. Of these, ~15 are justified
  (paper citations, non-obvious contracts), ~15 are borderline (duplicate type
  info), ~10 are pure prose that could go.

## 3. Findings (REMOVE)

### `src/p014/cli.py`

- Lines 189-191 (HIGH): the 3-line comment about `--json` being accepted for
  consistency describes a trivial stylistic choice and the `_ = bool(args.json)`
  below is itself dead code — the discard assignment is meaningless. Delete
  both.
- Line 270 `# Re-export for callers that want programmatic typing.` (HIGH):
  redundant, `__all__` speaks for itself.

### `src/p014/training.py`

- Line 58 docstring `"""Lightweight wrapper so we can fall back to local
  logging gracefully."""` (MEDIUM): paraphrases class name + fallback logic.
- Lines 120-129 `train_hippo` docstring (LOW): mostly redundant with code and
  config docs. The `P(τ) = τ/T` reference is useful; rest restates function
  signature.
- Lines 139 `# Datasets are deterministic across seeds — cache once.` (KEEP —
  this is a non-obvious "why").
- Lines 160-162 (KEEP — explains phone-vocab unioning, non-obvious).
- Lines 378-387 `_iter_cycle` docstring (MEDIUM): three sentences for a 2-line
  function. Collapse to one line: `"""Yield batches forever, re-creating the
  iterator each pass to preserve per-epoch shuffling."""`.
- Lines 413-417 (MEDIUM): "The 'primary' loader drives the epoch length..."
  paragraph reads like AI narration. Keep the operative sentence about the
  infinite read-aloud cycle; drop the preamble.
- Lines 448-450 `# Non-curriculum training always reports the scenario that
  matches settings.scenario.` (MEDIUM): restates the next line.
- Lines 667, 691 `_collect_metric_values` / `_sample_std` one-line docstrings
  (LOW): fine either way; if removing, they are redundant with the names.

### `src/p014/data.py`

- Lines 71-74 `load_read_aloud_resources` docstring (LOW): "Load / build all
  inputs required by the HiPPO read-aloud pipeline. Returns ..." — the return
  line is redundant with the type annotation. Keep the first sentence only.
- Lines 163-180 `load_freespeak_resources` 18-line docstring (MEDIUM): numbered
  paper-pipeline recap duplicates `docs/`. Collapse to the App. D citation plus
  the key non-obvious facts (GOP cache uses ASR phones; SSL cache is reused).

### `src/p014/blocks.py`

- Lines 11, 25, 73, 91, 131, 155, 207 one-line class docstrings (KEEP — each
  cites a paper / equation / figure; that is the pattern the user wants).

### `src/p014/model.py`

- Line 21 `"""Two-layer FFN regressor used for phone/word/utterance heads (paper
  App. B)."""` (KEEP).
- Lines 38, 62, 90, 104 similar (KEEP — paper equation numbers).

### `src/p014/features/ctc_gop.py`

- Lines 179-181 `_effective_vocab` docstring (LOW): fine as is.
- Lines 189-197 `_canonical_phone_ids` docstring (LOW): "Map SpeechOcean phones
  to CTC-model phone ids" is justified; the "We return raw ints so callers..."
  sentence is LARP (of course we return ints, the signature says so). Delete
  the trailing sentence.
- Line 247 `# Matches the default strided behaviour of ``datasets.Dataset.shard``.`
  (KEEP — genuine "why").
- Lines 392-397 `_ctc_log_forward` docstring (KEEP — explains log-space reason).
- Lines 415 `# blank position` (MEDIUM): the block under it is clearly a blank
  handler; comment is redundant. Either remove or move into a single line
  explaining the three-branch lattice recurrence (more useful).
- Lines 447 `# Fall back to the pre-decoded array when the path is not available.`
  (MEDIUM): restates the `if path is None` branch.
- Lines 485-489 (KEEP — documents sign conventions vs. reference impl, exactly
  the kind of "why" the user wants).
- Lines 503 `# Deletion path — drop this phone from the canonical sequence.`
  (KEEP — non-obvious that the blank_id branch is the deletion LPR).
- Lines 512-514 (KEEP — cites the reference convention).
- Lines 523 `# We only need a light-weight length check...` (MEDIUM): restates
  the one-line body.
- Lines 561-567 `phones_to_canonical_ids` docstring (LOW): fine; it's a public
  export.

### `src/p014/features/ssl_utterance.py`

- Lines 1-18 module docstring (MEDIUM): 18 lines to say "extract mean-pooled
  1024-dim embeddings from three SSL models and concatenate to 3072, one model
  at a time for VRAM". Trim by ~50%. Keep the paper citation, the three model
  ids, and the VRAM-sequential-load constraint. Drop the restatement of the
  concat formula.
- Line 126 `"""Extract 1024-dim mean-pooled embeddings from one SSL model."""`
  (KEEP — narrow and accurate).

### `src/p014/freespeak/align.py`

- Lines 1-13 module docstring (KEEP — genuinely explains why NW over plain
  Levenshtein, which is a real "why").
- Line 67 `# Cost table and operation back-pointers.` (LOW): mild redundancy
  but harmless.
- Line 86 `# Prefer diagonal on ties so matches/substitutions dominate shifts.`
  (KEEP — non-obvious tie-break policy).

### `src/p014/freespeak/annotation.py`

- Lines 1-15 module docstring (KEEP — documents the three paper-faithful
  assignment rules; useful).
- Lines 51-56 `_assign_word` docstring (KEEP — explains the `None` convention).
- Line 59 `# Insertion: no aligned reference → zero scores across the board.`
  (LOW): restates `if ref_word_index is None`. Remove.
- Line 73 `# Ref phone with no ASR counterpart — dropped per App. D.` (KEEP —
  paper reference).
- Line 77 `# Defensive: an emitted op must carry a hyp index.` (KEEP —
  invariant).
- Line 110 `# Reference word not produced by the learner — no ASR unit to
  score.` (KEEP — paper rule).
- Lines 133-136 "Pathological case" comment (KEEP — describes a real
  downstream crash constraint, with a specific behaviour justification).

### `src/p014/freespeak/asr.py`

- Lines 1-10 module docstring (MEDIUM): 10 lines restating paper App. D and
  cache format. Trim to 3 — citation + cache-key rationale.
- Lines 34-38 (KEEP — documents the punctuation-strip regex policy).
- Line 51 `"""Clean raw Whisper output into an uppercase string and a word
  tuple."""` (KEEP).
- Lines 131-145 `transcribe_split` docstring (MEDIUM): `Args:` block
  paraphrases type annotations. Shrink to the non-obvious parts (batch model
  lifecycle, cache path).

### `src/p014/freespeak/g2p.py`

- Lines 1-10 module docstring (KEEP — explains the stress-digit policy, a real
  invariant that cross-cuts `ctc_gop.py`).
- Line 20 `# g2p_en emits non-phone tokens for punctuation and spaces...`
  (KEEP).
- Lines 27-35 `grapheme_to_phones` `Args:` / `Returns:` (LOW): fine but
  duplicates types. Leave for public API.

### `src/p014/cono.py`

- Lines 1-14 module docstring (KEEP — explains the differentiable-centroid
  trick, which is non-obvious).
- Lines 30-42 `cono_loss` `Args:` / `Returns:` (LOW): fine for a public
  function.
- Lines 56-57 (KEEP — documents why `.detach()` is ok).
- Line 63, 72, 80, 85, 90, 95 inline comments (KEEP — all either cite an
  equation or explain a mathematical step).

### Tests

- `tests/test_freespeak_align.py:1-10` module docstring listing (a)-(f) cases
  (LOW): harmless but could drop without loss.
- `tests/test_freespeak_annotation.py:1-8` (KEEP — documents the monkeypatch).
- `tests/test_freespeak_train_smoke.py:1-6` (KEEP — explains monkeypatch
  rationale).
- `tests/test_read_aloud_pipeline.py:136` `# Each component loss is exactly
  1.0...` (KEEP — justifies the `10.0` expected value).
- `tests/test_read_aloud_pipeline.py:179` `"""End-to-end smoke test: one
  epoch, one seed, CONO enabled."""` (KEEP — narrow and true).
- `tests/test_cono.py:10-13` (KEEP — explains why the two-case structure).
- `tests/test_read_aloud_pipeline.py:223` `# Must be a finite scalar not
  greater than zero (log of a probability).` (KEEP — the assertion's "why").

## 4. Findings (REWRITE)

### `src/p014/training.py`

- Lines 413-417: rewrite as: `# When curriculum is on, the FS loader is the
  epoch clock and the read loader is wrapped infinitely; the Bernoulli toss
  below picks which one actually feeds this step.` (MEDIUM)
- Lines 729-731 `# Suppress a noisy warning from numpy when all values are
  identical inside safe_pcc ...` (MEDIUM): this is placed at module bottom
  with a global side-effect; rewrite tighter or move next to `safe_pcc`
  import. Suggested: `# safe_pcc already handles constant inputs; silence the
  accompanying numpy RuntimeWarning.`

### `src/p014/data.py`

- Line 582 `# Trim to the shorter length so the pipeline stays consistent
  when cache was built with a different phone count (e.g. stale cache).`
  (MEDIUM): reword as hidden-invariant — `# Stale GOP cache may disagree on
  phone count; trim to the minimum so the collator stays well-defined. Cache
  freshness is enforced at write time, so this only triggers for mid-flight
  schema changes.`

### `src/p014/features/ssl_utterance.py`

- Module docstring: see REMOVE; the retained version should mention the
  fixed 3072-dim contract (callers depend on it) rather than the three model
  names (already in `SSL_MODEL_IDS`).

## 5. Findings (KEEP)

Non-exhaustive list of comments/docstrings worth preserving outright:

- All paper/equation citations in `src/p014/blocks.py` and `src/p014/model.py`
  (these follow the user's "cite equation 3 of Gulati et al." pattern).
- `src/p014/cono.py` module + inline comments — documents the
  differentiable-centroid construction and eq. 15/16 derivation.
- `src/p014/config.py:190-196` `TrainingRunSettings` docstring — clarifies the
  distinction from `ExperimentConfig`, which is load-bearing.
- `src/p014/features/ctc_gop.py:1-31` — explains LPP, LPR, the 41-dim feature
  width derivation, and why the default checkpoint is chosen. Core domain doc.
- `src/p014/features/ctc_gop.py:485-489` — sign-convention note vs. frank613's
  reference implementation; this is a genuine bug-prevention comment.
- `src/p014/freespeak/align.py:1-13` — NW vs. Levenshtein justification.
- `src/p014/freespeak/annotation.py:1-15` — App. D score-assignment table.
- `src/p014/freespeak/g2p.py:1-10` — stress-digit retention policy.
- `src/p014/training.py:139` (dataset caching invariant), 160-162 (phone-vocab
  union), 301-305 (curriculum `τ/T` formula with paper citation).

## 6. Stubs / LARP

None found. Every `raise` is a real error path:

- `raise ValueError` for dim mismatches (justified invariants).
- `raise RuntimeError("training produced no evaluation metrics")`
  (`training.py:352`) — correct guard, not a stub.
- `raise RuntimeError("free-speaking scenario requires ...")` — real branch
  guard.

Empty `except: pass`: none. The two `except Exception as exc:` blocks in
`training.py:_TrackioHandle` log and degrade gracefully — legitimate
`# pragma: no cover` annotations.

No `pass`/`...` function bodies, no placeholder returns.

## 7. Confidence tagging summary

- HIGH-confidence removals (safe, unambiguous): `cli.py:189-191`, `cli.py:270`,
  `features/ctc_gop.py:196` trailing sentence, `freespeak/annotation.py:59`.
- MEDIUM-confidence rewrites/trims: `training.py:378-387`, `413-417`,
  `448-450`, `729-731`; `data.py:163-180` docstring, `582` rewrite;
  `features/ssl_utterance.py:1-18`; `freespeak/asr.py:1-10`, `131-145`;
  `features/ctc_gop.py:415`, `447`, `523`.
- LOW-confidence (style-only, consider with maintainer preferences):
  `training.py:58` docstring, `120-129` docstring, `667`, `691`; public-API
  `Args:` blocks (`cono.py`, `g2p.py`).

Total suggested edits: ~25 distinct sites. Net deletion if all applied: on the
order of 60-80 lines of prose, leaving comment-line count near 70 and
trimming ~40 lines of docstring. The codebase is already at a standard most
research repos would benefit from copying.
