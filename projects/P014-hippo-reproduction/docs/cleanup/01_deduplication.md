# P014 — Deduplication & DRY Assessment

## Executive summary

The codebase is **mildly DRY-sick** — not pathologically duplicated, but carrying
a consistent load of parallel boilerplate around caching, audio loading, and the
two read-aloud/free-speaking pipelines. Most duplication is shallow and
mechanical (three exact copies of `_load_waveform`, three independent
`TARGET_SAMPLE_RATE = 16_000` constants, three sidecar "is fresh?" checks of
identical shape). There is no premature over-abstraction; on the contrary, a
handful of paper-structural duplicates (`load_read_aloud_resources` vs
`load_freespeak_resources`, the train/eval forward+loss pair in `training.py`)
are approaching the size where copy-drift becomes a real risk.

**Single highest-value consolidation:** unify the three waveform decoders and
the four cache/sidecar freshness routines into one tiny `p014.features._io`
(or `p014._audio` + `p014._cache`) module. It collapses ~120 lines across four
files, removes three copies of a subtle resampling branch, and makes future
format-version bumps a one-line change.

## Findings

### 1. Triplicate `_load_waveform` / `_load_audio`

**Confidence: HIGH**
**Locations:**

- `src/p014/features/ctc_gop.py:436-461` (`_load_audio`)
- `src/p014/features/ssl_utterance.py:94-116` (`_load_waveform`)
- `src/p014/freespeak/asr.py:64-86` (`_load_waveform`)

**Current pattern:** Three byte-for-byte equivalent functions that accept an
HF audio payload, handle the `get_all_samples` / dict / path branches, mono
down-mix, and resample to 16 kHz. Each file also re-declares
`TARGET_SAMPLE_RATE = 16_000`.

**Proposed consolidation:** `p014/_audio.py` exporting
`TARGET_SAMPLE_RATE` and `load_waveform(audio_payload) -> Tensor`. Replace the
three copies with a single import.

**Risk / why NOT:** Essentially none — the three bodies are identical down to
variable names. The only debatable call is whether `TARGET_SAMPLE_RATE` belongs
in `_audio` vs `config`. Pick one and move on.

---

### 2. Cache-freshness check is copy-pasted four times

**Confidence: HIGH**
**Locations:**

- `src/p014/features/ctc_gop.py:263-287` (`_parts_manifest_is_fresh`)
- `src/p014/features/ctc_gop.py:532-555` (`_cache_is_fresh`)
- `src/p014/features/ssl_utterance.py:69-91` (`_cache_is_fresh`)
- `src/p014/freespeak/asr.py:98-117` (`_cache_is_fresh`)
- `src/p014/data.py:431-442` (inline sidecar freshness check in
  `load_or_build_word_embeddings`)

**Current pattern:** Each function opens a sidecar JSON, try/excepts the
parse, and checks a fixed set of keys (`version`, `model_id`, `split`,
`dataset_id`, `num_examples`, ...) against expected values.

**Proposed consolidation:** `p014/_cache.py::sidecar_matches(path, expected:
dict) -> bool` that loads the JSON with the shared try/except and compares
keys. Each caller passes its own `expected` dict; the caller keeps domain
knowledge but loses the boilerplate.

**Risk / why NOT:** Each caller has a slightly different expected key-set
(SSL uses `model_ids` tuple, CTC uses `phone_source`). A `dict`-driven helper
handles that cleanly. The only real risk is hiding the per-module cache
contract behind an indirection — keep each call-site's expected dict visible
and the contract stays readable.

---

### 3. Duplicated `CACHE_FORMAT_VERSION = 1`

**Confidence: HIGH**
**Locations:**

- `src/p014/features/ctc_gop.py:55`
- `src/p014/features/ssl_utterance.py:43`
- `src/p014/freespeak/asr.py:30`
- `src/p014/data.py:415` (`_WORD_EMBEDDING_CACHE_VERSION`)

**Current pattern:** Four independent version constants all equal to `1`.
When the cache schema evolves in one module, nothing forces the others to be
revisited.

**Proposed consolidation:** Either (a) one shared `CACHE_FORMAT_VERSION` in
`p014._cache` *if* they should always bump together, or more defensibly
(b) keep them separate but co-located in the `p014._cache` module with an
explanatory comment, so a reader of cache code sees all four versions at once.

**Risk / why NOT:** These *are* independent schemas — GOP parts, GOP cache,
SSL utt, Whisper transcripts, and ModernBERT word embeddings each version
their own payload. Forcing a single constant would cause spurious
invalidations. Recommendation is (b): co-locate, don't merge.

---

### 4. `load_read_aloud_resources` and `load_freespeak_resources` share ~70% structure

**Confidence: MEDIUM**
**Locations:**

- `src/p014/data.py:62-149` (read-aloud)
- `src/p014/data.py:152-306` (free-speaking)

**Current pattern:** Both: load annotations → clip to max examples → call
`extract_gop_for_split` twice → call `extract_ssl_utterance_for_split` twice →
`_load_gop_cache` / `_load_ssl_cache` four times → check gop_dim agreement →
build phone vocab → build word embeddings → construct two
`ReadAloudFeatureDataset`s. Only the annotation source and the GOP phone
source differ.

**Proposed consolidation:** Extract a private
`_assemble_datasets(train_annotations, test_annotations, *,
gop_phone_override, cache_suffix, ...)` that does the extract-load-validate-
build sequence. The two public entry points then reduce to the
annotation/transcript sourcing + a single call into `_assemble_datasets`.

**Risk / why NOT:** The two functions aren't quite symmetric — free-speaking
builds the phone vocab over the *union* of read-aloud refs and FS phones
(`data.py:282`), and its word-embedding cache uses a `_freespeak` suffix.
A naive helper that hides these differences would be worse than the current
duplication. A good abstraction must expose these parameters explicitly —
design discussion required, hence MEDIUM.

---

### 5. Train + eval duplicate model-forward + APA/CONO-loss logic

**Confidence: MEDIUM**
**Locations:**

- `src/p014/training.py:452-476` (training step)
- `src/p014/training.py:528-555` (evaluation step)

**Current pattern:** Both call `model(...)` with the same 7-argument unpack
from `device_batch`, both compute APA loss via `compute_loss`, both then
optionally add CONO on top with identical coefficients. The only difference
is that training `.backward()`s and eval appends to metric buffers.

**Proposed consolidation:** A `_forward_and_loss(model, device_batch,
settings) -> tuple[ReadAloudPredictions, Tensor, Tensor]` helper returning
`(predictions, apa_loss, total_loss_with_cono)`. Both call-sites shrink by
~20 lines each and become impossible to desync.

**Risk / why NOT:** Eval runs under `@torch.no_grad()` while training does
not; the wrapper needs to be neutral about grad mode. Also, a shared helper
makes it marginally harder to insert training-only logic (e.g. grad clipping,
mixed precision) later — but that's cheap to re-inline when needed.

---

### 6. Three `DataLoader` constructions in `_run_trial` differ only in dataset + shuffle

**Confidence: HIGH**
**Locations:** `src/p014/training.py:250-282` — `primary_loader`,
`test_loader`, `read_loader_for_curriculum` each wrap a near-identical
`DataLoader(..., batch_size=settings.batch_size,
num_workers=settings.num_workers, collate_fn=collate_read_aloud_batch)` call
inside a `cast`.

**Current pattern:** 33 lines of repeated kwargs.

**Proposed consolidation:** A tiny `_make_loader(dataset, *, shuffle: bool,
settings) -> DataLoader[ReadAloudBatch]` helper. Cast lives inside the
helper.

**Risk / why NOT:** None — this is mechanical.

---

### 7. Dynamic device resolution duplicated

**Confidence: HIGH**
**Locations:**

- `src/p014/training.py:652-655` (`resolve_device`)
- `src/p014/features/ctc_gop.py:637-639`
- `src/p014/features/ssl_utterance.py:204-206`
- `src/p014/freespeak/asr.py:172-174`

**Current pattern:** `device or torch.device("cuda" if
torch.cuda.is_available() else "cpu")`, repeated verbatim.

**Proposed consolidation:** Move `resolve_device` out of `training.py` into a
shared utility (`p014._device` or `p014.utils`) and import everywhere.

**Risk / why NOT:** None.

---

### 8. Identical `_build_tiny_dataset` helpers across test files

**Confidence: HIGH**
**Locations:**

- `tests/test_read_aloud_pipeline.py:147-175`
- `tests/test_freespeak_train_smoke.py:19-48`

**Current pattern:** Near-identical factories building a synthetic
`ReadAloudFeatureDataset` with hand-typed phone/word/utterance scores. The
FS variant also re-declares `shared_phone_vocab` as a local, whereas the RA
variant hard-codes the same `{"AA": 1, "BB": 2, "CC": 3}` inside.

**Proposed consolidation:** A `tests/conftest.py` fixture
`tiny_read_aloud_dataset` (or plain helper in `tests/_fixtures.py`) taking
`num_examples` and `phone_vocab` kwargs.

**Risk / why NOT:** None — test fixtures are the textbook case for shared
helpers. Also removes the `ReadAloudAnnotation(...)` literal from
`test_freespeak_annotation.py:25`.

---

### 9. Atomic write helpers only used in one module

**Confidence: LOW**
**Locations:** `src/p014/features/ctc_gop.py:251-260`
(`_write_atomic_json`, `_write_atomic_torch`).

**Current pattern:** These atomic-rename helpers are local to `ctc_gop.py`,
but several other cache writers (`ssl_utterance.py:237-255`, `asr.py:223-249`,
`data.py:462-476`) do *non-atomic* `Path.write_text` / `torch.save` directly.

**Proposed consolidation:** Promote the atomic helpers to `p014/_cache.py`
and use them everywhere. Bonus: eliminates a class of "half-written cache →
stale sidecar" bugs that the existing atomic helpers were added to fix.

**Risk / why NOT:** Not pure deduplication — it's *adding* consistency where
there was divergence. Design discussion territory. LOW because it's more
"should we be atomic everywhere?" than "is this duplicated?".

---

### 10. `key_padding_mask.unsqueeze(-1)` multiplication scattered through blocks/model

**Confidence: LOW**
**Locations:** `src/p014/blocks.py:150, 177, 180, 184, 193`; `src/p014/model.py:86, 100, ...`

**Current pattern:** Every post-residual step re-applies
`hidden * mask.unsqueeze(-1)` to zero out padded positions.

**Proposed consolidation:** A tiny `_apply_mask(x, mask)` helper.

**Risk / why NOT:** The three-character pattern is already idiomatic PyTorch
and a helper mostly saves characters, not clarity. Leave as-is.

---

### 11. Dynamic `import_module` + `cast(Any, ...)` for `transformers`

**Confidence: LOW**
**Locations:** `data.py:453`, `ctc_gop.py:635`, `ssl_utterance.py:128`,
`asr.py:176`.

**Current pattern:** Each module lazily imports `transformers` and does
`cast(Any, ...)` gymnastics to silence the type checker.

**Proposed consolidation:** Tempting but leaky — each call wants a different
sub-API (`AutoModel`, `Wav2Vec2ForCTC`, `AutoModelForSpeechSeq2Seq`,
`AutoFeatureExtractor`). A shared helper would just hide the lazy-import
boilerplate, not the differing usage. Crosses into "weak types" territory —
leave for that audit.

---

## Explicit non-findings

- **`@dataclass(frozen=True)` data classes** (`ReadAloudAnnotation`,
  `ReadAloudSample`, `ReadAloudBatch`, `TrialSummary`, `AggregateReport`,
  `_GopCacheSidecar`, `_GopPartsManifest`, `_SslCacheSidecar`,
  `TranscriptionResult`, `ReadAloudPredictions`): they *look* parallel, but
  each carries different tensors/fields with meaningful names. Merging them
  would destroy the domain model.
- **`ConvLlamaStack` vs `ConvLlamaBlock`**: Not duplication — block is one
  layer, stack is depth-N. Structural, not copy-pasted.
- **Argparse flag definitions in `cli.py:18-79`**: The `_add_training_flags`
  and `_add_gop_extract_flags` / `_add_gop_merge_flags` already factor what
  can be reasonably factored; further DRY (e.g. auto-deriving flags from
  Pydantic fields) would be clever but obscure — not worth it.
- **Three `Regressor` calls at `model.py` heads**: Same class, three
  instances. Already DRY.
- **`_ctc_log_forward` loop structure**: Paper-faithful CTC forward
  algorithm; don't refactor.

## Cross-cutting

- **Weak types:** Finding #11 (`cast(Any, import_module(...))`) overlaps.
  Any typed-wrapper approach belongs in that audit, not here.
- **try/except cleanup:** The sidecar `try/except (OSError, ValueError)`
  pattern in findings #2 and #9 repeats the same exception tuple four times.
  If a dedicated helper lands per finding #2, that `try/except` collapses to
  one place.
- **Legacy removal:** None of these findings gate deletion of dead code —
  they all touch live code paths.
