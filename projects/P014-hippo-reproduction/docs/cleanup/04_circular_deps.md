# 04 — Circular Dependency / Import Graph Health (P014)

Tooling: `grimp` 3.x (build_graph + find_shortest_chain). `pydeps`/`pycycle`
were not needed — grimp gave a complete, deterministic answer.
Scope: the `p014` package only (18 internal modules, ~4400 LOC).

## 1. Executive summary

The graph is **mostly healthy but has one real, already-papered-over cycle**
between `p014.data` and `p014.freespeak.*`. The author already noticed it
and broke it with function-local imports inside `data.load_freespeak_resources`
(`data.py:182-188`). That makes the package *import* cleanly at runtime, but
grimp still reports it as a cycle because the statements do execute the first
time the function is called. This is a latent tech-debt item, not a current
bug.

Other observations: no `if TYPE_CHECKING:` guards anywhere; no star imports;
no god module; `__init__.py` re-exports are minimal and safe; and the
layering `config/metrics → blocks/model, features, data → training → cli`
is respected everywhere *except* for the `data ↔ freespeak` edge noted
above.

## 2. Import graph (internal edges only)

Top-level imports (module-scope, executed on import):

```text
cli.py            → config, features, training
training.py       → config, cono, data, metrics, model
model.py          → blocks
data.py           → features                         (module-scope)
metrics.py        → data
features/__init__ → features.ctc_gop, features.ssl_utterance
freespeak/__init__→ freespeak.align, .annotation, .asr, .g2p
freespeak.annotation → config, data, freespeak.align, .asr, .g2p
freespeak.asr     → data                             (needs HF_DATASET_ID)
__init__.py       → config
__main__.py       → cli
```

Function-local imports (deferred, inside a function body):

```text
data.load_freespeak_resources()  → config.FreeSpeakingAssignmentConfig
                                 → freespeak (TranscriptionResult,
                                              build_freespeak_annotations,
                                              transcribe_split)
                                 → freespeak.g2p.grapheme_to_phones
```

Leaves with zero internal imports: `config.py`, `cono.py`, `blocks.py`,
`features/ctc_gop.py`, `features/ssl_utterance.py`, `freespeak/align.py`,
`freespeak/g2p.py`.

## 3. Cycles found

### Cycle A — `p014.data` ⇄ `p014.freespeak` (and its submodules)

- **Participants**: `src/p014/data.py`, `src/p014/freespeak/__init__.py`,
  `src/p014/freespeak/asr.py`, `src/p014/freespeak/annotation.py`.
- **Confidence: HIGH** — grimp reports shortest chains in both directions:
  - `data → freespeak → freespeak.asr → data`
  - `data → freespeak → freespeak.annotation → data`
- **How it manifests**: *function-local-import workaround*. The back-edge
  from `data.py` is intentionally deferred: `data.py:182-188` puts
  `from p014.freespeak import …` inside the body of
  `load_freespeak_resources`, with a docstring-length justification
  preceding it. The forward edges are ordinary module-scope imports
  (`freespeak/asr.py:27: from p014.data import HF_DATASET_ID`;
  `freespeak/annotation.py:20: from p014.data import ReadAloudAnnotation`).
  So the package imports fine — but the cycle re-appears the moment
  `load_freespeak_resources` is actually called, and it lies hidden in the
  graph for any future refactor.
- **Why the cycle exists**: `data.py` owns two very different concerns —
  (1) dataset-wide constants + dataclasses (`HF_DATASET_ID`,
  `ReadAloudAnnotation`, `UTTERANCE_SCORE_KEYS`, `WORD_SCORE_KEYS`,
  `load_annotations`), and (2) the high-level orchestrator
  `load_freespeak_resources` that stitches Whisper + G2P + alignment + CTC
  features together. The first concern is low-level; the second is
  top-of-stack. `freespeak/*` only needs the low-level piece.
- **Suggested fix (preferred)**: extract the shared types/constants into a
  new leaf module `p014/schema.py` (or `p014/annotations.py`) containing
  `HF_DATASET_ID`, `ReadAloudAnnotation`, the score-key tuples, and the
  `annotation_to_dict / annotation_from_dict` helpers. Then:
  - `data.py` imports from `schema.py` (re-export for back-compat if desired),
  - `freespeak/asr.py` and `freespeak/annotation.py` import from
    `schema.py`,
  - the function-local imports in `data.load_freespeak_resources` become
    normal top-of-file imports.
- **Alternative fix**: move `load_freespeak_resources` itself out of
  `data.py` and into `freespeak/pipeline.py`. `data.py` then becomes the
  leaf, `freespeak` stays high. Slightly bigger diff but arguably cleaner,
  since that function is really the freespeak orchestrator.

No other cycles exist in the package — every other pair grimp was asked
about had a chain in only one direction.

## 4. Layering violations

Using the suggested layering (`config, metrics, blocks, features leaves →
model, data, freespeak leaves → freespeak.annotation / training → cli`):

| Edge | Direction | Verdict |
|------|-----------|---------|
| `metrics.py → data.py` | leaf-ish utility → data | **Minor inversion.** `metrics.py` only needs the tuple constants `UTTERANCE_SCORE_KEYS`, `WORD_SCORE_KEYS` (`metrics.py:8`). These belong in a schema module, not in `data.py`. Same root cause as Cycle A; the fix above eliminates this too. |
| `freespeak.asr → data.HF_DATASET_ID` | freespeak leaf → data | Same issue: a string constant should not force a dependency on the 670-line data loader. |
| `freespeak.annotation → data.ReadAloudAnnotation` | freespeak → data | Same: a dataclass should live in schema. |
| `data → features` | data → features | Fine. `features/` is a leaf (no internal imports). |
| `training → {config, cono, data, metrics, model}` | top → all | Fine. `training.py` is correctly near the top. |
| `cli → {config, features, training}` | top → top | Fine. `cli.py` is the entry point. |

Nothing flows downward into `config.py`, `blocks.py`, `cono.py`, or the
`features` leaves — good.

## 5. Import hygiene issues

1. **Function-local imports that exist only to break the cycle**
   (`data.py:182, 183, 188`). These are tech debt, not a legitimate
   lazy-import optimisation — `freespeak/*` is not an optional extra,
   it is required whenever `load_freespeak_resources` runs. Remove
   them by doing the schema extraction in §3.
2. **No `if TYPE_CHECKING:` guards anywhere.** Both a plus (no hidden
   soft cycles) and a note — the function-local imports in `data.py`
   could *almost* have been `if TYPE_CHECKING:` guards, but they are
   not type-only (they call `build_freespeak_annotations`,
   `transcribe_split`, `grapheme_to_phones` at runtime), so
   `TYPE_CHECKING` would not help here. Only the real fix does.
3. **`__init__.py` re-exports are benign.** `p014/__init__.py` only
   pulls in three names from `config`. `features/__init__.py` and
   `freespeak/__init__.py` re-export their own submodules; they do not
   reach across the package, so they do not amplify coupling.
4. **`data.py` is the largest module (670 LOC)** and is doing two jobs
   (schema + freespeak orchestration). Splitting it as suggested in §3
   also addresses this size/responsibility concern.
5. **No star imports, no circular `from __future__` issues, no duplicated
   top-level imports.** Clean otherwise.

## TL;DR

One cycle, already masked by a function-local-import hack in
`data.py:182-188`. The fix is a ~30-line refactor: pull
`HF_DATASET_ID`, `ReadAloudAnnotation`, `UTTERANCE_SCORE_KEYS`,
`WORD_SCORE_KEYS` (and their dict helpers) into `p014/schema.py`, have
`data.py`, `metrics.py`, and `freespeak/{asr,annotation}.py` import from
that leaf, and delete the deferred imports. After that the graph is a
clean DAG with the layering `schema / config / blocks / features-leaves
/ cono / freespeak-leaves → model / data / freespeak.annotation →
metrics / training → cli`.
