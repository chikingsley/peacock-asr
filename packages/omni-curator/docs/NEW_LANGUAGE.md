# Adding a language

A language project is **pure config**. All curation logic, every CLI command, and the data
layout live in `omni_curator.project`; the project supplies values. Georgian and tajik are the
reference implementations — identical in shape, differing only in their config values.

## The recipe

1. **Create the project skeleton** (copy from `projects/georgian-asr`):

   ```text
   projects/<lang>-asr/
     pyproject.toml          deps: omni-curator[ingest,youtube,normalize] (+ omni-finetune-core
                             for the model side); a <lang>-curate script entry point
     src/<lang>_asr/
       __init__.py           LANGUAGE ("xxx_Yyyy"), SCRIPT, ROOT/DATA/DB path constants
       sources.py            the language config: FLEURS config, Common Voice ids, channel
                             registry (omni_curator.create.youtube.Channel via channel())
       curate.py             ~45 lines: build a CuratorProject, delegate to project.main
       models/               (gitignored) the omni tokenizer + base checkpoint
   ```

2. **Fill `sources.py`**: the FLEURS config code, MDC Common Voice dataset ids, and the vetted
   YouTube channel registry. Channel policy: a channel qualifies when a meaningful share of its
   content is the target language; only pure music/song channels are skipped. **Bilingual
   channels are safe only once the language has a gate registered** (step 4) — without one,
   `keep_for_language` keeps everything and the contaminant language trains in.

3. **Add ONE normalizer function in the package** — `omni_curator/process/normalize.py`
   `NORMALIZERS` registry, keyed by the language code. This is the only per-language *code*;
   it lives in the package, never in the project.

4. **(Optional) a language gate** — `omni_curator/process/language.py` `LANGUAGE_GATES`, when
   the language has a dominant contaminant to filter (see the Tajik-vs-Russian gate: exclusive
   letters first, function-word vocabulary tiebreak).

5. **Wire the project config** (`curate.py`): register ingest sources
   (`fleurs_source` / `commonvoice_source` / any `IngestFn`), and build the export coverage
   gate with `omni_curator.coverage.char_tokenizer_coverage(<tokenizer .model>)` — the
   tokenizer + fairseq2/omni-finetune-core live in the project venv, not the package.

6. **Run the pipeline**:

   ```text
   <lang>-curate ingest fleurs            # benchmark splits (dev/test ride along, never gated)
   <lang>-curate download                 # channel audio -> data/create/<slug>
   <lang>-curate enqueue / segment / labelq / harvest   # the split create pipeline
   <lang>-curate merge && <lang>-curate verify          # master store + script-aware scoring
   <lang>-curate export v0 --max-wer 0.35               # gated omni-parquet ablation
   ```

7. **Before the first training run**: freeze a held-out conversational test manifest (whole
   videos — see `tajik heldout_test_videos.json` and `Selection.heldout_test_videos`), and set
   the project's `mixture_weights` if a small clean corpus needs lifting against a large
   conversational mass (tajik's anti-drift recipe: `{"fleurs": 490.0}`).

## What the project may NOT contain

Curation logic, normalizers, exporters, scoring, queue/segment/label machinery — all package
code. If a new language needs a behavior change, change the package (everyone gets the fix);
the project only ever gains config values.

## Comparability rules (when iterating datasets)

- Compare models on the **same export's** test partition — gates evolve, so the held-out clip
  set evolves with them; re-eval old models on the new export rather than across exports.
- Benchmark splits (e.g. FLEURS dev/test) are exported **unfiltered** (`Selection.gated_splits`);
  the machine-labeled held-out IS gated — a deliberate asymmetry (gold vs machine references).
