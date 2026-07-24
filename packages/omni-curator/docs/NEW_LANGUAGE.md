# Adding a language

A language project is **pure config**. Curation logic and data layout live in shared packages; the project supplies source metadata, paths, model-family configuration, and thin console entry points. A project may use Omni, Parakeet, or both; do not add empty family packages or commands.

## Step 0 — find the data (always run this checklist)

Before scraping anything, sweep the open sources — for many languages a usable corpus already exists. Check **all** of these for the new language, every time:

1. **Google FLEURS** (`google/fleurs` on HF) — read speech, the benchmark anchor. Find the `xx_yy` config; if there's none, the language has no FLEURS (e.g. Dari).
1. **Common Voice** — the **Mozilla Data Collective** (MDC, not the old HF mirror): grab the dataset id + download with the MDC key. Maintain the running MDC id list as it's assembled.
1. **OpenSLR** — <https://www.openslr.org/resources.php>. **Always check** — a large share of low/mid-resource speech corpora live here (Google-funded SLR sets, university corpora, TTS data usable as clean ASR pairs).
1. **Hugging Face datasets, ASR filter** — <https://huggingface.co/datasets?task_categories=task_categories:automatic-speech-recognition> (and search the language name / ISO code; also browse by *modality: audio*). YODAS, MLS, community scrapes, etc.
1. **Multilingual LibriSpeech / LibriVox-derived** (MLS, M-AILABS) — public-domain audiobook read speech for the bigger languages.
1. **General web + paper search** — look for any released corpus (university, NGO, government, a paper's supplementary data). A lot exists outside the aggregators if you search the language + "speech corpus" / "ASR dataset".
1. **YouTube scrape (the create path)** — the fallback / conversational lever when little labelled data exists (Tajik, Dari, Georgian). Channel registry in `sources.py`.

Ingest sources (1–6) preserve their splits and need no labelling; the create path (7) is Scribe-labelled. A high-resource language (Russian) is mostly 1–6; a low-resource one (Dari) is mostly 7.

## The recipe

1. **Create the smallest project skeleton for the selected model family**:

   ```text
   projects/<lang>-asr/
     pyproject.toml          deps and console commands for the selected families only
     src/<lang>_asr/
       __init__.py           LANGUAGE ("xxx_Yyyy"), SCRIPT, ROOT/DATA/DB path constants
       sources.py            the language config: FLEURS config, Common Voice ids, channel
                             registry (omni_curator.create.youtube.Channel via channel())
       curate.py             ~45 lines: build a CuratorProject, delegate to project.main
       omni/                 optional Omni config, train, and eval adapters
       parakeet/             optional ParakeetProject, train, materialize, eval, artifacts.json
   ```

1. **Fill `sources.py`**: the FLEURS config code, MDC Common Voice dataset ids, and the vetted YouTube channel registry. Channel policy: a channel qualifies when a meaningful share of its content is the target language; only pure music/song channels are skipped. **Bilingual channels are safe only once the language has a gate registered** (step 4) — without one, `keep_for_language` keeps everything and the contaminant language trains in.

1. **Add ONE normalizer function in the package** — `omni_curator/process/normalize.py` `NORMALIZERS` registry, keyed by the language code. This is the only per-language *code*; it lives in the package, never in the project.

1. **(Optional) a language gate** — `omni_curator/process/language.py` `LANGUAGE_GATES`, when the language has a dominant contaminant to filter (see the Tajik-vs-Russian gate: exclusive letters first, function-word vocabulary tiebreak).

1. **Wire the project config** (`curate.py`): register ingest sources (`fleurs_source` / `commonvoice_source` / any `IngestFn`), and build the export coverage gate with `omni_curator.coverage.char_tokenizer_coverage(<tokenizer .model>)` — the tokenizer + fairseq2/omni-finetune-core live in the project venv, not the package.

1. **Run the pipeline**. Existing segmented corpora go directly through ingest; the create/VAD path is only for long unsegmented audio. Quality thresholds are source policies, not global language defaults:

   ```text
   <lang>-curate ingest <source>           # preserve authoritative labels and source provenance
   <lang>-curate download                 # channel audio -> data/create/<slug>
   <lang>-curate enqueue / segment / labelq / harvest   # the split create pipeline
   <lang>-curate merge && <lang>-curate verify          # master store + script-aware scoring
   <lang>-curate export v0 --max-duration 30             # no universal Scribe-WER gate
   ```

   Only exclude structural failures by default: missing/unreadable audio, empty/descriptor-only labels, confirmed wrong language, duplicates/leakage, nonpositive duration, and the target model's duration cap. Treat Scribe disagreement, edge error, CTC alignment, and rate heuristics as additive audit signals. An automatic source-specific exclusion threshold must first achieve at least 95% reject precision on human review and then win an equal-hour, matched two-seed training ablation. Never quality-filter dev/test.

1. **Before the first training run**: freeze a held-out conversational test manifest (whole videos — see `tajik heldout_test_videos.json` and `Selection.heldout_test_videos`), and set the project's `mixture_weights` if a small clean corpus needs lifting against a large conversational mass (tajik's anti-drift recipe: `{"fleurs": 490.0}`).

1. **Train + eval**: `<lang>-omni-train --regime gpu_max` budgets ~30 epochs from the export's true hours; once a recipe proves out, pin it as a named `TrainingPreset` (typed config — see tajik's `tajik-corpus-v3-300m`). Always set `fragment_cache_dir` to real disk. Register the best checkpoint as a ModelCard in `assets.py`, add it to `eval_models`, and score with `<lang>-omni-eval` (`--only-corpus-prefix youtube-` = the conversational held-out alone).

## The decode stack — after the CTC is trained

Fine-tuning the CTC is the *acoustic* half. The decode stack on top is where the per-language accuracy/speed trade is set. In order of effort:

1. **Greedy CTC** — the production baseline. Fast, no extra work.
1. **+ KenLM fusion (always do this)** — build a word n-gram from the language's own training text (`omni-build-lm <export>/version=0 <out_dir>`), fuse at decode (the `lm_decoding` experiment). Proven ~−16 % relative on Tajik for **zero training**, CPU-only. This is the guaranteed floor uplift for every language.
1. **+ NAR editor (the upside, where the omni Llama covers the language)** — a frozen omni Llama decoder edits the CTC draft in one bidirectional pass: LLM-class accuracy without autoregressive latency. Per language you only train a small LoRA + projector (the frozen CTC and frozen shared Llama are reused). See `projects/tajik-asr/experiments/nar/`. Falls back to KenLM where the Llama doesn't know the language well enough.

**Steady-state per new language:** find data (Step 0) → fine-tune the CTC → build the KenLM → (once NAR is validated) train the small NAR adapter. The shared Llama brain is never retrained; only the per-language CTC + adapter are. CTC+KenLM is always the floor, so a language is never worse than that baseline.

## What the project may NOT contain

Curation logic, normalizers, exporters, scoring, queue/segment/label machinery — all package code. If a new language needs a behavior change, change the package (everyone gets the fix); the project only ever gains config values.

## Comparability rules (when iterating datasets)

- Compare models on the **same export's** test partition — gates evolve, so the held-out clip set evolves with them; re-eval old models on the new export rather than across exports.
- Benchmark splits (e.g. FLEURS dev/test) are exported **unfiltered** (`Selection.gated_splits`); the machine-labeled held-out IS gated — a deliberate asymmetry (gold vs machine references).
