"""Curate Persian ASR data — pure config over :mod:`omni_curator.project`.

All commands and curation logic live in the package; this file only supplies the Persian wiring::

  persian-curate list|download|cookies        # source: size + pull YouTube channel audio
  persian-curate enqueue|segment|labelq       # split create pipeline (queue -> VAD -> Scribe)
  persian-curate harvest|merge                # labeled clips -> channel stores -> master store
  persian-curate ingest fleurs                # existing-labeled datasets
  persian-curate verify|rescore               # Scribe-score the store (script-aware)
  persian-curate export vN [--max-wer 0.35]   # store -> omni-parquet ablation
"""

from __future__ import annotations

from omni_curator.coverage import char_tokenizer_coverage
from omni_curator.project import CuratorProject, fleurs_source
from omni_curator.project import main as project_main

from farsi_asr import DATA, DB, LANGUAGE, ROOT, SCRIPT, sources

PROJECT = CuratorProject(
    name="persian",
    language=LANGUAGE,
    script=SCRIPT,
    data=DATA,
    db=DB,
    channels=sources.YOUTUBE_CHANNELS,
    # No Common Voice entry yet (sources.COMMONVOICE is empty — the legacy common_voice_25 corpus
    # was a local Mozilla dump, not an MDC id); register commonvoice_source(...) when ids land.
    #
    # TODO: port the legacy corpora as IngestFn sources. NOT done in this pass — their builders
    # live in legacy farsi_asr_dataset code, reading its own canonical-parquet data layout:
    #   - mana_tts: farsi_asr_dataset.canonical.mana_tts_samples, reads data/raw/mana_tts
    #   - neyshekar: farsi_asr_dataset.canonical.neyshekar_samples, reads
    #     data/raw/neyshekar_v3_asr_aligned; repair CLI persian-repair-neyshekar
    #   - worldspeech: farsi_asr_dataset.canonical.worldspeech_samples, reads
    #     data/raw/worldspeech_fa_ir
    #   - thomcles: farsi_asr_dataset.canonical.omni_samples over data/raw/thomcles_persian_omni,
    #     built by farsi_asr_dataset.dataset_prep.thomcles, CLI persian-prepare-omni-thomcles
    #   - common_voice_25: farsi_asr_dataset.canonical.cv25_samples, a local Mozilla CV v25 dump
    ingests={"fleurs": fleurs_source(sources.FLEURS_CONFIG)},
    env_file=ROOT.parents[1] / ".env",  # monorepo-root .env (Scribe / MDC / HF keys)
    # The omni tokenizer .model from the repo-level shared base-model cache — same file the
    # upstream omniASR_tokenizer_written_v2 card downloads; reused as the export coverage gate.
    coverage_check=char_tokenizer_coverage(
        ROOT.parents[1] / "base_models" / "omni" / "omniASR_tokenizer_written_v2.model"
    ),
    # No held-out conversational manifest yet — freeze one before the first new-pipeline training
    # run (see NEW_LANGUAGE.md step 7).
)


def main(argv: list[str] | None = None) -> int:
    return project_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
