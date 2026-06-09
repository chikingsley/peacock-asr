"""Curate Georgian ASR data — pure config over :mod:`omni_curator.project`.

All commands and curation logic live in the package; this file only supplies the Georgian wiring::

  georgian-curate list|download|cookies        # source: size + pull YouTube channel audio
  georgian-curate enqueue|segment|labelq       # split create pipeline (queue -> VAD -> Scribe)
  georgian-curate harvest|merge                # labeled clips -> channel stores -> master store
  georgian-curate ingest fleurs|commonvoice    # existing-labeled datasets
  georgian-curate verify|rescore               # Scribe-score the store (script-aware)
  georgian-curate export vN [--max-wer 0.25]   # store -> omni-parquet ablation
"""

from __future__ import annotations

from pathlib import Path

from omni_curator.coverage import char_tokenizer_coverage
from omni_curator.project import CuratorProject, commonvoice_source, fleurs_source
from omni_curator.project import main as project_main

from georgian_asr import DATA, DB, LANGUAGE, ROOT, SCRIPT, sources

_PKG = Path(__file__).resolve().parent

PROJECT = CuratorProject(
    name="georgian",
    language=LANGUAGE,
    script=SCRIPT,
    data=DATA,
    db=DB,
    channels=sources.YOUTUBE_CHANNELS,
    ingests={
        "fleurs": fleurs_source(sources.FLEURS_CONFIG),
        "commonvoice": commonvoice_source(sources.COMMONVOICE),
    },
    env_file=ROOT.parents[1] / ".env",  # monorepo-root .env (Scribe / MDC / HF keys)
    coverage_check=char_tokenizer_coverage(_PKG / "models" / "omniASR_tokenizer_written_v2.model"),
    # No held-out manifest yet — freeze one (whole videos) before the first training run,
    # and no tuned mixture recipe yet — set one if/when a small clean corpus needs lifting.
)


def main(argv: list[str] | None = None) -> int:
    return project_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
