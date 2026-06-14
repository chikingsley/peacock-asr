"""Curate Dari ASR data — pure config over :mod:`omni_curator.project`.

  dari-curate list|download|cookies        # source: size + pull YouTube channel audio
  dari-curate enqueue|segment|labelq       # split create pipeline (queue -> VAD -> Scribe)
  dari-curate harvest|merge                # labeled clips -> channel stores -> master store
  dari-curate verify|rescore               # Scribe-score the store (script-aware)
  dari-curate export vN [--max-wer 0.35]   # store -> omni-parquet ablation

No `ingest` subcommand wiring: Dari has no FLEURS/Common Voice/HF dataset, so the create path
(YouTube + Pimsleur -> Scribe) is the only source.
"""

from __future__ import annotations

from pathlib import Path

from omni_curator.coverage import char_tokenizer_coverage
from omni_curator.project import CuratorProject
from omni_curator.project import main as project_main

from dari_asr import DATA, DB, LANGUAGE, ROOT, SCRIPT, sources

_PKG = Path(__file__).resolve().parent

PROJECT = CuratorProject(
    name="dari",
    language=LANGUAGE,
    script=SCRIPT,
    data=DATA,
    db=DB,
    channels=sources.YOUTUBE_CHANNELS,
    ingests={},  # no pre-labeled Dari dataset exists — create path only
    env_file=ROOT.parents[1] / ".env",  # monorepo-root .env (Scribe / HF keys)
    coverage_check=char_tokenizer_coverage(_PKG / "models" / "omniASR_tokenizer_written_v2.model"),
    # Freeze a held-out manifest (whole videos) before the first training run.
)


def main(argv: list[str] | None = None) -> int:
    return project_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
