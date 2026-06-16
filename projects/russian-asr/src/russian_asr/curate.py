"""Curate Russian ASR data — pure config over :mod:`omni_curator.project`.

  russian-curate list|download|cookies        # source: size + pull YouTube channel audio
  russian-curate enqueue|segment|labelq       # split create pipeline (queue -> VAD -> Scribe)
  russian-curate harvest|merge                # labeled clips -> channel stores -> master store
  russian-curate verify|rescore               # Scribe-score the store (script-aware)
  russian-curate ingest fleurs                # FLEURS ru_ru (existing-labeled)
  russian-curate export vN [--max-wer 0.35]   # store -> omni-parquet ablation

Russian is ingest-heavy: FLEURS wired below; Common Voice when the MDC id is filled; the local
corpora (ru_open_stt / SOVA / TIMIT, under data/) each need a per-dataset ingest adapter (TODO).
"""

from __future__ import annotations

from pathlib import Path

from omni_curator.coverage import char_tokenizer_coverage
from omni_curator.project import CuratorProject, fleurs_source
from omni_curator.project import main as project_main

from russian_asr import DATA, DB, LANGUAGE, ROOT, SCRIPT, sources

_PKG = Path(__file__).resolve().parent

PROJECT = CuratorProject(
    name="russian",
    language=LANGUAGE,
    script=SCRIPT,
    data=DATA,
    db=DB,
    channels=sources.YOUTUBE_CHANNELS,
    ingests={
        "fleurs": fleurs_source(sources.FLEURS_CONFIG),
        # "commonvoice": commonvoice_source(sources.COMMONVOICE),  # wire when MDC id filled
        # ru_open_stt / sova / TIMIT: add per-dataset ingest adapters (sources.LOCAL_CORPORA)
    },
    env_file=ROOT.parents[1] / ".env",  # monorepo-root .env (Scribe / HF keys)
    coverage_check=char_tokenizer_coverage(_PKG / "models" / "omniASR_tokenizer_written_v2.model"),
)


def main(argv: list[str] | None = None) -> int:
    return project_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
