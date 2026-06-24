"""Curate Russian ASR data — pure config over :mod:`omni_curator.project`.

  russian-curate list|download|cookies        # source: size + pull YouTube channel audio
  russian-curate enqueue|segment|labelq       # split create pipeline (queue -> VAD -> Scribe)
  russian-curate harvest|merge                # labeled clips -> channel stores -> master store
  russian-curate verify|rescore               # Scribe-score the store (script-aware)
  russian-curate ingest fleurs|rulibrispeech|sova_rudevices|golos_crowd|golos_farfield|tonebooks
  russian-curate export vN [--max-wer 0.35]   # store -> omni-parquet ablation

Russian is ingest-heavy: FLEURS (benchmark) + the HF clean corpora below; Common Voice when the
user's pipeline provides the MDC id; full Golos / M-AILABS / RUSLAN and the local raw corpora
(ru_open_stt / SOVA / TIMIT) each need a per-dataset adapter (TODO, see sources.py).
"""

from __future__ import annotations

from pathlib import Path

from omni_curator.audit.coverage import char_tokenizer_coverage
from omni_curator.ingest.espeech import espeech_source
from omni_curator.ingest.local_corpora import local_corpus_source
from omni_curator.ingest.openslr import openslr_source
from omni_curator.project import CuratorProject, fleurs_source, huggingface_source
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
        # FLEURS ru_ru is the gold benchmark (dev/test ride along, never gated).
        "fleurs": fleurs_source(sources.FLEURS_CONFIG),
        # HF corpora -> source name = the dict key (so it matches the LICENSES registry); forced to
        # train so their own splits don't enter the benchmark.
        **{
            name: huggingface_source(repo, source=name, text_column=col, force_split="train")
            for name, (repo, col) in sources.HUGGINGFACE.items()
        },
        # ESpeech split-tar datasets (download+extract+parse the per-recording <uuid>.json).
        **{name: espeech_source(repo, source=name) for name, repo in sources.ESPEECH.items()},
        # Pure OpenSLR releases (Golos CC-BY-SA, RuLS public domain).
        "golos": openslr_source("114", source="golos"),  # ~1,240 h crowd+farfield
        "ruls": openslr_source("96", source="ruls"),     # ~98 h audiobooks
        # Local corpora (TIMIT excluded — English LDC, quarantined). SOVA empty until downloaded;
        # ru_open_stt only partly extracted (rest in data/ru_open_stt/archives/*.tar.gz).
        "ru_open_stt": local_corpus_source("ru_open_stt", DATA / "ru_open_stt"),
        "sova_dataset": local_corpus_source("sova_dataset", DATA / "sova_dataset"),
        # commonvoice: add commonvoice_source(sources.COMMONVOICE) when the MDC id is filled.
    },
    licenses=sources.LICENSES,  # stamped on every exported row; --commercial-only drops NC sources
    env_file=ROOT.parents[1] / ".env",  # monorepo-root .env (Scribe / HF keys)
    coverage_check=char_tokenizer_coverage(_PKG / "models" / "omniASR_tokenizer_written_v2.model"),
)


def main(argv: list[str] | None = None) -> int:
    return project_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
