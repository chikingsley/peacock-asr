"""Curate Tajik ASR data — pure config over :mod:`omni_curator.project`.

All commands and curation logic live in the package; this file only supplies the Tajik wiring::

  tajik-curate list|download|cookies        # source: size + pull YouTube channel audio
  tajik-curate enqueue|segment|labelq       # split create pipeline (queue -> VAD -> Scribe)
  tajik-curate harvest|merge                # labeled clips -> channel stores -> master store
  tajik-curate ingest fleurs|commonvoice    # existing-labeled datasets
  tajik-curate verify|rescore               # Scribe-score the store (script-aware)
  tajik-curate export vN [--max-wer 0.35]   # store -> omni-parquet ablation
"""

from __future__ import annotations

from pathlib import Path

from omni_curator.coverage import char_tokenizer_coverage
from omni_curator.project import (
    CuratorProject,
    commonvoice_hf_mirror_source,
    fleurs_source,
    huggingface_source,
)
from omni_curator.project import main as project_main

from tajik_omnilingual_asr import DATA, DB, LANGUAGE, ROOT, SCRIPT, sources

_PKG = Path(__file__).resolve().parent

PROJECT = CuratorProject(
    name="tajik",
    language=LANGUAGE,
    script=SCRIPT,
    data=DATA,
    db=DB,
    channels=sources.YOUTUBE_CHANNELS,
    # No MDC Common Voice for Tajik (sources.COMMONVOICE empty); CV-25 comes via HF instead.
    ingests={
        "fleurs": fleurs_source(sources.FLEURS_CONFIG),
        "commonvoice22": commonvoice_hf_mirror_source(
            *sources.COMMONVOICE_HF_MIRROR, source="hf-commonvoice22"
        ),
        # force_split="train": third-party/augmented datasets never enter the benchmark
        # partition (benchmark splits export ungated — a trust decision per dataset).
        **{
            name: huggingface_source(
                repo, config=config, source=f"hf-{name}", force_split="train"
            )
            for name, (repo, config) in sources.HUGGINGFACE.items()
        },
    },
    env_file=ROOT.parents[1] / ".env",  # monorepo-root .env (Scribe / MDC / HF keys)
    coverage_check=char_tokenizer_coverage(_PKG / "models" / "omniASR_tokenizer_written_v2.model"),
    heldout_manifest=_PKG / "heldout_test_videos.json",
    # The v2/v3 anti-drift recipe: lift FLEURS (true ~11.8 h) to ~12% of sampled batches.
    mixture_weights={"fleurs": 490.0},
)


def main(argv: list[str] | None = None) -> int:
    return project_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
