"""Train Persian omni CTC models — pure config over :mod:`omni_finetune_core.project`.

The pinned preset reconstructs the PRODUCTION recipe
(``src/finetune_omni/training/configs/persian-asr-scribe-v4-rewarm10k-ctc-300m-v2.yaml``)
as a typed config, proven field-equivalent to that YAML with three deliberate notes:

- ``model.name``: the YAML said ``omni_ctc_300m_v2_scribe_v4_20260530_best``, which AT RUN TIME
  pointed at scribe-v4 step_34000; that card was later repointed at the rewarm output, and the
  step_34000 weights are now ``omni_ctc_300m_v2_scribe_v4_baseline_step34000`` — the preset uses
  that name, i.e. the exact weights the original run warm-restarted from.
- ``optimizer.config.lr``: the YAML scalar ``2e-06`` parses as a *string* under YAML 1.1;
  numerically identical to the preset's float.
- ``common``: the preset emits fairseq2's defaults explicitly (``no_sweep_dir=False``,
  ``sweep_format="ws_{world_size}.{hash}"``) where the YAML omitted them — same behavior (the
  original run's ``ws_1.5318420d`` sweep dir proves sweep mode was on), though the explicit keys
  enter the config sha, so a re-run lands in a fresh ``ws_*`` hash dir.

Past experiments (FLEURS/thomcles baselines, 100h ablations, scribe v3/1B) live in
EXPERIMENTS.md and the legacy ``persian-omni-train`` / ``persian-train-omni`` CLIs.
"""

from __future__ import annotations

from omni_curator.process import normalize
from omni_finetune_core.presets import warm_restart
from omni_finetune_core.project import FinetuneProject, TrainingPreset, train_main

from persian_asr import DATA, LANGUAGE, ROOT
from persian_asr.assets import PRODUCTION_MODEL

# The legacy scribe-v4 corpus the production model trained on (mixture TSV next to the parquet).
_SCRIBE_V4_TSV = (
    ROOT / "src/finetune_omni/data/training/omnilingual/scribe-v4/language_distribution_0.tsv"
)

PROJECT = FinetuneProject(
    name="persian",
    language=LANGUAGE,
    root=ROOT,
    presets={
        # The exact production recipe (see module docstring): warm-restart from scribe-v4
        # step_34000 with a fresh optimizer + tri-stage schedule, peak 2e-6, 10k steps.
        # Produced omni_ctc_300m_v2_scribe_v4_20260530_best (step_7000, dev WER 11.15).
        # Output dir is a FRESH home under runs/ — never write into the legacy
        # src/finetune_omni/runs/scribe-v4-rewarm10k artifact.
        "persian-scribe-v4-rewarm10k-300m": TrainingPreset(
            config=lambda: warm_restart(
                checkpoint_card="omni_ctc_300m_v2_scribe_v4_baseline_step34000",
                dataset="scribe_v4",
                tokenizer="omniASR_tokenizer_written_v2",
                dataset_summary_path=str(_SCRIBE_V4_TSV),
                num_steps=10_000,
                peak_lr=2e-6,
                max_num_elements=3_840_000,
                validate_every=500,
                no_sweep_dir=False,  # the original run used a ws_{hash} sweep dir
            ),
            output_dir=ROOT / "runs/omni-ctc-300m-scribe-v4-rewarm10k",
        ),
    },
    # Generic-regime cards (--regime gpu_max|1b|warm_restart): the legacy scribe-v4 corpus until
    # the first new-pipeline export lands (then: a persian_asr_corpus_vN card + its weighted TSV).
    # NOTE: scribe-v4 predates export_summary.json, so --regime needs an explicit --num-steps.
    model_card="omniASR_CTC_300M_v2",
    dataset_card="scribe_v4",
    tokenizer_card="omniASR_tokenizer_written_v2",
    dataset_summary_path=_SCRIBE_V4_TSV,
    fragment_cache_dir=DATA / "cache/fragments",
    default_output_dir=ROOT / "runs/omni-ctc-300m-persian-adhoc",
    # No new-pipeline export yet -> no eval_dataset_root; legacy evals run via the old benchmark
    # suite (persian-benchmark-suite / persian-omni-eval). persian-eval-v2 needs --dataset-root
    # until `persian-curate export vN` produces datasets/vN.
    eval_dataset_root=None,
    eval_models={
        "base": "omniASR_CTC_300M_v2",
        "scribe-v4-rewarm": PRODUCTION_MODEL,
    },
    normalize=lambda text: normalize(text, LANGUAGE),
)


def main(argv: list[str] | None = None) -> int:
    return train_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
