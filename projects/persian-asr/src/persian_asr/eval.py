"""Score Persian model cards on an export's test split — delegates to the core eval.

No new-pipeline export exists yet, so ``--dataset-root`` is required for now (the project's
``eval_dataset_root`` is None); legacy evals run via the old benchmark suite
(``persian-benchmark-suite`` / ``persian-omni-eval``).

  persian-eval-v2 --dataset-root data/datasets/v0/version=0
  persian-eval-v2 --dataset-root ... \
      --models prod=omni_ctc_300m_v2_persian_scribe_v4_rewarm_production
"""

from __future__ import annotations

from omni_finetune_core.project import eval_main

from persian_asr.train import PROJECT


def main(argv: list[str] | None = None) -> int:
    return eval_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
