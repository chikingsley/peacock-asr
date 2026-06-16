"""Train Dari omni CTC models — pure config over :mod:`omni_finetune_core.project`.

  dari-train --regime gpu_max                # 300M cold fine-tune from the omni base
  dari-train --regime warm_restart --lr 2e-6 # warm-start from the Farsi production checkpoint
  dari-train --preset dari-corpus-v0-300m    # pinned, exactly reproducible

Plan: train BOTH the cold-base fine-tune (template-conformant baseline) and the Farsi-warm restart
(likely winner — Dari shares fas_Arab with Iranian Farsi), compare on the same v0 test split.
"""

from __future__ import annotations

from omni_curator.process import normalize
from omni_finetune_core.presets import gpu_max_finetune
from omni_finetune_core.project import FinetuneProject, TrainingPreset, train_main

from dari_asr import DATA, LANGUAGE, ROOT

PROJECT = FinetuneProject(
    name="dari",
    language=LANGUAGE,
    root=ROOT,
    presets={
        # v0 = the first Dari export (YouTube + Pimsleur). num_steps is a placeholder until the
        # export prints real hours — set it for 3-5 epochs at that hour count before launching.
        "dari-corpus-v0-300m": TrainingPreset(
            config=lambda: gpu_max_finetune(
                model="omni_ctc_300m_v2_dari_base",
                dataset="dari_asr_corpus",
                tokenizer="omni_asr_tokenizer_written_v2_local",
                dataset_summary_path=str(DATA / "datasets/v0/language_distribution_0.tsv"),
                num_steps=20_000,
                lr=1e-5,
                validate_every=1_000,
                fragment_cache_dir=str(DATA / "cache/fragments"),
            ),
            output_dir=ROOT / "runs/omni-ctc-300m-dari-asr-corpus-v0",
        ),
    },
    model_card="omni_ctc_300m_v2_dari_base",
    dataset_card="dari_asr_corpus",
    tokenizer_card="omni_asr_tokenizer_written_v2_local",
    dataset_summary_path=DATA / "datasets/v0/language_distribution_0.tsv",
    fragment_cache_dir=DATA / "cache/fragments",
    default_output_dir=ROOT / "runs/omni-ctc-300m-dari-asr-corpus-v0",
    eval_dataset_root=DATA / "datasets/v0/version=0",
    eval_models={"base": "omni_ctc_300m_v2_dari_base"},
    normalize=lambda text: normalize(text, LANGUAGE),
)


def main(argv: list[str] | None = None) -> int:
    return train_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
