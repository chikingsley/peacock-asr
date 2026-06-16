"""Train Tajik omni CTC models — pure config over :mod:`omni_finetune_core.project`.

Presets are typed ``TrainingConfig`` factories (proven field-equivalent to the YAML recipe
configs they replaced); ``tajik-train --preset tajik-corpus-v3-300m`` runs the shipping recipe.
Past experiments (v0/v1/v2) live in EXPERIMENTS.md + git history; their runs are complete.
"""

from __future__ import annotations

from omni_curator.process import normalize
from omni_finetune_core.presets import gpu_max_finetune
from omni_finetune_core.project import FinetuneProject, TrainingPreset, train_main

from tajik_asr import DATA, LANGUAGE, ROOT

PROJECT = FinetuneProject(
    name="tajik",
    language=LANGUAGE,
    root=ROOT,
    presets={
        # v3 = the shipping corpus (held-out excluded). The exact recipe that produced
        # omni_ctc_300m_v2_tajik_v3_step_20000: weighted FLEURS mix (anti-drift), lr 5e-6
        # tri-stage (recipe default), 20k steps ≈ 1 epoch, fragment cache on real disk.
        "tajik-corpus-v3-300m": TrainingPreset(
            config=lambda: gpu_max_finetune(
                model="omni_ctc_300m_v2_base",
                dataset="tajik_asr_corpus_v3",
                tokenizer="omni_asr_tokenizer_written_v2_local",
                dataset_summary_path=str(
                    DATA / "datasets/v3/language_distribution_weighted.tsv"
                ),
                num_steps=20_000,
                lr=5e-6,
                validate_every=500,
                fragment_cache_dir=str(DATA / "cache/fragments"),
            ),
            output_dir=ROOT / "runs/omni-ctc-300m-tajik-asr-corpus-v3",
        ),
    },
    # Generic-regime config (--regime gpu_max|1b|warm_restart): same cards + the v3 weighted
    # mixture, for ad-hoc runs beyond the pinned preset.
    model_card="omni_ctc_300m_v2_base",
    dataset_card="tajik_asr_corpus_v3",
    tokenizer_card="omni_asr_tokenizer_written_v2_local",
    dataset_summary_path=DATA / "datasets/v3/language_distribution_weighted.tsv",
    fragment_cache_dir=DATA / "cache/fragments",
    default_output_dir=ROOT / "runs/omni-ctc-300m-tajik-asr-corpus-v3-adhoc",
    eval_dataset_root=DATA / "datasets/v3/version=0",
    eval_models={
        "base": "omni_ctc_300m_v2_base",
        "v0": "omni_ctc_300m_v2_tajik_step_1800",
        "v3": "omni_ctc_300m_v2_tajik_v3_step_20000",
    },
    normalize=lambda text: normalize(text, LANGUAGE),
)


def main(argv: list[str] | None = None) -> int:
    return train_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
