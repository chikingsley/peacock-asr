"""Train Georgian omni CTC models — pure config over :mod:`omni_finetune_core.project`.

  georgian-train --regime gpu_max               # 300M fresh fine-tune, ~30-epoch auto budget
  georgian-train --regime gpu_max --num-steps 6000
  georgian-train --regime warm_restart --lr 2e-6
"""

from __future__ import annotations

from omni_curator.process import normalize
from omni_finetune_core.project import FinetuneProject, train_main

from georgian_asr import DATA, LANGUAGE, ROOT

PROJECT = FinetuneProject(
    name="georgian",
    language=LANGUAGE,
    root=ROOT,
    # Generic-regime config: the v0 export (~145 h). Pin a named preset here once a recipe
    # proves out (see tajik's tajik-corpus-v3-300m for the shape).
    model_card="omni_ctc_300m_v2_georgian_base",
    dataset_card="georgian_asr_corpus",
    tokenizer_card="omni_asr_tokenizer_written_v2_local",
    dataset_summary_path=DATA / "datasets/v0/language_distribution_0.tsv",
    fragment_cache_dir=DATA / "cache/fragments",
    default_output_dir=ROOT / "runs/omni-ctc-300m-georgian-asr-corpus-v0",
    eval_dataset_root=DATA / "datasets/v0/version=0",
    eval_models={"base": "omni_ctc_300m_v2_georgian_base"},
    normalize=lambda text: normalize(text, LANGUAGE),
)


def main(argv: list[str] | None = None) -> int:
    return train_main(PROJECT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
