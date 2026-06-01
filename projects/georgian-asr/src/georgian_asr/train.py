"""Fine-tune the omni 300M CTC v2 model on the Georgian corpus.

Entry point ``georgian-train``. Builds a typed ``TrainingConfig`` from
:mod:`omni_finetune_core.presets` wired to the Georgian asset cards (registered via
``assets.py``) and hands it to :func:`omni_finetune_core.train.train`, which writes the recipe
YAML into the run dir and invokes the in-housed wav2vec2-ASR recipe.

  georgian-train                              # gpu_max 300M, auto step budget (~30 epochs)
  georgian-train --num-steps 6000             # override the step budget
  georgian-train --regime 1b --lr 1e-5        # 1B regime instead
  georgian-train --regime warm_restart        # second-wind from the base card's weights

Georgian v0 is ~145 h of audio (see data/datasets/v0/export_summary.json); the default step
budget targets ~30 epochs via :func:`presets.recommend_num_steps`. Read the real steps/epoch off
the first epoch's logs and early-stop on the dev-WER plateau rather than trusting the estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from georgian_asr import ROOT

if TYPE_CHECKING:
    from omni_finetune_core.config import TrainingConfig

# Card names registered in assets.py.
MODEL_CARD = "omni_ctc_300m_v2_georgian_base"
DATASET_CARD = "georgian_asr_corpus"
TOKENIZER_CARD = "omni_asr_tokenizer_written_v2_local"

DATASET_DIR = ROOT / "data" / "datasets" / "v0"
SUMMARY_PATH = DATASET_DIR / "language_distribution_0.tsv"
EXPORT_SUMMARY = DATASET_DIR / "export_summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "omni-ctc-300m-georgian-asr-corpus-v0"

REGIMES = ("gpu_max", "1b", "warm_restart")


def corpus_hours() -> float:
    """Total audio hours in the v0 export, read from export_summary.json (falls back to the tsv)."""
    if EXPORT_SUMMARY.exists():
        return float(json.loads(EXPORT_SUMMARY.read_text(encoding="utf-8"))["hours"])
    # tsv fallback: corpus<TAB>language<TAB>hours, with a header row.
    total = 0.0
    for line in SUMMARY_PATH.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            total += float(line.split("\t")[2])
    return total


def build_config(regime: str, num_steps: int, lr: float | None) -> TrainingConfig:
    """Build the typed TrainingConfig for ``regime`` wired to the Georgian cards.

    ``lr`` overrides the preset's default peak LR when given (warm_restart's ``peak_lr``).
    """
    from omni_finetune_core import presets

    summary_path = str(SUMMARY_PATH)
    if regime == "gpu_max":
        return presets.gpu_max_finetune(
            model=MODEL_CARD,
            dataset=DATASET_CARD,
            tokenizer=TOKENIZER_CARD,
            dataset_summary_path=summary_path,
            num_steps=num_steps,
            lr=lr if lr is not None else 1e-5,
        )
    if regime == "1b":
        return presets.gpu_max_finetune_1b(
            model=MODEL_CARD,
            dataset=DATASET_CARD,
            tokenizer=TOKENIZER_CARD,
            dataset_summary_path=summary_path,
            num_steps=num_steps,
            lr=lr if lr is not None else 1e-5,
        )
    # warm_restart: load weights from the base card with a fresh optimizer + lower peak LR.
    return presets.warm_restart(
        checkpoint_card=MODEL_CARD,
        dataset=DATASET_CARD,
        tokenizer=TOKENIZER_CARD,
        dataset_summary_path=summary_path,
        num_steps=num_steps,
        peak_lr=lr if lr is not None else 2e-6,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune omni CTC on the Georgian corpus.")
    p.add_argument("--regime", choices=REGIMES, default="gpu_max")
    p.add_argument("--num-steps", type=int, default=None, help="default: ~30 epochs from hours")
    p.add_argument("--lr", type=float, default=None, help="peak LR (warm_restart: peak_lr)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from omni_finetune_core import presets
    from omni_finetune_core.train import configure_environment, train

    configure_environment(ROOT)

    num_steps = args.num_steps
    if num_steps is None:
        num_steps = presets.recommend_num_steps(corpus_hours(), target_epochs=30)

    cfg = build_config(args.regime, num_steps, args.lr)
    output_dir = args.output_dir.resolve()
    print(
        f"regime={args.regime} num_steps={num_steps} hours={corpus_hours():.1f} "
        f"output_dir={output_dir}",
        flush=True,
    )
    train(cfg, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
