"""fairseq2 asset cards for the template-shaped Persian package.

The base model (``omniASR_CTC_300M_v2``) and tokenizer (``omniASR_tokenizer_written_v2``) cards
come from the upstream ``omnilingual-asr`` package (auto-downloaded); this module registers only
the project's fine-tuned model card(s) under its own in-memory source.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omni_finetune_core.assets import ModelCard, register_cards

if TYPE_CHECKING:
    from fairseq2.runtime.dependency import DependencyContainer

# Upstream tokenizer card name (provided by the omnilingual-asr package).
TOKENIZER_NAME = "omniASR_tokenizer_written_v2"

# This file lives at src/farsi_asr/assets.py, so parents[2] is the persian-asr project root.
_PROJECT = Path(__file__).resolve().parents[2]

#: The PRODUCTION model. STEP ACCOUNTING (so the name is honest): the scribe-v4 baseline trained
#: to step 34,000, then a warm restart (fresh optimizer + tri_stage schedule, lr 2e-6) loaded those
#: step-34,000 weights and ran 7,000 more steps to its best checkpoint. The run dir labels that
#: checkpoint ``step_7000`` because the warm restart reset the counter — but cumulatively the model
#: has seen 34,000 + 7,000 = 41,000 training steps. So the card is named ``..._v4_step_41000``
#: and the on-disk checkpoint path below is the rewarm's ``step_7000``. Dev WER 11.15, FLEURS
#: 8.51 — a ~0.2-pt (noise) move over the baseline's 8.69; shipped on HF (omni-ctc-300m-farsi).
PRODUCTION_MODEL = "omni_ctc_300m_v2_farsi_v4_step_41000"

CARDS = [
    ModelCard(
        name=PRODUCTION_MODEL,
        checkpoint=(
            _PROJECT
            / "src/finetune_omni/runs/scribe-v4-rewarm10k/ws_1.5318420d"
            / "checkpoints/step_7000/model/pp_00/tp_00/sdp_00.pt"  # rewarm-relative; 34k+7k=41k
        ),
        tokenizer_ref=TOKENIZER_NAME,
    ),
]


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    register_cards(container, "farsi_asr", CARDS)
