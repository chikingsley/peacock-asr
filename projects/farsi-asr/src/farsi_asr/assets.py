"""fairseq2 asset cards for the template-shaped Persian package.

Card-collision resolution: fairseq2's ``StandardAssetStore`` raises ``AssetMetadataError`` at
store construction when two metadata providers register the same card name, and BOTH this
package's ``fairseq2.extension`` entry point and the legacy one
(``farsi_omnilingual_asr.fairseq2_assets``) load on every fairseq2 init. So this module does
NOT redefine the legacy cards — it imports and re-exports them (every existing name keeps
resolving through the legacy extension, unchanged) and registers only NEW names under its own
in-memory source.

The base model (``omniASR_CTC_300M_v2``) and tokenizer (``omniASR_tokenizer_written_v2``) cards
come from the upstream ``omnilingual-asr`` package (auto-downloaded), as in the legacy module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omni_finetune_core.assets import ModelCard, register_cards

# Re-exported so template-side code has one import surface for every Persian card.
from farsi_omnilingual_asr.fairseq2_assets import (
    CARDS as LEGACY_CARDS,  # noqa: F401
)
from farsi_omnilingual_asr.fairseq2_assets import (
    TOKENIZER_NAME,
)

if TYPE_CHECKING:
    from fairseq2.runtime.dependency import DependencyContainer

# This file lives at src/farsi_asr/assets.py, so parents[2] is the persian-asr project root.
_PROJECT = Path(__file__).resolve().parents[2]

#: The PRODUCTION model: the 300M scribe-v4 re-warm best checkpoint (step_7000, dev WER 11.15;
#: benchmarked as benchmarks/suites/canonical-tests-scribe-v4-rewarm-20260530 — it beat the 1B
#: on every benchmark). Same checkpoint the legacy ``omni_ctc_300m_v2_scribe_v4_20260530_best``
#: card points at, under a stable production alias for the template-side eval surface.
PRODUCTION_MODEL = "omni_ctc_300m_v2_farsi_production"

CARDS = [
    ModelCard(
        name=PRODUCTION_MODEL,
        checkpoint=(
            _PROJECT
            / "src/finetune_omni/runs/scribe-v4-rewarm10k/ws_1.5318420d"
            / "checkpoints/step_7000/model/pp_00/tp_00/sdp_00.pt"
        ),
        tokenizer_ref=TOKENIZER_NAME,
    ),
]


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    # Only the NEW cards: the legacy extension already registers LEGACY_CARDS, and duplicate
    # names across providers are a hard fairseq2 error (see module docstring).
    register_cards(container, "farsi_asr", CARDS)
