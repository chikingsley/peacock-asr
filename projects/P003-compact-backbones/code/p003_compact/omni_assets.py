from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
V4_OMNI_ROOT = (
    REPO_ROOT
    / "projects"
    / "P004-training-from-scratch"
    / "third_party"
    / "omnilingual-asr"
)
OMNI_ASSET_ROOT = PROJECT_ROOT / "code" / "omni"
OMNI_TOKENIZER_ROOT = OMNI_ASSET_ROOT / "tokenizers"
OMNI_CARD_ROOT = OMNI_ASSET_ROOT / "cards"
PHONEME_TOKENIZER_NAME = "omniASR_tokenizer_arpabet_41_v1"
PHONEME_MODEL_NAME = "omniASR_CTC_300M_v2_arpabet_41"
STOCK_MODEL_NAME = "omniASR_CTC_300M_v2"
STOCK_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M-v2.pt"
VOCAB_JSON = PROJECT_ROOT / "code" / "training" / "vocab.json"


@dataclass(frozen=True)
class OmniPhonemeAssets:
    tokenizer_dir: Path
    tokenizer_model: Path
    tokenizer_vocab: Path
    tokenizer_mapping: Path
    card_dir: Path
    tokenizer_card: Path
    model_card: Path


def omni_phoneme_assets(base_dir: Path | None = None) -> OmniPhonemeAssets:
    if base_dir is None:
        tokenizer_dir = OMNI_TOKENIZER_ROOT / "arpabet_41_spm"
        card_dir = OMNI_CARD_ROOT
    else:
        tokenizer_dir = base_dir / "tokenizers" / "arpabet_41_spm"
        card_dir = base_dir / "cards"
    return OmniPhonemeAssets(
        tokenizer_dir=tokenizer_dir,
        tokenizer_model=tokenizer_dir / "arpabet_41.model",
        tokenizer_vocab=tokenizer_dir / "arpabet_41.vocab",
        tokenizer_mapping=tokenizer_dir / "token_to_id.json",
        card_dir=card_dir,
        tokenizer_card=card_dir / "tokenizer_arpabet_41.yaml",
        model_card=card_dir / "model_omniasr_ctc_300m_v2_arpabet_41.yaml",
    )


def load_ordered_vocab(vocab_json: Path = VOCAB_JSON) -> list[str]:
    vocab = json.loads(vocab_json.read_text(encoding="utf-8"))
    return [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
