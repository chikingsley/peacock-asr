#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Short aliases -> google/fleurs config name. Any FLEURS config code (e.g. "fr_fr",
# "es_419", "fa_ir") can be passed directly via --languages and is used as-is.
LANGUAGE_CONFIGS = {
    "en_us": "en_us",
    "ru": "ru_ru",
}

# A diverse default spread so the two-path eval exercises G2P across scripts/families.
DEFAULT_LANGUAGES = [
    "en_us", "ru_ru", "fr_fr", "de_de", "es_419",
    "it_it", "fa_ir", "hi_in", "tr_tr", "ja_jp",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a small FLEURS audio manifest for real-speech smoke tests."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/fleurs_smoke"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--per-language", type=int, default=5)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help="FLEURS config codes (e.g. en_us ru_ru fr_fr es_419 fa_ir). Short aliases "
        "in LANGUAGE_CONFIGS are expanded; anything else is used as the config directly.",
    )
    args = parser.parse_args()

    try:
        import soundfile as sf
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "FLEURS manifest building needs `datasets` and `soundfile`. "
            "Install with `uv pip install datasets soundfile`."
        ) from exc

    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"

    seen_ids: set[str] = set()
    with manifest_path.open("w", encoding="utf-8") as out:
        for language in args.languages:
            try:
                _emit_language(out, audio_dir, language, args, load_dataset, sf, seen_ids)
            except Exception as exc:  # noqa: BLE001 - skip a language that fails to load/download
                print(f"  skip {language}: {exc}")
    print(f"Wrote {manifest_path}")


def _emit_language(
    out: Any,
    audio_dir: Path,
    language: str,
    args: argparse.Namespace,
    load_dataset: Any,
    sf: Any,
    seen_ids: set[str],
) -> None:
    config = LANGUAGE_CONFIGS.get(language, language)
    print(f"Loading google/fleurs config={config} split={args.split}")
    dataset = load_dataset(
        "google/fleurs", config, split=args.split, streaming=True, trust_remote_code=True
    )
    count = 0
    for example in dataset:
        sample_id = f"fleurs_{language}_{example['id']}"
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        audio = example["audio"]
        wav_path = audio_dir / f"{sample_id}.wav"
        sf.write(wav_path, audio["array"], audio["sampling_rate"])
        row: dict[str, Any] = {
            "id": sample_id,
            "source": "google/fleurs",
            "language": language,
            "category": "fleurs_real_speech",
            "text": example.get("transcription") or example.get("raw_transcription", ""),
            "raw_transcription": example.get("raw_transcription", ""),
            "audio": str(wav_path),
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
        if count >= args.per_language:
            break


if __name__ == "__main__":
    main()
