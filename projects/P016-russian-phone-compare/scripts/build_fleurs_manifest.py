#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LANGUAGE_CONFIGS = {
    "en_us": "en_us",
    "ru": "ru_ru",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a small FLEURS audio manifest for real-speech smoke tests."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/fleurs_smoke"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--per-language", type=int, default=5)
    parser.add_argument("--languages", nargs="+", default=["en_us", "ru"])
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
            config = LANGUAGE_CONFIGS.get(language, language)
            print(f"Loading google/fleurs config={config} split={args.split}")
            dataset = load_dataset(
                "google/fleurs",
                config,
                split=args.split,
                streaming=True,
                trust_remote_code=True,
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
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
