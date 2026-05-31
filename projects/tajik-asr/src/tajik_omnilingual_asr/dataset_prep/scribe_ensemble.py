"""Multi-pass / multi-run Scribe ensemble (prototype).

Runs ElevenLabs Scribe over one audio file across several language settings AND repeated
runs per setting, so both cross-language differences and run-to-run variance/quirks are
visible for the "compile-down" (deciding what was actually said, per language/region).

The key resolves device-aware (env var -> macOS cache -> Mac-mirrored), see
``superwhisper_api.auth``.

  uv run python -m tajik_omnilingual_asr.dataset_prep.scribe_ensemble <audio.wav> \
      --langs en,tgk,fas --repeats 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _norm(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for consensus comparison."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def run_pass(
    audio: Path, *, model: str, api_key: str, language: str | None, diarize: bool
) -> dict[str, Any]:
    from superwhisper_api.audio.models import audio_model
    from superwhisper_api.audio.transcribe import create_process_fn

    spec = audio_model(model)
    payload = create_process_fn(spec, api_key, language=language, diarize=diarize)(audio).as_dict()
    raw = payload.get("raw_response")
    raw = raw if isinstance(raw, dict) else {}
    return {
        "detected_language": raw.get("language_code"),
        "language_probability": raw.get("language_probability"),
        "transcript": str(payload.get("transcript") or ""),
        "error": str(payload.get("error") or ""),
    }


def parse_langs(value: str) -> list[tuple[str, str | None]]:
    """'auto' -> detect (None); otherwise the literal language code."""
    out: list[tuple[str, str | None]] = []
    for token in value.split(","):
        label = token.strip()
        if label:
            out.append((label, None if label == "auto" else label))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-pass / multi-run Scribe ensemble.")
    p.add_argument("audio", type=Path)
    p.add_argument("--langs", default="auto,en,tgk,fas", help="comma list; 'auto' = detect")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--key", default=os.environ.get("ELEVENLABS_API_KEY"))
    p.add_argument("--model", default="scribe-v2")
    p.add_argument("--diarize", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.audio.is_file():
        raise SystemExit(f"audio not found: {args.audio}")
    key = args.key
    if not key:
        from superwhisper_api.auth import ensure_elevenlabs_key

        key = ensure_elevenlabs_key()

    results: dict[str, list[dict[str, Any]]] = {}
    for label, language in parse_langs(args.langs):
        runs: list[dict[str, Any]] = []
        for i in range(args.repeats):
            result = run_pass(
                args.audio, model=args.model, api_key=key, language=language, diarize=args.diarize
            )
            runs.append(result)
            print(
                f"[{label} {i + 1}/{args.repeats}] detected={result['detected_language']} "
                f"err={bool(result['error'])} :: {result['transcript'][:150]}"
            )
        results[label] = runs
        norms = [_norm(r["transcript"]) for r in runs if not r["error"]]
        counts = Counter(norms)
        consensus_norm, agree = counts.most_common(1)[0] if counts else ("", 0)
        consensus_raw = next(
            (r["transcript"] for r in runs if _norm(r["transcript"]) == consensus_norm), ""
        )
        n = len(norms) or 1
        verdict = "clean" if agree / n >= 0.7 else "AMBIGUOUS -> agent"  # noqa: PLR2004
        print(f"  -> {label}: consensus {agree}/{len(norms)} agree, {len(counts)} variant(s)  [{verdict}]")
        print(f"     {consensus_raw[:200]}\n")

    out = args.out or args.audio.with_suffix(".scribe_ensemble.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
