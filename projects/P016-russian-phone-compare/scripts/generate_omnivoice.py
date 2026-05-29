#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate wav files for a JSONL text manifest with k2-fsa/OmniVoice."
    )
    parser.add_argument("--manifest", type=Path, default=Path("manifests/omnivoice_smoke.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/omnivoice_smoke/audio"))
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = list(_read_jsonl(args.manifest))
    if args.limit is not None:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
        import torch
        from omnivoice import OmniVoice
    except ImportError as exc:
        raise SystemExit(
            "OmniVoice generation needs `omnivoice`, `soundfile`, and `torch` installed. "
            "Install in the generation environment with `uv pip install omnivoice`."
        ) from exc

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = OmniVoice.from_pretrained(args.model, device_map=args.device_map, dtype=dtype)

    generated_manifest = args.out_dir.parent / "generated_manifest.jsonl"
    with generated_manifest.open("w", encoding="utf-8") as manifest_out:
        for index, row in enumerate(rows, start=1):
            sample_id = str(row["id"])
            output_path = args.out_dir / f"{sample_id}.wav"
            if output_path.exists() and not args.overwrite:
                print(f"[{index}/{len(rows)}] exists {output_path}")
            else:
                print(f"[{index}/{len(rows)}] generating {sample_id}: {row['text']}")
                kwargs: dict[str, Any] = {
                    "text": row["text"],
                    "num_step": args.num_step,
                    "speed": float(row.get("speed", args.speed)),
                }
                if row.get("instruct"):
                    kwargs["instruct"] = row["instruct"]
                if row.get("language_id"):
                    kwargs["language_id"] = row["language_id"]
                if row.get("duration"):
                    kwargs["duration"] = float(row["duration"])
                audio = model.generate(**kwargs)
                sf.write(output_path, audio[0], 24_000)

            enriched = dict(row)
            enriched["audio"] = str(output_path)
            enriched["source"] = "omnivoice"
            manifest_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    print(f"Wrote {generated_manifest}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "id" not in row or "text" not in row or "language" not in row:
                raise ValueError(f"{path}:{line_number} needs id, text, and language")
            rows.append(row)
    return rows


if __name__ == "__main__":
    main()
