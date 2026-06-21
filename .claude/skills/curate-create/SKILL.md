---
name: curate-create
description: Use when generating transcripts for raw, untranscribed audio (YouTube, shows, lesson audio) via the omni-curator create pipeline. Covers the two segmentation paths (vad vs chunks) and when each applies. For datasets that already have labels, use curate-ingest instead.
---

# Curate: create labels for raw audio

Generate labels for audio that has **no transcript**. Pipeline per clip:
segment → Scribe ensemble → consensus → (stitch) → polish.

Two paths differ only in how audio is cut and reassembled:

| path | segmenter | reassembly | use for |
|------|-----------|------------|---------|
| `vad`    | cut at silences, no overlap | labels joined in order | dense, continuous speech (TV banter, multi-speaker) |
| `chunks` | fixed overlapping windows, 100% coverage | `stitch` reconciles seams | clean / sparse / drill-style audio (VAD drops short utterances) |

Pick empirically: on clean French (Pimsleur drills) `chunks` scored ~2% WER vs `vad` ~11% (VAD
dropped short syllables); on dense Tajik TV the two tied, so `vad` (simpler) wins. **Measure on a
labelled sample before committing.** `polish` is a conservative final repair pass on both
(turn off with `--no-polish` / `do_polish=False`).

## CLI — label one local audio file (writes a transcript JSON, not a store row)

```bash
uv run omni-curator <audio> --path vad|chunks --out-dir <dir> \
  --language <L> --script "<S>" --langs auto,<code> \
  [--runs N] [--workers 8] [--chunk 40] [--overlap 10] [--no-polish]
```

Examples:

```bash
# dense Cyrillic speech
uv run omni-curator show.flac --path vad --out-dir out \
  --language Tajik --script "Cyrillic script (tgk_Cyrl)" --langs auto,tgk

# clean / drill audio, Latin script
uv run omni-curator lesson.flac --path chunks --out-dir out \
  --language French --script "standard French orthography" --langs auto,fr
```

`--langs auto` = code-switching auto-detect. Writes `<path>_transcript.json` to `--out-dir`.

## Land labels in the store (library — no CLI yet)

The CLI only writes a transcript file. To get clips **into the curator store**, call the create
runners (`omni_curator.create.run`) from a small project script:

```python
from pathlib import Path
from omni_curator.store import CuratorStore
from omni_curator.create.run import label_to_store, label_youtube

store = CuratorStore(Path("data/curator.sqlite"))

# a local file
label_to_store(Path("show.flac"), store=store, source="show",
               language="Tajik", script="Cyrillic script (tgk_Cyrl)",
               id_prefix="show01", out_dir=Path("data/canonical_audio/show"), path="vad")

# a YouTube url (downloads audio first, then labels)
label_youtube("https://youtu.be/...", store=store, language="Tajik",
              script="Cyrillic script (tgk_Cyrl)", work_dir=Path("data/yt"), path="vad")
store.close()
```

There is **no first-class YouTube/create CLI** in a project yet — source ingestion currently lives
in per-language scripts. Wire these helpers into a project entry point when you need it repeatedly.

## Setup

- ElevenLabs Scribe + the free SuperWhisper text endpoint (`superwhisper-api`). ElevenLabs key
  resolves env → macOS cache → Mac-mirror (`superwhisper_api.auth`).
- `vad` needs `nemo-toolkit[asr]` (frame-VAD checkpoint downloads on first use, or point
  `OMNI_CURATOR_VAD_MODEL` at a local `.nemo`). `chunks` does **not** need NeMo.
- `ffmpeg` on PATH.

Flow position: **create** → store → `curate-verify-export`.
