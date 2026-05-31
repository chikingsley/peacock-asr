# Archived dataset-prep scripts

One-time or superseded scripts, kept for provenance. **Not part of the day-to-day
pipeline** and intentionally have **no `[project.scripts]` entry point**. Run via
`uv run python -m tajik_omnilingual_asr.dataset_prep.archive.<name>` if ever needed.

| Script | Why archived |
|---|---|
| `hf_raw.py` | One-time downloader/inventory of raw Tajik HF data. Needs network + writes `data/raw/` (now gone). |
| `fleurs.py` | Old per-source FLEURS exporter. Superseded by `combined.py` (which has its own FLEURS parser). Reads `data/raw/` (gone). |
| `commonvoice.py` | Old per-source Common Voice exporter. Superseded by `combined.py`. Reads `data/raw/` (gone). |
| `repair_labels.py` | One-time label repair over the artifact sqlite with the project normalizer. Already applied. |
| `resample_16k.py` | One-time backfill that resampled the existing artifact audio to 16 kHz mono. Its logic now lives in `combined.py`'s `copy_audio()`, so future builds are 16 kHz by construction. |
| `build_persian_augmentation.py` | Built the Persian→Tajik transliteration augmentation (v1 dataset). **Dead end** — the augmentation was a wash on the real Tajik test (see `docs/persian-augmentation-experiment-20260530.md`). Code valid, approach abandoned. |
| `parstranslit/` | Vendored ParsTranslit FA→TG transliterator (MIT, char-level CTranslate2) used by `build_persian_augmentation.py`. `ct2_fatg/` model + `tajik_vocab.txt` are gitignored; re-export from github.com/merchantrayyan/ParsTranslit. |

The live pipeline is: `tajik-build-combined` → `tajik-export-parquet` (+ `tajik-export-manifest`) → `tajik-audit-tokenizer` → `tajik-train`. Shared normalization stays in `../text_normalization.py`.
