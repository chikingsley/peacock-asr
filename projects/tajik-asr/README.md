# Tajik ASR Data Workspace

Finished combined dataset:

```text
src/tajik_omnilingual_asr/dataset_prep/artifacts/tajik_asr_combined_v0/
  train/data.tsv
  train/audio/
  dev/data.tsv
  dev/audio/
  test/data.tsv
  test/audio/
  tajik_asr_combined.sqlite
  dataset_summary.json
```

The root `data/` tree has been removed after assembly. The combined SQLite
keeps source provenance for every retained row:

- `source`
- `source_split`
- `source_id`
- `raw_transcription`
- `transcription`
- `normalized_text`
- `source_metadata_json`

## Current Counts

| Split | Rows |
| --- | ---: |
| train | 1,884 |
| dev | 263 |
| test | 440 |

| Source retained after dedupe | Rows |
| --- | ---: |
| `fleurs_tg_tj` | 1,815 |
| `common_voice_25_tg` | 572 |
| `muhtasham_tajik_asr_augmented_test` | 200 |

Build details live in `dataset_summary.json`.

## Scribe v2 Baseline

Scribe v2 results are stored in the combined SQLite table `scribe_curation`.
The current run id is `scribe-20260529T140514Z-444a2c81`.

| Scope | Rows | WER | CER |
| --- | ---: | ---: | ---: |
| All retained rows | 2,587 | 16.74% | 8.19% |
| `common_voice_25_tg` | 572 | 26.50% | 14.08% |
| `fleurs_tg_tj` | 1,815 | 8.95% | 3.03% |
| `muhtasham_tajik_asr_augmented_test` | 200 | 59.56% | 38.13% |

| Source | Split | Rows | WER | CER |
| --- | --- | ---: | ---: | ---: |
| `common_voice_25_tg` | dev | 123 | 27.70% | 14.87% |
| `common_voice_25_tg` | test | 121 | 32.01% | 15.38% |
| `common_voice_25_tg` | train | 328 | 24.02% | 13.31% |
| `fleurs_tg_tj` | dev | 140 | 8.89% | 3.04% |
| `fleurs_tg_tj` | test | 319 | 13.28% | 5.10% |
| `fleurs_tg_tj` | train | 1,356 | 7.93% | 2.54% |
| `muhtasham_tajik_asr_augmented_test` | train | 200 | 59.56% | 38.13% |

## v0 Fine-tune vs Scribe v2 (dev split)

omniASR CTC 300M fine-tuned on v0 (~5.8 h audio), best dev checkpoint ≈ step 1800.
Head-to-head on the **same 263 dev rows**, jiwer corpus-level, whitespace-normalized
(so both numbers are computed the same way):

| Model | Dev WER | Dev CER |
| --- | ---: | ---: |
| Scribe v2 | **13.1%** | 5.5% |
| Omni 300M fine-tune (v0, ~step 1800) | 17.1% | **4.3%** |

They split the win: Scribe v2 has the lower **WER**, the fine-tune has the lower
**CER**. Lower CER + higher WER means the model gets the *characters* right but the
*word boundaries* wrong (spacing/segmentation) — the same failure mode as Persian's
ZWNJ. For a 5.8 h v0, matching a strong commercial ASR within a few WER points and
beating it on CER says the lever is normalization + more data, not the model.

Note: the **16.74%** in the table above is a per-source *macro* average (inflated by
muhtasham's likely mis-scripting). Corpus-level by split, same method, Scribe v2 is:
train **11.7%** · dev **13.1%** · test **14.7%**.

> Fine-tune test-split WER/CER not yet measured (needs GPU inference on the
> step-1800 checkpoint; GPU is currently on the Persian scribe-v4 run).

## Where everything lives

The dataset is a single **versioned artifact** under `dataset_prep/artifacts/`.
Everything for one version sits in one folder; bump the version by making a new
`..._v1`, `_v2` dir.

```text
src/tajik_omnilingual_asr/
  dataset_prep/
    artifacts/tajik_asr_combined_v0/      # the versioned dataset
      train|dev|test/audio/ + data.tsv    #   16 kHz source audio
      tajik_asr_combined.sqlite           #   THE sql file (tables below)
      omni_manifest/                      #   TSV/.wrd, read by tajik-audit-tokenizer
      omni_parquet/version=0/             #   what training actually reads
    combined.py, omni_parquet.py, omni_manifest.py, text_normalization.py
    curation/scribe.py                    # Scribe verification
    archive/                              # one-time / superseded scripts (no CLI)
  models/                                 # weights + tokenizer (gitignored)
  fairseq2_assets.py                      # model/tokenizer/dataset cards (in-process)
  training/{configs,train.py,tokenizer_audit.py}
```

### SQLite tables (`tajik_asr_combined.sqlite`)

- `utterances` — corpus rows + provenance (source, splits, raw/normalized text)
- `scribe_runs` — one row per Scribe run (provider, model, status, counts)
- `scribe_transcripts` — per-row Scribe prediction + WER/CER + raw response
- `scribe_curation` — tidy per-row WER/CER view (drives the baseline table above)

### Day-to-day commands

`tajik-build-combined` · `tajik-export-parquet` · `tajik-export-manifest` ·
`tajik-curate-scribe` · `tajik-youtube-ingest` · `tajik-audit-tokenizer` ·
`tajik-train`

## YouTube pilot: Learning Tajik with Achilovs

Channel listing:

```bash
uv run tajik-youtube-ingest list-channel --limit 20
```

One-video ingest:

```bash
uv run tajik-youtube-ingest download-one 1ckpPxcC30o --scribe
```

Export the full Scribe transcript and word timeline for agent review:

```bash
uv run tajik-youtube-ingest export-scribe-review 1ckpPxcC30o
```

Cut clips from an agent-authored plan:

```bash
uv run tajik-youtube-ingest cut-plan 1ckpPxcC30o \
  --plan src/tajik_omnilingual_asr/dataset_prep/artifacts/youtube_learning_tajik_v0/review/1ckpPxcC30o/accepted_tajik_cut_plan.jsonl
```

Export a denormalized clip manifest with source metadata:

```bash
uv run tajik-youtube-ingest export-dataset-manifest
```

The pilot artifact lives under:

```text
src/tajik_omnilingual_asr/dataset_prep/artifacts/youtube_learning_tajik_v0/
  youtube_learning_tajik.sqlite
  videos/<video-id>/<video-id>.flac
  videos/<video-id>/<video-id>.info.json
  review/<video-id>/scribe_transcript.txt
  review/<video-id>/scribe_timeline.jsonl
  review/<video-id>/agent_cut_prompt.md
  review/<video-id>/accepted_tajik_cut_plan.jsonl
  review/<video-id>/accepted_scribe_check.jsonl
  cuts/<video-id>/<scribe-run-id>/*.flac
  cuts/<video-id>/<scribe-run-id>/cut_manifest.jsonl
  dataset_manifest.jsonl
```

The command downloads 16 kHz mono FLAC, stores yt-dlp metadata, stores manual
creator captions when YouTube exposes them, stores selected automatic captions
by default (`en-orig,en`), and writes Scribe v2 transcript plus word timings
into SQLite. The cut plan is accepted-only: Tajik-only or overwhelmingly Tajik
phrase repetition. English setup, mixed teaching talk, and uncertain cuts stay
out. `cut-plan` materializes those approved start/end ranges, and
`export-dataset-manifest` joins clip rows with title, channel, upload date,
source URL, original audio path, info JSON path, Scribe run metadata, and the
accepted Scribe recheck text when present.

## How to continue (iterating to v1, v2, …)

`data/raw` is gone, so **new versions are derived from v0's sqlite, not rebuilt
from raw** (`combined.py` is kept as the original recipe but needs raw to run).
The Scribe WER/CER in `scribe_curation` is the signal for what to cut or fix.
For example, `muhtasham` at ~60% WER is the first candidate to drop or inspect.

To produce **v1**:

1. Inspect `scribe_curation` (high WER = suspect labels/audio).
2. Copy `tajik_asr_combined_v0/` → `tajik_asr_combined_v1/`; in v1's sqlite
   drop/fix the flagged rows and remove their audio files.
3. `tajik-export-parquet --dataset-dir <v1> --overwrite` → writes
   `v1/omni_parquet/version=0/`.
4. Point `fairseq2_assets.py` `_PARQUET` and the config's
   `dataset_summary_path` at v1.
5. `tajik-audit-tokenizer` → `tajik-train`.

> **Gap / next tool:** there is no script yet that derives a filtered v1
> sqlite+audio from v0 (step 2 is manual). Worth writing a
> `tajik-derive-version` once we settle filtering rules.
