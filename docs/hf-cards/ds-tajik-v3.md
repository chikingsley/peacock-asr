---
license: cc-by-4.0
language:
  - tg
task_categories:
  - automatic-speech-recognition
tags:
  - tajik
  - machine-labeled
size_categories:
  - 100K<n<1M
---

# tajik-asr-corpus-v3

1,071 hours of Tajik ASR training data: 41 Tajik YouTube channels (~1,059 h, machine-labeled) plus FLEURS `tg_tj` (11.8 h, gold). This is the dataset behind [Peacockery/omni-ctc-300m-tajik](https://huggingface.co/Peacockery/omni-ctc-300m-tajik) (16.9% WER on FLEURS test, 37.6% on held-out conversational speech).

## Layout

Hive-partitioned parquet under `version=0/corpus=<source>/split=<split>/language=tgk_Cyrl/`. Each row holds `text` (the normalized label), `audio_bytes` (16 kHz mono FLAC as an int8 list), and `audio_size` (sample count). `language_distribution_0.tsv` lists hours per corpus.

## Test split

`split=test` holds two partitions: FLEURS test (gold labels, exported unfiltered), and a conversational held-out set — 1,625 clips from 157 whole videos carved out before training, so no clip from those videos appears in train. The conversational references are machine labels that passed the same agreement bar as training data, which makes that partition a relative benchmark; FLEURS is the absolute anchor.

## How the labels were made

FLEURS keeps its original transcripts. YouTube audio carries machine labels from an ElevenLabs Scribe ensemble, kept only where independent passes agree (WER ≤ 0.35 between passes), then language-gated (Tajik Cyrillic vs Russian/Persian) and normalized for CTC training. Machine labels are unverified by native speakers.

## Sources

FLEURS: CC-BY-4.0, google/fleurs. YouTube: public channels, labels produced by this project; per-clip channel and video identifiers are in the corpus partition names.
