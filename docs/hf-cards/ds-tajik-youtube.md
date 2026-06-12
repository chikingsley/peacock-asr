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
- 1M<n<10M
---

# tajik-asr-youtube

Every Tajik-language clip from this project's YouTube scrape — 41 channels of news, talk
shows, podcasts, audiobooks, and learning content — with machine transcripts and the
verification scores left **in** as columns instead of applied as a filter. Pick your own
quality threshold; the training corpus this project actually ships
([tajik-asr-corpus-v3](https://huggingface.co/datasets/Peacockery/tajik-asr-corpus-v3))
is the gated subset.

## Layout

Parquet shards under `data/` with an `audio` struct column (16 kHz mono FLAC bytes) plus
`text`, `channel`, `video_id`, `clip_id`, `duration`, `scribe_wer`, `scribe_cer`, and
`citation`. `scribe_wer`/`scribe_cer` measure agreement between two independent ElevenLabs
Scribe passes on the same clip — low values mean the transcript is corroborated, high values
mean the passes disagreed (hard audio, music, crosstalk). Null scores mean the clip was
never double-passed.

## How it was made

Channel audio was segmented (VAD for conversational sources, chunk+align for clean reads),
transcribed with an ElevenLabs Scribe ensemble, and language-gated (Tajik Cyrillic vs the
Russian and Persian that shares these channels). Non-speech descriptor clips are dropped.
No WER gate is applied — that's the column.

Labels are machine-generated and unverified by native speakers.
