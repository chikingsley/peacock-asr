# Common Voice Scripted Speech → Hugging Face

`cv26` mirrors Mozilla Data Collective (MDC) Common Voice Scripted Speech 26.0 archives into the public dataset [`Peacockery/common-voice-scripted-speech-26`](https://huggingface.co/datasets/Peacockery/common-voice-scripted-speech-26) as appendable parquet shards.

It is one space-bounded pipeline: **download an archive → convert it to parquet → upload the shards → verify them on the Hub → delete the local archive *and* the local shards.** The shards live on the Hub, so nothing data-bearing accumulates locally; adding a language adds shard files to the Hub, nothing is rewritten.

## Setup

```bash
uv sync
```

Secrets come from the environment or an ignored `.env` (repo root or this dir). Never paste keys into commands, tmux logs, or committed files.

- `MDC_API_KEY` (or `MOZILLA_DATA_COLLECTIVE_API_KEY`) — to download from MDC
- `HF_TOKEN` — to upload to the Hub

## Commands

```bash
uv run cv26 queue snippets.txt        # add languages to the manifest from pasted MDC curl snippets
uv run cv26 download --watch          # download manifest archives (resumable, rate-limit aware)
uv run cv26 process --upload --delete-after-upload --watch   # convert → upload → verify → delete
uv run cv26 card --upload             # regenerate + push the dataset card (README/LICENSE/manifest)
```

`download` and `process` run concurrently under `--watch`: the downloader lands archives while the processor drains them, keeping local disk bounded. `process` without `--upload` just builds parquet locally; `--delete-after-upload` only removes an archive after its shards verify on the Hub.

State is tracked append-only, so every command is safe to restart:

- `data/reports/downloads.jsonl` — one line per downloaded archive
- `data/hf-parquet/processing-state.jsonl` — each archive moves `converted` → `uploaded` → `complete`

## Layout

```text
manifests/datasets.jsonl   # one MDC dataset per line (dataset_id, locale, language, filename, license, …)
src/cv26/                  # download · convert · upload · pipeline · queue · card · config · manifest · cli
data -> /mnt/tiny-2t/peacock-asr/common-voice-scripted-speech-26-0
                           # gitignored: raw/archives/, hf-parquet/, reports/
```

## Output schema

MDC archives use the standard Common Voice layout (`train/dev/test/validated/invalidated/other.tsv`, `clip_durations.tsv`, `clips/*.mp3`); the ASR TSVs share one header. Splits may be empty — the upstream split name is preserved in `upstream_split`, and only non-empty splits produce a shard at `data/<upstream_split>/<collection>__<locale>__<dataset_id>.parquet`.

Each row embeds audio bytes plus full upstream provenance:

- `audio` (`bytes`, `path`), `source_audio_path`, `duration_ms`
- `sentence`, `locale`, `language`, `upstream_split`
- `source_dataset_id`, `source_archive`, `collection`, `license`, `license_url`
- upstream Common Voice columns: `client_id`, `sentence_id`, `sentence_domain`, `up_votes`, `down_votes`, `age`, `gender`, `accents`, `variant`, `segment`

## License

MDC records these archives as CC0-1.0. The Hub repo is private.
