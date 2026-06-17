# Common Voice Scripted Speech 25.0

Temporary working area for collecting Mozilla Data Collective Common Voice Scripted Speech 25.0
archives and staging them for `Peacockery/common-voice-scripted-speech-25-0` on Hugging Face.

The first pass preserves the upstream archives unchanged and records enough metadata to combine
them later into trainable ASR rows while still allowing users to recover per-language subsets.

## Layout

```text
manifests/datasets.jsonl                         # one Mozilla Data Collective dataset per line
scripts/download_mdc_archives.py                 # download by MDC dataset ID
scripts/inspect_archives.py                      # inspect tar members and TSV headers
scripts/stage_hf_release.py                      # stage raw archives, README, and summary JSON
../../data/common-voice-scripted-speech-25-0/    # ignored raw downloads/reports
../../data/common-voice-scripted-speech-25-0/hf-staging/
```

## Commands

Set the Mozilla Data Collective API key in the environment, or in an ignored `.env` file
at the Peacock repo root or this project directory. The scripts accept either name. Do not
paste API keys into command text, tmux logs, or committed files.

```bash
MDC_API_KEY=...
# or:
MOZILLA_DATA_COLLECTIVE_API_KEY=...
```

Download all manifest entries:

```bash
uv run --no-project projects/common-voice-scripted-speech-25-0/scripts/download_mdc_archives.py
```

Inspect archive structure without extracting the full dataset:

```bash
uv run --no-project projects/common-voice-scripted-speech-25-0/scripts/inspect_archives.py
```

Stage the Hugging Face upload folder:

```bash
uv run --no-project projects/common-voice-scripted-speech-25-0/scripts/stage_hf_release.py
```

Upload the staged folder:

```bash
hf upload-large-folder Peacockery/common-voice-scripted-speech-25-0 \
  data/common-voice-scripted-speech-25-0/hf-staging \
  --type dataset --num-workers 2
```

## Current Policy

- License metadata is recorded as `CC0-1.0` per the Mozilla Data Collective listing you provided.
- Raw MDC archives are preserved unchanged.
- Per-source fields are kept in the manifest so a later conversion can emit columns such as
  `source_dataset`, `source_dataset_id`, `source_archive`, `language`, `locale`,
  `original_split`, `audio`, and `sentence`.
