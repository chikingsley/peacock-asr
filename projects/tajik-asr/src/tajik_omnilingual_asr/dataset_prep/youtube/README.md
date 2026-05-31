# YouTube → Tajik ASR dataset pipeline

Turns Tajik YouTube audio into omni-parquet training data, all through one sqlite DB.
Entry point: `tajik-youtube-ingest` (→ `cli.py`).

## Pipeline (5 stages, each reads/writes the artifact sqlite)

```
download          audio (16 kHz mono FLAC) + captions + metadata  ->  youtube_videos / youtube_captions
transcribe-scribe Scribe (via superwhisper-api) on full videos    ->  youtube_scribe_runs  (word-level timing)
segment           NeMo VAD windows + Scribe-word alignment         ->  youtube_segments     (cut clips + aligned text)
transcribe-omni   our omni CTC model on each clip (CPU by default) ->  youtube_omni_transcripts
export            Scribe<->omni agreement gate                     ->  omni-parquet (corpus=youtube_tajik)
```

The Scribe word-timestamps drive automatic segment-text alignment (`segment.aligned_text`)
— there is **no manual cut-plan**. `export` keeps a segment only if Scribe and our model
agree (`--max-agreement-wer`, default 0.3); run `export` before `transcribe-omni` and it
falls back to Scribe-only labels (ungated).

## Runbook

```bash
# 1. find sources (see ../artifacts/channel_discovery/tajik_youtube_candidate_channels.md)
tajik-youtube-ingest list-channel --channel-url <url> --limit 30

# 2. download (URLs or a channel, with filters)
tajik-youtube-ingest download <url> ...
tajik-youtube-ingest download --channel-url <url> --max-duration-seconds 1800 --exclude-title-regex '(?i)shorts'

# 3. Scribe (full video; provides the word timing segment alignment needs)
tajik-youtube-ingest transcribe-scribe            # all videos; superwhisper-api handles the key

# 4. segment: NeMo VAD + align (CPU)
tajik-youtube-ingest segment

# 5. (optional) our model per segment, for the agreement gate — CPU is fine
tajik-youtube-ingest transcribe-omni --model-card omni_ctc_300m_v2_tajik_step_1800

# 6. export to omni-parquet (drop-in next to the base corpora under version=0)
tajik-youtube-ingest export --output-root <artifact>/omni_parquet/version=0 --max-agreement-wer 0.3
```

## Files

| File | Role |
|---|---|
| `cli.py` | one argparse router for the 5 stages |
| `db.py` | sqlite: `connect` (WAL), `ensure_schema` (5 tables), query helpers |
| `ingest.py` | download (yt-dlp) + caption parsing (json3/vtt/srt) + store |
| `transcribe.py` | `transcribe-scribe` (full video) and `transcribe-omni` (per segment) |
| `segment.py` | NeMo VAD + Scribe-word alignment → segments |
| `export.py` | agreement gate → omni-parquet |
| `process.py` | subprocess glue: yt-dlp, ffmpeg `cut_audio`, timestamps |
| `paths.py` | artifact dir + default channel/language constants |

Live external deps: `text_normalization.normalize_text` and `curation.scribe.compute_wer/cer`.

## Notes

- **Source selection matters.** "Learning Tajik"-style channels are mostly *English*
  instruction with Tajik snippets — weak for ASR. Prefer native Tajik content (news,
  interviews, broadcast); see the candidate-channels doc.
- The NeMo VAD model (`artifacts/models/nemo_vad_multilingual_frame_marblenet_*/...nemo`)
  must be present for `segment`.
- Agent-in-the-loop adjudication of *disagreements* (a `review-disagreements` step) is the
  planned extension on top of the automatic agreement gate.
