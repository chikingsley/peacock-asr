# omni-curator

In-house ASR dataset **curator** — turns raw audio into fine-tuning transcripts. Sibling of
`omni-finetune-core` (the training side); together they're our equivalent of NVIDIA's NeMo
Curator, built for the Omni-ASR toolchain.

```text
audio ─► segment ─► Scribe ensemble ─► compile-down ─► (stitch) ─► polish ─► transcript
```

## Two finalized paths

Both share the per-clip core (cut → Scribe ensemble → compile-down, run over a thread pool).
They differ in how the audio is cut and how the labels are reassembled:

| path | segmenter | reassembly | use for |
|------|-----------|------------|---------|
| `vad_path`    | `segment_vad` — cut at silences, no overlap | labels joined in order | dense, continuous speech |
| `chunks_path` | `segment_chunks` — fixed overlapping windows, 100% coverage | `stitch` reconciles the seams | sparse / drill-style audio (VAD drops short utterances) |

`polish` is a cheap, conservative final pass on both that repairs obvious machine glitches
(a split word, an unambiguous mishearing) while preserving natural speech — repetitions, filler,
code-switching. It can be turned off (`do_polish=False`).

The segmenter choice is empirical: on clean French (Pimsleur, with a human reference) `chunks_path`
scored ~2% WER while `vad_path` lost ~11% by dropping the short drilled syllables; on dense Tajik
TV banter the two were equivalent and `vad_path` is simpler. Measure on a labelled sample first.

## Use

```python
from pathlib import Path
from omni_curator import vad_path, chunks_path

# non-Latin target (default prompt forces script + transliterates)
t = vad_path(Path("show.flac"), out_dir=Path("out"),
             language="Tajik", script="Cyrillic script (tgk_Cyrl)", langs=("auto", "tgk"))

# Latin / bilingual target -> pass a custom compile-down instruction
t = chunks_path(Path("lesson.flac"), out_dir=Path("out"),
                language="French", script="standard French orthography",
                langs=("auto", "fr"), instruction=MY_FRENCH_INSTRUCTION)

print(t.text)        # the transcript
t.write_json(...)    # transcript + per-clip variants/labels
```

CLI: `omni-curator <audio> --path vad|chunks --out-dir <dir> --language <L> --script "<S>" --langs auto,<code>`

## Dependencies / setup

- ElevenLabs Scribe + the free SuperWhisper text endpoint via `superwhisper-api` (path dep). The
  ElevenLabs key resolves env → macOS cache → Mac-mirror; see `superwhisper_api.auth`.
- `nemo-toolkit[asr]` for `segment_vad`. The frame-VAD checkpoint is downloaded on first use, or
  point `OMNI_CURATOR_VAD_MODEL` at a local `.nemo`. `chunks_path` does not need NeMo.
- `ffmpeg` on PATH (clip cutting).

## The full pipeline (all implemented)

`create` (YouTube → VAD/chunks → Scribe ensemble → labels) and `ingest` (FLEURS, Common Voice via
MDC, any HF audio dataset via `load_hf_audio`) both feed `store` (the SQLite master pool). `process`
normalizes text per-language (`process/normalize`) and gates content-language (`process/language`).
`verify` Scribe-scores every clip. `export` materializes a `datasets/vN` omni-parquet ablation
(normalize → language gate → quality filter → tokenizer-coverage gate → write).
