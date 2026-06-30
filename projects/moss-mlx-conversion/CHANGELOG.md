# MOSS MLX Conversion Changelog

Historical record of completed project work. Live commands, current results,
and next steps live in `docs/PROGRESS.md`; durable design context lives in
`docs/PLAN.md`.

## 2026-06-30

- Built a role-organized MOSS MLX conversion project with `reference/`,
  `conversion/`, `runtime/`, `model/`, and `docs/` package/doc areas.
- Pinned `OpenMOSS-Team/MOSS-Transcribe-preview-2B` at
  `c98175cb20e48bd9be4e95f6c85f2af18899f780`.
- Captured PyTorch BF16 reference transcript, tensors, and processor parity for
  the LibriSpeech smoke fixture.
- Converted all 838 BF16 source tensors into an MLX-layout safetensors artifact
  with no skipped source tensors.
- Verified the converted BF16 artifact on Apple Silicon through
  `moss-mlx-smoke`; the first 5 generated token IDs and transcript matched the
  PyTorch reference exactly.
- Added `moss-streaming-eval`, which streams Hugging Face Dataset Viewer rows
  and audio asset bytes in memory and scores WER/CER with `jiwer`.
- Ran the first 20 `openslr/librispeech_asr` clean-test rows on Apple Silicon:
  WER 1.58%, CER 0.42%, RTF 0.65.
- Organized the Mac working copy at
  `/Users/simonpeacocks/GitHub/moss-mlx-conversion`.
