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
- Added paired 100-row MLX/PyTorch streamed evaluation and comparison:
  MLX BF16 on Apple Silicon scored WER 1.80%, CER 0.51%, RTFx 1.61; PyTorch
  BF16 on the RTX 5070 scored WER 2.01%, CER 0.61%, RTFx 19.03 on the same
  rows.
- Profiled the MLX runtime and confirmed generation is the main bottleneck;
  an experimental `fast-greedy` path preserved quality but did not beat the
  default MLX-LM generation path.
- Organized the Mac working copy at
  `/Users/simonpeacocks/GitHub/moss-mlx-conversion`.
- Added gated real-weight pytest coverage for converted weight load, fixture
  transcription, and a one-row streamed eval on Apple Silicon.
- Added a local `MossTranscribeBackend` / `STTOutput` API plus serial serving
  adapter as the bridge toward an `mlx-audio` backend shape.
- Added quantized artifact loading and `moss-quantize`, using the MLX-LM
  quantization/config pattern with scoped predicates.
- Built and smoke-tested four private quantized candidates:
  text-decoder 8-bit, text-decoder 4-bit, all-module 8-bit, and all-module
  4-bit. The best current 20-row candidate is text-decoder 4-bit at 2.48 RTFx.
- Added `moss-package-manifest` and generated private local manifests for BF16
  and all quantized candidates. Public actions remain explicitly `none`.
- Ran FluidAudio Parakeet TDT v3 on full LibriSpeech `test-clean`: WER 2.63%,
  CER 1.03%, overall RTFx 39.61 on the same Mac.
- Started full MOSS benchmarks, then intentionally stopped them after BF16 and
  text-decoder 4-bit partials confirmed the MLX decoder-only path is far slower
  than FluidAudio/CoreML Parakeet. Recorded the partials and reframed MOSS as a
  teacher/reference candidate rather than a fast serving backend.
- Removed the temporary Mac project copy after copying useful artifacts back to
  the Linux project. Retained BF16, text-decoder 4-bit, and all-module 4-bit
  weights locally; kept 8-bit configs/reports/manifests without their weaker
  multi-GB weight files.
- Added a private CoreML/Mobius workbench and `moss-coreml-plan`, which records
  the MOSS component split, fixed prefill/cache shapes, Mobius prior-art notes,
  and parity gates before any actual `.mlpackage` export.
- Completed the first CoreML component probe on `home-mac`: extracted the MOSS
  token embedding tensor, exported `moss_token_embedding.mlpackage`, validated
  CoreML prediction against PyTorch with `max_abs_diff=0.0`, compiled it to
  `.mlmodelc`, and copied the generated artifacts back locally.
