# 08: Adjacent Repo Scan Archive

This file is now an archive note.
Active findings from this scan were folded into:

- `06_REALTIME_RL_TRACK.md` for research decisions and evidence.
- `07_REALTIME_PRONUNCIATION_BLUEPRINT.md` for implementation steps.

Last consolidated: 2026-03-01

## Original Scan Scope

Reviewed during this scan:

- `references/voxtral-finetune`
- `references/Finetune-Voxtral-ASR`
- `https://github.com/kyutai-labs`
- `https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B`
- `https://github.com/yazinsai/offline-tarteel`

## Consolidated Verdict

- Kyutai-related repos were the strongest streaming systems reference.
- Voxtral fine-tune repos were useful as training scaffolds, not complete
  production pipelines.
- Liquid references were useful for architecture comparison, but less
  clear for end-to-end fine-tuning workflows.
- `offline-tarteel` was valuable mainly as an experiment process model.

## Papers Pulled During The Scan

Added to `docs/papers/streaming_realtime/`:

- `2509.08753_delayed_streams_modeling.pdf`
- `2511.23404_lfm2_technical_report.pdf`
- `2602.11072_hibiki_zero.pdf`
- `2502.03382_hibiki_streaming_translation.pdf`
- `2410.00037_moshi.pdf`

## Notes

If this archive becomes noisy again, move any new actionable detail into
`06` or `07` and keep this file as a dated pointer only.
