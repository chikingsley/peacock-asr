# P012 Experiment Log

## 2026-03-27

- Created isolated P012 scaffold for original HConv reproduction.
- Wired the official `SSL_Interface` module into `s3prl` SUPERB task wrappers.
- Added CPU smoke tests using pseudo-audio and `fbank` upstream to verify the integration without benchmark downloads.
- Real benchmark runs still require external datasets:
  - LibriSpeech for `pr`
  - IEMOCAP for `er`
  - Fluent Speech Commands for `ic`
  - VoxCeleb1 for `asv`
