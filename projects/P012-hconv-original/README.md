# P012: Original HConv Reproduction

This project isolates the original HConv benchmark setup from the MuFFIN / SpeechOcean work.

The goal here is narrow:

- use the official `SSL_Interface` HConv module
- keep the stock `s3prl` SUPERB task code
- replace only the default weighted-sum featurizer with the interface module

This is the closest local path to reproducing the original HConv paper's benchmark-style setup without mixing in pronunciation-assessment code.

## What This Runs

P012 currently wraps these `s3prl` SUPERB tasks:

- `pr`: phoneme recognition on LibriSpeech
- `er`: emotion recognition on IEMOCAP
- `ic`: intent classification on Fluent Speech Commands
- `asv`: speaker verification on VoxCeleb1

The paper also reports ML-SUPERB results. Those are not wired yet because the current local benchmark base we cloned is the public `s3prl` SUPERB stack.

## Third-Party Sources

- `third_party/SSL_Interface`
- `third_party/s3prl`

P012 imports those local checkouts directly and does not modify them.

## Usage

Print the default config for a task:

```bash
uv run --project projects/P012-hconv-original p012 show-config --task pr
```

Run a SUPERB task with the official HConv interface:

```bash
uv run --project projects/P012-hconv-original p012 run-superb \
  --task pr \
  --dataset-root /path/to/LibriSpeech \
  --target-dir /tmp/p012-pr-hubert \
  --upstream hubert
```

Run the same task with the official weighted-sum interface for comparison:

```bash
uv run --project projects/P012-hconv-original p012 run-superb \
  --task pr \
  --dataset-root /path/to/LibriSpeech \
  --target-dir /tmp/p012-pr-hubert-ws \
  --upstream hubert \
  --interface weighted_sum
```

Preview the resolved config without launching training:

```bash
uv run --project projects/P012-hconv-original p012 run-superb \
  --task pr \
  --dataset-root /path/to/LibriSpeech \
  --target-dir /tmp/p012-pr-hubert \
  --upstream hubert \
  --dry-run
```

## Dataset Roots

- `pr`: LibriSpeech root
- `er`: IEMOCAP root
- `ic`: Fluent Speech Commands root
- `asv`: VoxCeleb1 root

## Notes

- HConv output width is inferred from the official module's own layer-collapse arithmetic when `--output-dim` is omitted.
- This project validates the HConv integration path with local smoke tests, but it does not bundle the benchmark datasets.
