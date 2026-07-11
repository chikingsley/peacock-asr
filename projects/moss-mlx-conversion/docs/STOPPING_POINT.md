# MOSS FluidAudio Stopping Point

This is the clean pause point for the private MOSS/CoreML/FluidAudio work.

## Bottom Line

The private FluidAudio-shaped path works for the validated short-window bundle, but it is not production FluidAudio-level.

Working:

- Private FluidAudio scaffold patch: `patches/fluid-audio-moss-private-scaffold.patch`
- Local active CoreML bundle: `projects/moss-mlx-conversion/bundles/moss-fluid-audio-coreml-active`
- Local CoreML build tree used to rebuild that bundle: `projects/moss-mlx-conversion/coreml/build`
- Bundle smoke command shape: `fluidaudiocli moss-benchmark --model-dir <bundle> --cache-preset short-512 ...`
- Best bundle smoke result: WER/CER `0.0` / `0.0`, 14.23s audio, 18.43s processing, 0.77 RTFx on LibriSpeech row `6930-75918-0001`.
- Best real slice result: first 20 LibriSpeech clean-test rows through the private FluidAudio scaffold with the 512-cache bucket: WER `0.0158`, CER `0.00418`, 164.49s audio, 237.57s processing, 0.69 RTFx.

Not production-ready:

- The runtime is still single-window only, capped by the 30-second static audio package.
- Decode is autoregressive and moves explicit KV-cache tensors through CoreML every generated token.
- It is far slower than FluidAudio Parakeet-class ASR.
- The matched 768-token prefill package Torch-validates but crashes in the private FluidAudio `cpu-gpu` runtime before row output.
- Automatic cache-bucket selection was tried after the stable bundle work, but the first `--cache-preset auto` run crashed in CoreML/MPSGraph before row output. That experiment is not part of the committed stable baseline.

## Upload Readiness

Do not upload this as a finished public FluidAudio model.

Reasonable future upload target, after explicit approval:

- A private or clearly experimental Hugging Face model repo containing the active bundle files, with a model card that states:
  - single-window English MOSS ASR
  - manual `short-512` cache preset
  - 20-row validation only for the FluidAudio scaffold
  - known speed and long-audio limitations
  - no claim of production FluidAudio parity

## Disk State

The Mac has been cleaned. The removed Mac-side paths were:

- `/Users/simonpeacocks/GitHub/moss-mlx-conversion`
- `/tmp/FluidAudio`
- `/tmp/FluidAudio-stable`
- the private MOSS scaffold paths inside `/Users/simonpeacocks/GitHub/FluidAudio`
- `/Users/simonpeacocks/GitHub/FluidAudio/.build`

After cleanup, `home-mac` free space under `/Users` was about 62GiB.

The local Linux project now retains the needed ignored artifacts:

- `coreml/build`: 15G
- `bundles/moss-fluid-audio-coreml-active`: 15G
- `artifacts`: 56G
- full project: 68G

The local bundle uses reflink-style copies where the filesystem supports them, but it should still be treated as a 15G bundle if copied or uploaded.

## Keep

- Tracked `projects/moss-mlx-conversion` history.
- `runtime/moss_bundle_manifest.json`
- `scripts/build_fluid_audio_bundle.sh`
- `patches/fluid-audio-moss-private-scaffold.patch`
- Ignored local active bundle: `bundles/moss-fluid-audio-coreml-active`
- Ignored local `coreml/build` while continuing CoreML development; it contains the compiled packages used to rebuild the active bundle, including the working 512-cache pair and experimental matched-768 prefill package.

## Cleanup Candidates

The Mac cleanup is complete. Do not remove local `coreml/build` or the local bundle unless you are comfortable rebuilding or recopying multi-GB CoreML packages later.

## Next Work

The next real implementation choices are:

1. Keep MOSS as a teacher/reference and avoid more production-runtime work.
1. Continue FluidAudio runtime work by debugging CoreML/MPSGraph failures for larger buckets and automatic bucket selection.
1. Add long-audio chunking and stitching around the existing 30-second window.
1. Profile or redesign explicit KV-cache movement; this is the main speed blocker.
