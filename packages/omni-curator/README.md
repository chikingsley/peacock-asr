# omni-curator

Shared ASR dataset curation for Peacock language projects. The package owns one queue-driven
pipeline; per-language projects provide source registries, language/script settings, and policy.

```text
download -> enqueue -> segment -> labelq -> harvest -> merge -> verify -> export
```

The working queue is SQLite. `segment` claims whole recordings, decodes each source once, runs a
resident VAD engine, cuts 16 kHz FLAC clips through claim-token staging, and publishes clip rows
atomically. `labelq` drains those rows through Scribe. `harvest` preserves clip metadata in the
per-channel stores, so segmentation provenance survives through export.

## Multi-VAD contract

Three adapters implement the same boundary:

- input: one decoded, mono float array at 16 kHz;
- output: engine-native `(start_seconds, end_seconds)` speech intervals;
- no adapter performs Peacock padding, short-span filtering, gap merging, or hard splitting.

The shared versioned postprocessor owns those operations in this fixed order: sanitize and clamp,
pad, sort, merge, remove short spans, then split above the selected ASR model's hard duration cap.
Every emitted clip records the engine, exact model/runtime revision, threshold/backend,
postprocessor values, policy revision, and an effective profile hash.

Supported engines:

- `marblenet`: exact multilingual MarbleNet v2 `.nemo`; no fallback to the older similarly named
  checkpoint;
- `cobra`: Picovoice Cobra, requiring `PICOVOICE_API_KEY`;
- `silero`: Silero VAD, ONNX on CPU or JIT on CUDA.

Production still defaults to `marblenet` with `legacy-marblenet-v1` boundaries until a measured
per-project pilot selects a replacement. Engine choice is explicit; failures never silently route
to another adapter.

## Language-project CLI

Every project receives the same commands through `CuratorProject`. For Farsi, for example:

```bash
uv run --project projects/farsi-asr --locked farsi-curate enqueue
uv run --project projects/farsi-asr --locked farsi-curate segment \
  --vad-engine marblenet --vad-profile legacy-marblenet-v1 --max-duration 30
uv run --project projects/farsi-asr --locked farsi-curate labelq
uv run --project projects/farsi-asr --locked farsi-curate harvest
uv run --project projects/farsi-asr --locked farsi-curate merge
uv run --project projects/farsi-asr --locked farsi-curate verify
uv run --project projects/farsi-asr --locked farsi-curate export v1
```

Set `OMNI_CURATOR_VAD_MODEL` or pass `--vad-model` for the exact MarbleNet v2 checkpoint. Project
environment files are loaded before adapter preflight, including the Cobra activation key.

## Isolated VAD pilots

`vad-pilot` requires an exact JSONL selector. Each line has `id`, `path`, `tier`, and `channel`.
It never opens the production queue, refuses to write inside production `data/clips`, and emits a
self-contained `run.json`, `intervals.jsonl`, optional clips, and optional Scribe sample:

```bash
uv run --project projects/farsi-asr --locked farsi-curate vad-pilot \
  --manifest projects/farsi-asr/docs/vad-pilots/farsi-clean-noisy-32.jsonl \
  --output-dir /mnt/workerssd-2t/peacock-asr/pilots/farsi-vad-YYYY-MM-DD \
  --device cpu --write-clips --scribe-max-clips-per-engine 20
```

The default pilot runs Cobra, Silero, and MarbleNet over the same sources with the new
`conservative-v1` shared profile. The older FLEURS policy used engine-native semantics, so its
quality rows are evidence for candidates, not an exact prediction of this central policy.

## Dependencies

Core dependencies include NumPy, SoundFile, Torchaudio, and NeMo. Adapter extras are deliberately
small and do not import the benchmark project's incompatible Torch/NeMo environment:

```bash
uv sync --project packages/omni-curator --extra vad-cobra
uv sync --project packages/omni-curator --extra vad-silero
```

Cobra is activation-key gated. Silero is MIT licensed. MarbleNet v2 weights use the NVIDIA Open
Model License; preserve the model attribution and license when redistributing the checkpoint.

The canonical operating and storage contract is
[`docs/CURATION_FACTORY.md`](docs/CURATION_FACTORY.md). Root `TODO.md` contains active work only;
completed changes are recorded in root `CHANGELOG.md`.
