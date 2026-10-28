# Tajik ASR

Tajik (`tgk_Cyrl`) ASR data preparation, Omni CTC training, and Parakeet TDT training. Language configuration lives here; reusable data and model mechanics live in `omni-curator`, `omni-finetune-core`, and `parakeet-finetune-core`.

## Released model

[Peacockery/tajik-parakeet-tdt-110m](https://huggingface.co/Peacockery/tajik-parakeet-tdt-110m) carries CC BY 4.0, NVIDIA/Suno attribution, a model card, and a notice. Its original weights remain the published baseline. The October 28, 2026 check reproduced 19.03% normalized WER on all 600 FLEURS test clips; NeMo 2.7.3 and 3.0.0 produced identical prediction records.

The historical `train_big.jsonl` preserves the released model's provenance. Its SQL exporter included clips from 143 reserved evaluation videos and 17,814 jingle-descriptor labels. Current training defaults use the curated v3 corpus and explicit identity checks. The conversational benchmark is relative to machine-generated references; FLEURS supplies the independent reference benchmark.

Publication and audit evidence lives in `docs/license-publication-2026-10-28.json`, `docs/license-and-training-audit-2026-10-28.json`, and `src/tajik_asr/parakeet/artifacts.json`.

## Data and provenance

The retained curator database contains 347,897 clips and 1,841.55 hours. The published [curated v3 corpus](https://huggingface.co/datasets/Peacockery/tajik-asr-corpus-v3) contains 183,146 clips and 1,070.81 hours across training, development, and test. It stores complete FLAC audio in Parquet. Its training partition contains 180,683 clips and 1,057.54 hours before the Parakeet duration cap and additional identity exclusions.

The broader [YouTube corpus](https://huggingface.co/datasets/Peacockery/tajik-asr-youtube) retains clip identities and Scribe agreement scores. Original recordings remain in `/mnt/massive-22t/peacock-asr-archive/tajik` on this host. The clean retrain consumes the curated corpus locally on the worker SSD. Source recordings serve future curation and segmentation experiments.

YouTube labels originated from Scribe hypotheses and LLM fusion. Scribe agreement measures consistency between machine transcripts. FLEURS retains its reference transcripts. The preparation report records those origins separately and retains `unknown` for individual YouTube recording licenses.

## Prepare clean Parakeet training

Run commands from the repository root:

```bash
uv sync --project projects/tajik-asr --locked
uv run --project projects/tajik-asr --locked tajik-parakeet-prepare --download --all-splits
```

`--source-database` selects a retained curator snapshot. `--dataset-root` and `--output-root` select storage locations. On this host, `data/datasets/v3` and `data/parakeet/v3-safe` resolve to the worker SSD. The verified curator snapshot is `/mnt/workerssd-2t/peacock-asr/tajik-asr/provenance/curator-2026-10-28.sqlite`.

Preparation verifies every Parquet shard against the pinned Hub revision, extracts deterministic FLAC files, reconstructs candidate source identities by corpus, normalized text, and sample count, and records ambiguous matches. Training excludes every candidate matching a reserved video or evaluation identity. Its manifest includes audio hashes, candidate source IDs, label origins, dataset revision, and recording-license status. Development and test preserve the published v3 duration range through 40 seconds; training uses 0.5–30 seconds.

`source-audit.json` and `materialization-<split>.json` record the resulting counts and exclusions. Training entry points check the frozen video list, evaluation identities, audio availability, and manifest completeness before restoring the model.

The October 28 preparation retained 173,240 training clips and 987.54 hours after excluding 7,200 overlong or short clips and 243 evaluation-identity conflicts. All 62 source Parquet shards passed SHA-256 verification. Header checks covered 176,104 distinct training and evaluation audio files: 16 kHz mono throughout, with source-duration rounding below 10 milliseconds. Future curator Parquet exports also carry sample and video IDs directly, and the shared materializer preserves those fields.

Local SSD staging keeps network and rotating-disk latency outside the training loop. The Parquet archive occupies 75.6 GB; extracted FLAC files provide direct access for Lhotse's shuffled batches. Streaming with a local cache is a viable option for a larger corpus. This corpus fits locally with over 500 GB of SSD space remaining.

## Train and evaluate

Train a fresh tokenizer from the prepared training manifest:

```bash
uv run --project projects/tajik-asr --locked tajik-parakeet-train-tokenizer \
  --manifest projects/tajik-asr/data/parakeet/v3-safe/manifests/train.jsonl \
  --name v3-safe --vocab-size 1024
```

The tokenizer wrapper uses NVIDIA's `scripts/tokenizers/process_asr_text_tokenizer.py` under the configured NeMo recipe cache. This host has the script from the NeMo Speech `v3.0.0` tag. The base checkpoint is `nvidia/parakeet-tdt_ctc-110m`; its SHA-256 is `1e6b5c3bcaa390b479bd8112512ede515d2ef5b61495568f2c266d6d5e680c3a`.

Start with a short pilot, then run the clean baseline:

```bash
uv run --project projects/tajik-asr --locked tajik-parakeet-train-tdt \
  --name tajik-parakeet-tdt-110m-clean-20261028 \
  --max-steps 100000 --warmup 2000 --lr 0.0003 \
  --batch-dur 120 --max-dur 30 --fused-batch-size 2
```

The curated labels use lowercase, punctuation-free text. This is the new model's output-style contract. TDT uses validation loss for checkpoint selection and standalone fp32 WER for quality comparisons. Each invocation records its arguments, package versions, and input hashes in `run_invocations.jsonl`. Training saves last-step and best-validation-loss model artifacts separately.

The initial October 28 service completed its 50-step pilot and saved a clean checkpoint at step 14,000. At Simon's request, that service and its waiting refinement worker were stopped, and `peacock-tajik-upgraded-20261028.service` immediately continued from the preserved checkpoint with the upgrades below. The checkpoint includes model weights, optimizer state, learning-rate scheduler, and global step. Initial scripts and logs live in `runs/parakeet/clean-retrain-20261028`; continuation evidence lives in `runs/parakeet/upgrades-20261028`. `ExecStopPost` restarts `vllm-asr` and checks its health after success, failure, or a service stop. The host launcher explicitly provides `CUDA_PATH=/opt/cuda` for Numba's TDT loss compiler.

NeMo is upgraded to 3.0.0. Datasets remains on 4.8.5 because Datasets 5 imports Hub 1.x APIs while the current Transformers stack requires Hub below 1. The dependency audit records the remaining version constraints in `docs/license-and-training-audit-2026-10-28.json`.

Compare the best checkpoint with the released baseline on the same FLEURS and conversational manifests. Report those benchmarks separately. Promotion follows measured quality and an explicit model-card update.

## Applied training upgrades

The October 28 runtime comparison tested identical 200-step short-clip workloads. Unfused loss with BF16 mixed precision measured 0.14125 seconds per step, versus 0.16875 for the fused path: about 19.5% higher audio throughput. Peak allocated CUDA memory was 2.52 GB and 2.47 GB respectively. The baseline process was suspended with its GPU memory resident during this comparison. Both full-BF16 trials reached the hybrid CTC loss's FP32 requirement, so the selected runtime uses mixed precision. The continuation uses full-duration audio up to 30 seconds and a 120-second batch budget. A CUDA memory failure switches to fused loss while preserving the upgraded data mixture and saved optimizer state.

The active continuation applies 12% FLEURS clip sampling immediately. The prepared sources contain 2,250 reference-labeled FLEURS clips and 170,990 Scribe-labeled clips. A live NeMo sampler check drew 12.2% FLEURS across 5,000 examples. The run preserves the original 100,000-step optimizer schedule, restores step 14,000, and saves checkpoints every 2,000 steps. The earlier waiting refinement service has been retired from this run.

Each source uses shuffled 200-clip manifest shards. The workflow compares the continuation's final, best-validation-loss, and averaged artifacts on the development set in fp32. It selects the lowest development WER, then evaluates that fixed choice on the full 600-clip FLEURS benchmark and the 1,625-clip conversational benchmark. ASR restoration runs on both success and failure. Candidate results remain clearly associated with their source models and manifests. This continuation measures the combined changes; the earlier controlled refinement commands remain available for separate experiments.

The repeatable command is `uv run --project projects/tajik-asr --locked tajik-parakeet-upgrade`. Its `continue --checkpoint <saved.ckpt> --run-name <new-run>` action immediately resumes training with the prepared upgrades and manages ASR restoration. The other actions are `prepare`, `verify-sampling`, `benchmark`, `refine`, and `wait-and-refine`. This workflow targets the October 28 baseline and its tokenizer. Run details live in `runs/parakeet/upgrades-20261028`, and the tracked measurements and official references live in `docs/training-upgrades-2026-10-28.json`.

## Collection pipeline

```text
curate download → enqueue → segment → labelq → harvest → merge → ingest → verify → export
```

The project exposes `tajik-curate`, `tajik-omni-train`, `tajik-omni-eval`, and the `tajik-parakeet-*` commands. FLEURS, Common Voice, and external Hugging Face ingestion are configured in `curate.py`. Current LLM fusion accepts one explicit transcript element and limits corrections to wording supported by the supplied hypotheses and script conversion.
