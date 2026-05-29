# Parakeet CTC 109M Scribe Exact-Match 20260525 bd700

## Dataset

- Dataset: `data/training/scribe-verified/exact-match-20260525`
- Selection: rows from the classified Scribe training manifests where `difference_category == "exact_match"`
- Source manifests: `data/training/scribe-verified/full-20260523-classified-keep/manifests/*.filtered-maxchars400-cps60.jsonl`
- Summary: `data/training/scribe-verified/exact-match-20260525/summary.md`

| split | rows | hours |
|---|---:|---:|
| train | 210,217 | 224.34 |
| dev | 9,429 | 11.41 |
| test | 7,828 | 10.64 |
| total | 227,474 | 246.40 |

| source | rows | pct rows | hours | train | dev | test |
|---|---:|---:|---:|---:|---:|---:|
| common_voice_25_0 | 177,121 | 77.86% | 177.23 | 165,184 | 6,815 | 5,122 |
| mana_tts | 29,867 | 13.13% | 32.85 | 26,895 | 1,504 | 1,468 |
| neyshekar | 11,902 | 5.23% | 16.51 | 10,722 | 567 | 613 |
| asr_farsi_youtube | 3,772 | 1.66% | 8.22 | 3,030 | 384 | 358 |
| thomcles_persian_farsi_speech | 3,091 | 1.36% | 6.33 | 3,056 | 35 | 0 |
| fleurs | 1,304 | 0.57% | 4.53 | 951 | 110 | 243 |
| worldspeech | 417 | 0.18% | 0.72 | 379 | 14 | 24 |

Verification on 2026-05-25:

- `wc -l`: `227474 total`
- Category check: all rows have `difference_category == "exact_match"`
- Audio check: `missing_audio=0` for train/dev/test

## 109M Launch

- Run name: `parakeet-ctc-109m-scribe-exact-match-20260525-bd700`
- Tmux session: `parakeet-exact-109m`
- Launch log: `parakeet/runs/_launch_logs/parakeet-ctc-109m-scribe-exact-match-20260525-bd700.log`
- Command file: `parakeet/runs/_launch_logs/parakeet-ctc-109m-scribe-exact-match-20260525-bd700.command.txt`
- Experiment root: `parakeet/runs/parakeet-ctc-109m-scribe-exact-match-20260525-bd700/2026-05-25_07-40-48`

Main parameters:

```text
batch_duration=700
quadratic_duration=15
num_buckets=30
min_steps=11540
max_steps=34620
val_check_interval=1154
learning_rate=1e-4
warmup_steps=1500
min_lr=5e-6
early_stopping_patience=5
disable_cudnn=true
disable_progress_bar=true
```

The step schedule was scaled from the previous full classified run by train hours: `224.34h / 844.35h`.

## Live Status

Started at `2026-05-25 07:40:44` in tmux. TensorBoard scalars confirmed training progress at global step `299` with GPU utilization active.

First validation:

- Step: `1154`
- Wall time: `2026-05-25 07:47:58`
- `val_wer`: `0.9791349769`
- `val_loss`: `113.5859146`
- Checkpoint: `parakeet/runs/parakeet-ctc-109m-scribe-exact-match-20260525-bd700/2026-05-25_07-40-48/checkpoints/parakeet-ctc-109m-scribe-exact-match-20260525-bd700--val_wer=0.9791-epoch=0.ckpt`

Latest checked after first validation:

- Step: `1449`
- Wall time: `2026-05-25 07:49:42`
- `train_step_timing in s`: `0.2629699707`
- GPU: active around `90%`

Later live checks:

- `2026-05-25 07:52:39`: step `1949`, epoch scalar `1.0`, GPU active around `91%`
- `2026-05-25 07:56:33`: step `2599`, epoch scalar `1.0`, `training_batch_wer=0.7197149396`, GPU active at `100%`
- `2026-05-25 07:58:02`: step `2828`, epoch scalar `1.0`, second validation `val_wer=0.6223899722`, `val_loss=36.8857688904`
- `2026-05-25 08:08:01`: step `4502`, epoch scalar `2.0`, third validation `val_wer=0.4990411997`, `val_loss=26.9539833069`
- `2026-05-25 08:10:15`: step `4899`, epoch scalar `2.0`, GPU active around `85%`
- `2026-05-25 08:17:50`: step `6175`, epoch scalar `3.0`, fourth validation `val_wer=0.4229317605`, `val_loss=21.8846015930`
- `2026-05-25 08:21:08`: step `6749`, epoch scalar `4.0`, GPU active around `82%`
- `2026-05-25 08:27:35`: step `7849`, epoch scalar `4.0`, fifth validation `val_wer=0.3706702292`, `val_loss=18.4854793549`
- `2026-05-25 08:32:31`: step `8699`, epoch scalar `5.0`, GPU active around `66%`
- `2026-05-25 08:37:26`: step `9523`, epoch scalar `5.0`, sixth validation `val_wer=0.3339014947`, `val_loss=16.3816165924`
- `2026-05-25 08:37:40`: step `9549`, epoch scalar `5.0`, GPU active around `99%`

## Omni 300M Prep

Derived Omni input manifest:

```text
data/training/scribe-verified/exact-match-20260525/omni_manifest.jsonl
```

Omni config:

```text
configs/omni/persian-asr-scribe-exact-match-20260525-ctc-300m-v2.yaml
```

Exported Omni parquet dataset:

```text
data/training/omnilingual/persian_asr_scribe_exact_match_20260525/
```

Export summary:

- Summary: `data/training/omnilingual/persian_asr_scribe_exact_match_20260525/export_summary.json`
- Distribution: `data/training/omnilingual/persian_asr_scribe_exact_match_20260525/language_distribution_0.tsv`
- Train rows: `210,217`
- Dev rows: `9,429`
- Total training/dev rows: `219,646`
- Train hours: `224.34131333335637`
- Dev hours: `11.414422187499975`
- Skipped test rows: `7,828`

The Omni exporter uses train/dev splits for training and validation; exact-match test rows remain in the NeMo manifests and were not routed into Omni training.

## Post-109M Evaluation Commands

Run after the 109M process exits and the final/best `.nemo` is present:

```bash
cd /home/simon/github/peacock-asr/projects/persian-asr

.venv/bin/persian-benchmark-asr \
  --source manifest \
  --model nvidia \
  --manifest data/training/scribe-verified/exact-match-20260525/manifests/test.jsonl \
  --output-dir benchmarks/parakeet-ctc-109m-scribe-exact-match-20260525-heldout-test \
  --batch-size 8 \
  --nvidia-model-name parakeet/runs/parakeet-ctc-109m-scribe-exact-match-20260525-bd700/2026-05-25_07-40-48/checkpoints/parakeet-ctc-109m-scribe-exact-match-20260525-bd700.nemo \
  --decoder-type ctc

.venv/bin/persian-benchmark-suite \
  --suite-name canonical-tests-parakeet-scribe-exact-match-20260525 \
  --models parakeet-exact-match-20260525 \
  --local-nvidia-model-name parakeet/runs/parakeet-ctc-109m-scribe-exact-match-20260525-bd700/2026-05-25_07-40-48/checkpoints/parakeet-ctc-109m-scribe-exact-match-20260525-bd700.nemo \
  --local-nvidia-model-label parakeet-exact-match-20260525 \
  --local-nvidia-batch-size 8 \
  --local-nvidia-decoder-type ctc
```

Expected outputs:

- Held-out exact-match test: `benchmarks/parakeet-ctc-109m-scribe-exact-match-20260525-heldout-test/summary.md`
- Canonical suite: `benchmarks/suites/canonical-tests-parakeet-scribe-exact-match-20260525/summary.md`

## Post-109M Automation

Started on `2026-05-25 08:09:37 PDT`:

```text
tmux session: exact-match-post109m
script: scripts/run_exact_match_post109m_pipeline.sh
log: runs/pipeline/exact-match-post109m-20260525.log
```

The script waits for `parakeet-exact-109m` to exit, runs the held-out exact-match test evaluation,
runs the canonical suite, then launches the Omni 300M CTC run with:

```text
config: configs/omni/persian-asr-scribe-exact-match-20260525-ctc-300m-v2.yaml
output: runs/omni-ctc-300m-scribe-exact-match-20260525
```

Updated and restarted on `2026-05-25 08:11:01 PDT` so the wait loop handles NeMo's stale post-fit
process behavior. It only terminates the Parakeet session/process if the final `.nemo` exists and
the launch log contains the explicit ``Trainer.fit`` stopped marker.

## Omni 300M Exact-Match Benchmark Results

The 300M exact-match run finished at step `34000`; validation best was step `32000`.
The benchmark asset card points to the step `32000` checkpoint:

```text
omni_ctc_300m_v2_persian_scribe_exact_match_20260525_best
```

Canonical suite output:

```text
benchmarks/suites/canonical-tests-omni-ctc-300m-scribe-exact-match-20260525-best/
```

Full six-split results:

| Model | Dataset | WER | CER | Samples | Audio h | RTFx |
|---|---|---:|---:|---:|---:|---:|
| omni-scribe-exact-300m-best | common_voice_25 | 21.8685 | 5.3868 | 10702 | 14.6950 | 354.9840 |
| omni-scribe-exact-300m-best | fleurs | 9.8093 | 2.8246 | 852 | 3.6028 | 438.6264 |
| omni-scribe-exact-300m-best | mana_tts | 27.5860 | 4.6107 | 3987 | 5.3485 | 290.1766 |
| omni-scribe-exact-300m-best | neyshekar | 28.3151 | 4.5277 | 1331 | 2.0517 | 401.0638 |
| omni-scribe-exact-300m-best | worldspeech | 38.7759 | 19.8776 | 359 | 1.3314 | 105.4755 |
| omni-scribe-exact-300m-best | youtube | 36.3050 | 14.3965 | 13890 | 33.3270 | 307.1633 |

Same six-split suite comparison against recent Scribe/109M runs:

| Model | Common Voice WER/CER | FLEURS WER/CER | Mana WER/CER | Neyshekar WER/CER | WorldSpeech WER/CER | Youtube WER/CER | Macro WER | Macro CER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scribe-v2 | 31.2813 / 14.2129 | 9.9009 / 3.7559 | 14.5397 / 5.2297 | 15.1685 / 5.9494 | 31.2997 / 19.5489 | 29.2287 / 15.8821 | 21.9031 | 10.7631 |
| parakeet-scribe-classified-20260524 | 22.8926 / 6.1653 | 17.6863 / 4.8107 | 12.5752 / 3.3624 | 18.6476 / 4.2302 | 36.1756 / 20.0199 | 27.3863 / 12.8958 | 22.5606 | 8.5807 |
| omni-scribe-exact-300m-best | 21.8685 / 5.3868 | 9.8093 / 2.8246 | 27.5860 / 4.6107 | 28.3151 / 4.5277 | 38.7759 / 19.8776 | 36.3050 / 14.3965 | 27.1100 | 8.6040 |
| parakeet-exact-match-20260525 | 34.6371 / 10.4967 | 27.1488 / 8.7340 | 20.6366 / 5.6452 | 27.8801 / 7.5718 | 46.4384 / 24.3951 | 46.2195 / 21.3231 | 33.8267 | 13.0276 |

Readout:

- The 300M exact model is strongest on Common Voice and FLEURS among the recent full-suite runs.
- CER is competitive with the 109M classified run, but WER is much worse on Mana, Neyshekar, WorldSpeech, and Youtube.
- The WER/CER split points to word-level, spacing, boundary, or lexical convention errors more than pure character accuracy failure.
- The exact-only 300M run looks like a clean-domain win, rather than the best broad-domain recipe.
- The 109M classified run still has the best broad six-split balance.

Older partial 300M comparisons use different benchmark surfaces and sample counts, so treat them as directional:

| Model | FLEURS WER/CER | Neyshekar WER/CER | Youtube WER/CER |
|---|---:|---:|---:|
| omni-scribe-exact-300m-best | 9.8093 / 2.8246 | 28.3151 / 4.5277 | 36.3050 / 14.3965 |
| omni-target-300m | 16.0716 / 4.4575 | 19.56 / 4.30 | 27.74 / 12.22 |
| omni-balanced-100h | - | 20.29 / 4.39 | 28.08 / 12.48 |
| omni-clean-100h | - | 26.44 / 5.25 | 38.77 / 15.95 |
| omni-wer35-fastconformer | 13.13 / 3.98 | 25.29 / 4.72 | 28.59 / 12.48 |

Next move:

1. Keep the 300M exact checkpoint as the current best clean/FLEURS/Common Voice result.
2. Use the 109M classified recipe as the broad-domain reference, because it wins macro WER and nearly ties macro CER.
3. Build the next 300M training set from exact rows plus reviewed script-equivalent rows plus high-confidence classified rows, then export one Persian-script manifest.
4. Before adding rows, resolve the Latin-Scribe/Persian-reference issue as a review signal only: Latin Scribe can prove same spoken content, while training labels remain Persian script.
5. Re-run the same six-split canonical suite for every new checkpoint, because older single-dataset benchmark names hide sample-count drift.
