# Parakeet CTC 109M Scribe Classified 20260524 Run

This note records the data, launch parameters, training state, and post-training benchmark results
for `parakeet-ctc-109m-scribe-classified-20260524-bd700-min10`.

## Current Data Artifacts

Completed Scribe audit/filter source:

- Audit input:
  `data/curation/scribe_jobs/scribe-canonical-all-20260516T192536Z/joined.full-20260523.audit.jsonl`
- Filtered audit directory:
  `data/curation/scribe_jobs/scribe-canonical-all-20260516T192536Z/filtered-audit/`
- Kept rows:
  `571,092`
- Rejected rows:
  `133,363`

The filter keeps these categories:

- `exact_match`
- `near_match`
- `punctuation_or_orthography_only`
- `number_or_symbol_mismatch`
- `named_entity_mismatch`

The new NeMo manifests are ready here:

```text
data/training/scribe-verified/full-20260523-classified-keep/manifests/
```

Filtered manifest stats:

| Split | Manifest | Rows | Hours |
| --- | --- | ---: | ---: |
| train | `train.filtered-maxchars400-cps60.jsonl` | 520,812 | 844.35 |
| dev | `dev.filtered-maxchars400-cps60.jsonl` | 24,434 | 42.51 |
| test | `test.filtered-maxchars400-cps60.jsonl` | 23,910 | 43.86 |

The manifest label policy is:

```text
source.normalized_text else audit.normalized_reference else source.text else audit.reference_text
```

Scribe is used as a verifier/filter. The model trains on canonical/reference labels, not raw
`scribe_text`.

Validation already done:

- `0` missing source rows
- `0` missing audio files
- `0` empty labels
- Filtered manifests use `max_chars=400` and `max_chars_per_second=60.0`, matching the prior
  Parakeet run filter.

## Previous Real Parakeet Run

Previous full run:

```text
parakeet/runs/parakeet-ctc-109m-broad-plus-mana-20260515-bd700-min10/2026-05-15_02-55-55/
```

Previous command used:

```text
--use-lhotse
--batch-duration 700
--quadratic-duration 15
--num-buckets 30
--batch-size 1
--validation-batch-size 8
--accumulate-grad-batches 1
--min-epochs 10
--max-epochs 30
--min-steps 49410
--max-steps 148230
--early-stopping
--early-stopping-patience 5
--early-stopping-min-delta 0.001
--learning-rate 1e-4
--warmup-steps 5000
--min-lr 5e-6
--val-check-interval 4941
--log-every-n-steps 50
--num-workers 4
--disable-cudnn
```

Important old-run evidence:

- `4941` was the old run's validation interval and first observed epoch progress-bar count.
- The old launch log confirms validation at `Epoch 0, global step 4941`.
- The old run stopped by early stopping at `Epoch 17, global step 138378`.
- Best checkpoint was epoch 12:
  `parakeet-ctc-109m-broad-plus-mana-20260515-bd700-min10--val_wer=0.2926-epoch=12.ckpt`

## What Must Change For The Next Run

Keep the old recipe:

- same base CTC model:
  `models/parakeet-ctc-109m/ctc.nemo`
- same tokenizer:
  `tokenizers/parakeet/fa_spe_bpe_v1024/tokenizer_spe_bpe_v1024`
- same Lhotse dynamic bucketing recipe:
  `batch_duration=700`, `quadratic_duration=15`, `num_buckets=30`
- same optimizer schedule:
  `lr=1e-4`, `warmup_steps=5000`, `min_lr=5e-6`
- same minimum/maximum epoch plan:
  `min_epochs=10`, `max_epochs=30`
- same early stopping:
  `val_wer`, patience `5`, min delta `0.001`

Update these fields:

- train manifest path
- validation manifest path
- experiment name
- old fixed step controls

Do not blindly copy the old fixed step controls:

```text
--min-steps 49410
--max-steps 148230
--val-check-interval 4941
```

Those values were derived from the old manifest's effective epoch size. The new manifest has a
different size and distribution, so the old integers are not general-purpose settings.

NeMo's Lhotse path cannot infer dataset length for scheduler setup, so this run still needs
explicit step controls. Scale the old interval by filtered training hours:

```text
old train: 960.69h
new train: 844.35h
4941 * 844.35 / 960.69 = 4342
```

Use these fixed values for the next launch:

```text
--val-check-interval 4342
--min-steps 43420
--max-steps 130260
```

This preserves the old intent:

- train at least roughly 10 old-recipe epoch intervals
- validate once per estimated epoch interval
- allow early stopping after the minimum step window
- cap training at roughly 30 estimated epoch intervals

## Proposed Launch Command

Run this from inside a tmux session named `4` on `gmk-server`; do not launch it from a fragile SSH
foreground command.

```bash
cd /home/simon/github/peacock-asr/projects/persian-asr
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/persian-finetune-parakeet-ctc \
  --train-manifest data/training/scribe-verified/full-20260523-classified-keep/manifests/train.filtered-maxchars400-cps60.jsonl \
  --validation-manifest data/training/scribe-verified/full-20260523-classified-keep/manifests/dev.filtered-maxchars400-cps60.jsonl \
  --tokenizer-dir tokenizers/parakeet/fa_spe_bpe_v1024/tokenizer_spe_bpe_v1024 \
  --init-from-nemo-model models/parakeet-ctc-109m/ctc.nemo \
  --exp-dir parakeet/runs \
  --name parakeet-ctc-109m-scribe-classified-20260524-bd700-min10 \
  --use-lhotse \
  --batch-duration 700 \
  --quadratic-duration 15 \
  --num-buckets 30 \
  --batch-size 1 \
  --validation-batch-size 8 \
  --accumulate-grad-batches 1 \
  --min-epochs 10 \
  --max-epochs 30 \
  --min-steps 43420 \
  --max-steps 130260 \
  --early-stopping \
  --early-stopping-patience 5 \
  --early-stopping-min-delta 0.001 \
  --learning-rate 1e-4 \
  --warmup-steps 5000 \
  --min-lr 5e-6 \
  --val-check-interval 4342 \
  --log-every-n-steps 50 \
  --num-workers 4 \
  --disable-progress-bar \
  --disable-cudnn
```

`--disable-progress-bar` is optional for interactive launches, but preferred for logged tmux runs.
With Lhotse dynamic duration batching, Lightning cannot know the epoch length, so the default
progress bar shows `?/??` and noisy carriage-return updates when captured into logs.

## Current Launch

Training was launched in tmux session `4` on 2026-05-24 with the same command above, except the
currently running process does not include `--disable-progress-bar` because that option was added
after launch.

Active launch log:

```text
parakeet/runs/_launch_logs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10-steps4342.log
```

Active run directory:

```text
parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/
```

First validation completed successfully:

- global step: `4342`
- epoch: `0`
- `val_wer`: `0.67165`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.6716-epoch=0.ckpt`
- exported `.nemo`:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued past the validation point without OOM or traceback.

Second validation completed successfully:

- TensorBoard step: `10254`
- epoch: `1`
- `val_wer`: `0.43442`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.4344-epoch=1.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Note: with Lightning's integer `val_check_interval`, the interval is batch-local within each
epoch. In this Lhotse run, the progress counter resets at each epoch, so validation occurs near
batch `4342` of each epoch rather than exactly every `4342` global steps.

Third validation completed successfully:

- TensorBoard step: `16166`
- Lightning log global step: `16167`
- epoch: `2`
- `val_wer`: `0.35184`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.3518-epoch=2.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Fourth validation completed successfully:

- TensorBoard step: `22078`
- Lightning log global step: `22079`
- epoch: `3`
- `val_wer`: `0.31054`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.3105-epoch=3.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Fifth validation completed successfully:

- TensorBoard step: `27992`
- Lightning log global step: `27993`
- epoch: `4`
- `val_wer`: `0.28491`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2849-epoch=4.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Sixth validation completed successfully:

- TensorBoard step: `33904`
- Lightning log global step: `33905`
- epoch: `5`
- `val_wer`: `0.26588`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2659-epoch=5.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Seventh validation completed successfully:

- TensorBoard step: `39816`
- Lightning log global step: `39817`
- epoch: `6`
- `val_wer`: `0.25395`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2539-epoch=6.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Eighth validation completed successfully:

- TensorBoard step: `45730`
- Lightning log global step: `45731`
- epoch: `7`
- `val_wer`: `0.24343`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2434-epoch=7.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Ninth validation completed successfully:

- TensorBoard step: `51642`
- Lightning log global step: `51643`
- epoch: `8`
- `val_wer`: `0.23535`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2353-epoch=8.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Tenth validation completed successfully:

- TensorBoard step: `57555`
- Lightning log global step: `57556`
- epoch: `9`
- `val_wer`: `0.22903`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2290-epoch=9.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Eleventh validation completed successfully:

- TensorBoard step: `63468`
- Lightning log global step: `63469`
- epoch: `10`
- `val_wer`: `0.22435`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2244-epoch=10.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation because `val_wer` improved by `0.00468`, above the
early-stopping `min_delta=0.001`.

Twelfth validation completed successfully:

- TensorBoard step: `69381`
- Lightning log global step: `69382`
- epoch: `11`
- `val_wer`: `0.21933`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2193-epoch=11.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation because `val_wer` improved by `0.00503`, above the
early-stopping `min_delta=0.001`.

Thirteenth validation completed successfully:

- TensorBoard step: `75292`
- Lightning log global step: `75293`
- epoch: `12`
- `val_wer`: `0.21538`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2154-epoch=12.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation because `val_wer` improved by `0.00395`, above the
early-stopping `min_delta=0.001`.

Fourteenth validation completed successfully:

- TensorBoard step: `81203`
- Lightning log global step: `81204`
- epoch: `13`
- `val_wer`: `0.21228`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2123-epoch=13.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation because `val_wer` improved by `0.00310`, above the
early-stopping `min_delta=0.001`.

Fifteenth validation completed successfully:

- TensorBoard step: `87116`
- Lightning log global step: `87117`
- epoch: `14`
- `val_wer`: `0.21004`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2100-epoch=14.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation because `val_wer` improved by `0.00223`, above the
early-stopping `min_delta=0.001`.

Sixteenth validation completed successfully:

- TensorBoard step: `93029`
- Lightning log global step: `93030`
- epoch: `15`
- `val_wer`: `0.20781`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2078-epoch=15.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation because `val_wer` improved by `0.00224`, above the
early-stopping `min_delta=0.001`.

Seventeenth validation completed successfully:

- TensorBoard step: `98941`
- Lightning log global step: `98942`
- epoch: `16`
- `val_wer`: `0.20626`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2063-epoch=16.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation because `val_wer` improved by `0.00154`, above the
early-stopping `min_delta=0.001`.

Eighteenth validation completed successfully:

- TensorBoard step: `104854`
- Lightning log global step: `104855`
- epoch: `17`
- `val_wer`: `0.20465`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2046-epoch=17.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation because `val_wer` improved by `0.00161`, above the
early-stopping `min_delta=0.001`.

Nineteenth validation completed successfully:

- TensorBoard step: `110766`
- Lightning log global step: `110767`
- epoch: `18`
- `val_wer`: `0.20326`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2033-epoch=18.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation because `val_wer` improved by `0.00139`, above the
early-stopping `min_delta=0.001`.

Twentieth validation completed successfully:

- TensorBoard step: `116679`
- Lightning log global step: `116680`
- epoch: `19`
- `val_wer`: `0.20248`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2025-epoch=19.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation, but `val_wer` improved by only `0.00078`, below the
early-stopping `min_delta=0.001`. This likely starts the early-stopping patience counter unless a
later validation improves by at least `0.001` from the tracked best.

Twenty-first validation completed successfully:

- TensorBoard step: `122591`
- Lightning log global step: `122592`
- epoch: `20`
- `val_wer`: `0.20122`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2012-epoch=20.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation because `val_wer` improved by `0.00125`, above the
early-stopping `min_delta=0.001`.

Twenty-second validation completed successfully:

- TensorBoard step: `128504`
- Lightning log global step: `128505`
- epoch: `21`
- `val_wer`: `0.20084`
- checkpoint:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10--val_wer=0.2008-epoch=21.ckpt`
- exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

Training continued after this validation, but `val_wer` improved by only `0.00038`, below the
early-stopping `min_delta=0.001`. The run is near the hard `max_steps=130260` cap.

Training then stopped at the hard step cap:

- stop reason in launch log: ``Trainer.fit` stopped: `max_steps=130260` reached.`
- final TensorBoard train step observed: `130249`
- final validation remained epoch `21` with `val_wer=0.20084`
- final exported `.nemo` updated:
  `parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo`

## Post-Training Evaluation Commands

After training stops, choose the best checkpoint/export from the run's `checkpoints/` directory,
then score the held-out Scribe test manifest:

```bash
.venv/bin/persian-benchmark-asr \
  --source manifest \
  --manifest data/training/scribe-verified/full-20260523-classified-keep/manifests/test.filtered-maxchars400-cps60.jsonl \
  --model nvidia \
  --nvidia-model-name parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo \
  --decoder-type ctc \
  --batch-size 8 \
  --output-dir benchmarks/parakeet-ctc-109m-scribe-classified-20260524-heldout-test
```

Run the canonical individual benchmark suite with the same model path:

```bash
.venv/bin/persian-benchmark-suite \
  --suite-name canonical-tests-parakeet-scribe-classified-20260524 \
  --models parakeet-scribe-classified-20260524 \
  --local-nvidia-model-name parakeet/runs/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10/2026-05-24_15-37-02/checkpoints/parakeet-ctc-109m-scribe-classified-20260524-bd700-min10.nemo \
  --local-nvidia-model-label parakeet-scribe-classified-20260524 \
  --local-nvidia-batch-size 8 \
  --local-nvidia-decoder-type ctc
```

Compare the suite output against:

```text
benchmarks/suites/canonical-tests/summary.md
benchmarks/suites/canonical-tests/summary.json
benchmarks/suites/canonical-tests/summary.tsv
```

## Post-Training Evaluation Results

Held-out Scribe test manifest:

- command output: `benchmarks/parakeet-ctc-109m-scribe-classified-20260524-heldout-test/summary.md`
- samples: `23910`
- WER: `20.69%` (`0.2069`)
- CER: `6.63%` (`0.0663`)
- audio: `43.86h`
- throughput: `625.13x`

Canonical individual benchmark suite:

- command output: `benchmarks/suites/canonical-tests-parakeet-scribe-classified-20260524/summary.md`
- aggregate TSV: `benchmarks/suites/canonical-tests-parakeet-scribe-classified-20260524/summary.tsv`
- aggregate JSON: `benchmarks/suites/canonical-tests-parakeet-scribe-classified-20260524/summary.json`

| Dataset | Split | WER | CER | Samples | Audio h |
|---|---|---:|---:|---:|---:|
| common_voice_25 | test | 22.89% | 6.17% | 10702 | 14.69 |
| fleurs | test | 17.69% | 4.81% | 852 | 3.60 |
| mana_tts | test | 12.58% | 3.36% | 3989 | 5.35 |
| neyshekar | test | 18.65% | 4.23% | 1331 | 2.05 |
| worldspeech | test | 36.18% | 20.02% | 359 | 1.33 |
| youtube | test | 27.39% | 12.90% | 13899 | 33.35 |

Comparison against the previous canonical suite (`benchmarks/suites/canonical-tests/summary.md`):

- versus `omni-target-300m`: better on common_voice_25 (`-10.38` WER points),
  worldspeech (`-1.03`), and youtube (`-0.35`); worse on fleurs (`+1.61`).
- versus `parakeet-ctc-109m-broad-plus-mana-best`: better on common_voice_25
  (`-39.89`), mana_tts (`-2.20`), and worldspeech (`-0.67`); worse on fleurs
  (`+0.42`) and youtube (`+1.96`).
- versus `parakeet-110m-broad-canonical-filtered-best-ctc`: better on every
  overlapping dataset by `24.16` to `35.40` WER points.

Post-evaluation process note: after the model export and evaluations completed, the original trainer
process (`PID 3597266`) was still alive in a futex wait using CPU while GPU utilization was `0%`.
It was not interrupted during evaluation.
