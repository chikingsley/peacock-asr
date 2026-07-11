#!/usr/bin/env bash
# Matched-audio 0.6B confirmation: two 60-second microbatches per optimizer update.
set -euo pipefail
cd /home/simon/github/peacock-asr/projects/farsi-asr

UV=/home/simon/.local/bin/uv
NAME=d1-bpe512-spec-off-06b-ada-acc2-s0
MODEL=../../base_models/parakeet/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo
TOKENIZER=data/parakeet/tokenizers/fa_spe_bpe_v512_scribe_v4/tokenizer_spe_bpe_v512

echo "=== ARM $NAME start $(date +%H:%M:%S) ==="
"$UV" run --project . --frozen farsi-parakeet-train-tdt \
  --name "$NAME" \
  --model-name "$MODEL" \
  --train-manifest data/parakeet/manifests/gate2_full_train.jsonl \
  --validation-manifest data/parakeet/manifests/val_fixed.jsonl \
  --tokenizer-dir "$TOKENIZER" \
  --recipe simple \
  --max-steps 10000 \
  --val-every 500 \
  --warmup 1000 \
  --batch-dur 60 \
  --accumulate-grad-batches 2 \
  --fused-batch-size 2 \
  --lr 5e-4 \
  --optim adafactor \
  --seed 0 \
  --spec-augment off \
  2>&1 | tee "experiments/parakeet/${NAME}.log" | awk '/loaded |RNNTLoss|val @|SpecAugment profile|saved last-step|Traceback|CUDA out of memory|error:/'

mkdir -p experiments/parakeet/evals
for surface in fleurs_dev youtube_hf_dev_conv fleurs_test neyshekar_test worldspeech_test youtube_hf_test_conv; do
  echo "=== EVAL $NAME $surface ==="
  "$UV" run --project . --frozen farsi-parakeet-eval \
    --kind tdt \
    --model-name "runs/parakeet/${NAME}/${NAME}_best-valloss.nemo" \
    --manifest "data/parakeet/manifests/${surface}.jsonl" \
    --device cuda \
    --batch-size 8 \
    --sample-count 0 \
    --output-summary-json "experiments/parakeet/evals/${NAME}_${surface}.json" \
    2>&1 | tee "experiments/parakeet/evals/${NAME}_${surface}.log" \
    | awk '/normalized WER\/CER|raw WER\/CER|RTFx=|peak CUDA|Traceback|CUDA out of memory|error:/'
done
echo "MATCHED_AUDIO_06B_DONE $(date +%H:%M:%S)"
