#!/usr/bin/env bash
# Confirm the promoted 110M pure-TDT BPE-512, no-SpecAugment recipe with seed 1.
set -euo pipefail
cd /home/simon/github/peacock-asr/projects/farsi-asr

UV=/home/simon/.local/bin/uv
NAME=c3-bpe512-spec-off-s1
TOKENIZER=data/parakeet/tokenizers/fa_spe_bpe_v512_scribe_v4/tokenizer_spe_bpe_v512

echo "=== ARM $NAME start $(date +%H:%M:%S) ==="
"$UV" run --no-sync farsi-parakeet-train-tdt \
  --name "$NAME" \
  --model-name ../../base_models/parakeet/parakeet-tdt_ctc-110m-base-hybrid.nemo \
  --train-manifest data/parakeet/manifests/gate2_full_train.jsonl \
  --validation-manifest data/parakeet/manifests/val_fixed.jsonl \
  --tokenizer-dir "$TOKENIZER" \
  --max-steps 10000 \
  --val-every 500 \
  --warmup 1000 \
  --batch-dur 120 \
  --lr 5e-4 \
  --seed 1 \
  --ctc-loss-weight 0.0 \
  --spec-augment off \
  2>&1 | tee "experiments/parakeet/${NAME}.log" | awk '/val @|SpecAugment profile|saved last-step|Traceback|error:/'

mkdir -p experiments/parakeet/evals
for surface in fleurs_dev youtube_hf_dev_conv fleurs_test neyshekar_test worldspeech_test youtube_hf_test_conv; do
  echo "=== EVAL $NAME $surface ==="
  "$UV" run --no-sync farsi-parakeet-eval \
    --kind tdt \
    --model-name "runs/parakeet/${NAME}/${NAME}_best-valloss.nemo" \
    --manifest "data/parakeet/manifests/${surface}.jsonl" \
    --device cuda \
    --batch-size 32 \
    --sample-count 0 \
    --output-summary-json "experiments/parakeet/evals/${NAME}_${surface}.json" \
    2>&1 | tee "experiments/parakeet/evals/${NAME}_${surface}.log" \
    | awk '/normalized WER\/CER|raw WER\/CER|RTFx=|peak CUDA|Traceback|error:/'
done
echo "WINNER_SEED1_DONE $(date +%H:%M:%S)"
