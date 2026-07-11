#!/usr/bin/env bash
# Matched 2k-step auxiliary-CTC-weight ablations on the mix50 surface.
set -euo pipefail
cd /home/simon/github/peacock-asr/projects/farsi-asr

UV=/home/simon/.local/bin/uv
COMMON=(--model-name ../../base_models/parakeet/parakeet-tdt_ctc-110m-base-hybrid.nemo
  --train-manifest data/parakeet/manifests/mix50_train.jsonl
  --validation-manifest data/parakeet/manifests/val_fixed.jsonl
  --tokenizer-dir data/parakeet/tokenizers/fa_spe_bpe_v1024_scribe_v4/tokenizer_spe_bpe_v1024
  --max-steps 2000 --val-every 250 --warmup 200 --batch-dur 120 --lr 5e-4 --seed 0)

run_arm() {
  local name=$1
  local weight=$2
  echo "=== ARM $name ctc_loss_weight=$weight start $(date +%H:%M:%S) ==="
  "$UV" run --no-sync farsi-parakeet-train-tdt --name "$name" "${COMMON[@]}" \
    --ctc-loss-weight "$weight" \
    2>&1 | tee "experiments/parakeet/${name}.log" | awk '/val @|saved last-step|Traceback|error:/'

  for surface in fleurs_dev youtube_hf_dev_conv; do
    echo "=== EVAL $name $surface ==="
    "$UV" run --no-sync farsi-parakeet-eval \
      --kind tdt \
      --model-name "runs/parakeet/${name}/${name}_best-valloss.nemo" \
      --manifest "data/parakeet/manifests/${surface}.jsonl" \
      --device cuda \
      --batch-size 32 \
      --sample-count 0 \
      --output-summary-json "experiments/parakeet/evals/${name}_${surface}.json" \
      2>&1 | tee "experiments/parakeet/evals/${name}_${surface}.log" \
      | awk '/normalized WER\/CER|raw WER\/CER|RTFx=|peak CUDA|Traceback|error:/'
  done
  echo "=== ARM $name done $(date +%H:%M:%S) ==="
}

mkdir -p experiments/parakeet/evals
run_arm b7-ctc00 0.0
run_arm b8-ctc10 0.1
run_arm b9-ctc01 0.01
run_arm b10-ctc001 0.001
echo "ALL_HYBRID_WEIGHT_ARMS_DONE"
