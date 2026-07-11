#!/usr/bin/env bash
# Matched 2k-step ablation arms on the mix50 surface (b3 recipe = lr 5e-4 baseline).
set -euo pipefail
cd /home/simon/github/peacock-asr/projects/farsi-asr

COMMON=(--model-name ../../base_models/parakeet/parakeet-tdt_ctc-110m-base-hybrid.nemo
  --train-manifest data/parakeet/manifests/mix50_train.jsonl
  --validation-manifest data/parakeet/manifests/val_fixed.jsonl
  --tokenizer-dir data/parakeet/tokenizers/fa_spe_bpe_v1024_scribe_v4/tokenizer_spe_bpe_v1024
  --max-steps 2000 --val-every 250 --warmup 200 --batch-dur 120 --lr 5e-4)

run_arm() {
  local name=$1; shift
  echo "=== ARM $name start $(date +%H:%M:%S) ==="
  uv run --no-sync farsi-parakeet-train-tdt --name "$name" "${COMMON[@]}" "$@" \
    2>&1 | tee "experiments/parakeet/${name}.log" | awk '/val @|saved last-step|Traceback|error:/'
  echo "=== ARM $name done $(date +%H:%M:%S) ==="
}

run_arm b4a-seed0 --seed 0
run_arm b4b-seed1 --seed 1
run_arm b5-adafactor --seed 0 --optim adafactor
echo "ALL_ARMS_DONE"
