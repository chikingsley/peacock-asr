#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="$ROOT/third_party/P014-hmamba-original/data/so762"

base_dir="${1:-runs/repro-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$base_dir"

seeds=(824 17 2413 168 623)

for seed in "${seeds[@]}"; do
  exp_dir="$base_dir/seed${seed}"
  mkdir -p "$exp_dir"

  echo "=== TRAIN ${seed} ==="
  uv run --group train p012-train \
    --seed "$seed" \
    --lr 2e-3 \
    --warmup-step 300 \
    --batch-size 50 \
    --n-epochs 20 \
    --loss-w-phn 1 \
    --loss-w-word 1 \
    --loss-w-utt 1 \
    --loss-type dexent \
    --loss-w-a 0.7 \
    --loss-w-xent 0.003 \
    --selection-metric phone_mse \
    --model-conf conf/so762/HMamba.yaml \
    --am librispeech \
    --phn-dict local/so762/vocab_merge.json \
    --gop-dir "$DATA_ROOT/gop-librispeech-bies" \
    --ssl-dir "$DATA_ROOT/wav2vec2-large-xlsr-53 $DATA_ROOT/hubert-large-ll60k $DATA_ROOT/wavlm-large" \
    --raw-dir "$DATA_ROOT/raw-audio" \
    --exp-dir "$exp_dir" \
    2>&1 | tee "$exp_dir/train.log"

  echo "=== RECOG ${seed} ==="
  uv run --group train p012-recog \
    --remove-sil \
    --remove-special-token \
    --checkpoint-name best_audio_model.pth \
    --model-conf conf/so762/HMamba.yaml \
    --am librispeech \
    --phn-dict local/so762/vocab_merge.json \
    --gop-dir "$DATA_ROOT/gop-librispeech-bies" \
    --ssl-dir "$DATA_ROOT/wav2vec2-large-xlsr-53 $DATA_ROOT/hubert-large-ll60k $DATA_ROOT/wavlm-large" \
    --raw-dir "$DATA_ROOT/raw-audio" \
    --exp-dir "$exp_dir" \
    2>&1 | tee "$exp_dir/recog.log"

  echo "=== MDD ${seed} ==="
  uv run --group train p012-mdd-eval \
    --exp-dir "$exp_dir" \
    --output "$exp_dir/mdd_result.txt" \
    --json | tee "$exp_dir/mdd_result.json"
done
