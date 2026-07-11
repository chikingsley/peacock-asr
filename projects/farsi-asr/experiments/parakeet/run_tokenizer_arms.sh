#!/usr/bin/env bash
# Matched 2k-step tokenizer-size ablations using the winning hybrid-loss weight.
set -euo pipefail
cd /home/simon/github/peacock-asr/projects/farsi-asr

UV=/home/simon/.local/bin/uv
CTC_WEIGHT=${CTC_WEIGHT:-0.0}
CORPUS=experiments/lm_decoding/corpus.txt
TOKENIZER_ROOT=data/parakeet/tokenizers
COMMON=(--model-name ../../base_models/parakeet/parakeet-tdt_ctc-110m-base-hybrid.nemo
  --train-manifest data/parakeet/manifests/mix50_train.jsonl
  --validation-manifest data/parakeet/manifests/val_fixed.jsonl
  --max-steps 2000 --val-every 250 --warmup 200 --batch-dur 120 --lr 5e-4 --seed 0
  --ctc-loss-weight "$CTC_WEIGHT")

build_tokenizer() {
  local size=$1
  local name="fa_spe_bpe_v${size}_scribe_v4"
  local model="${TOKENIZER_ROOT}/${name}/tokenizer_spe_bpe_v${size}/tokenizer.model"
  if [[ -f "$model" ]]; then
    echo "=== TOKENIZER $size already exists: $model ==="
    return
  fi
  echo "=== TOKENIZER $size build start $(date +%H:%M:%S) ==="
  "$UV" run --no-sync farsi-parakeet-train-tokenizer \
    --data-file "$CORPUS" \
    --output-root "$TOKENIZER_ROOT" \
    --name "$name" \
    --vocab-size "$size"
  echo "=== TOKENIZER $size build done $(date +%H:%M:%S) ==="
}

run_arm() {
  local run_name=$1
  local size=$2
  local tokenizer="${TOKENIZER_ROOT}/fa_spe_bpe_v${size}_scribe_v4/tokenizer_spe_bpe_v${size}"
  echo "=== ARM $run_name bpe=$size ctc_loss_weight=$CTC_WEIGHT start $(date +%H:%M:%S) ==="
  "$UV" run --no-sync farsi-parakeet-train-tdt --name "$run_name" "${COMMON[@]}" \
    --tokenizer-dir "$tokenizer" \
    2>&1 | tee "experiments/parakeet/${run_name}.log" | awk '/val @|saved last-step|Traceback|error:/'

  for surface in fleurs_dev youtube_hf_dev_conv; do
    echo "=== EVAL $run_name $surface ==="
    "$UV" run --no-sync farsi-parakeet-eval \
      --kind tdt \
      --model-name "runs/parakeet/${run_name}/${run_name}_best-valloss.nemo" \
      --manifest "data/parakeet/manifests/${surface}.jsonl" \
      --device cuda \
      --batch-size 32 \
      --sample-count 0 \
      --output-summary-json "experiments/parakeet/evals/${run_name}_${surface}.json" \
      2>&1 | tee "experiments/parakeet/evals/${run_name}_${surface}.log" \
      | awk '/normalized WER\/CER|raw WER\/CER|RTFx=|peak CUDA|Traceback|error:/'
  done
  echo "=== ARM $run_name done $(date +%H:%M:%S) ==="
}

mkdir -p experiments/parakeet/evals
build_tokenizer 512
build_tokenizer 2048
run_arm b11-bpe512 512
run_arm b12-bpe2048 2048
echo "ALL_TOKENIZER_ARMS_DONE"
