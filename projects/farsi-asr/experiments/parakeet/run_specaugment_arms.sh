#!/usr/bin/env bash
# Full-data 10k-step SpecAugment ablations for the pure-TDT BPE-512 winner.
set -euo pipefail
cd /home/simon/github/peacock-asr/projects/farsi-asr

UV=/home/simon/.local/bin/uv
TOKENIZER=data/parakeet/tokenizers/fa_spe_bpe_v512_scribe_v4/tokenizer_spe_bpe_v512
COMMON=(--model-name ../../base_models/parakeet/parakeet-tdt_ctc-110m-base-hybrid.nemo
  --train-manifest data/parakeet/manifests/gate2_full_train.jsonl
  --validation-manifest data/parakeet/manifests/val_fixed.jsonl
  --tokenizer-dir "$TOKENIZER"
  --max-steps 10000 --val-every 500 --warmup 1000 --batch-dur 120 --lr 5e-4 --seed 0
  --ctc-loss-weight 0.0)

run_arm() {
  local name=$1
  local profile=$2
  echo "=== ARM $name spec_augment=$profile start $(date +%H:%M:%S) ==="
  "$UV" run --no-sync farsi-parakeet-train-tdt --name "$name" "${COMMON[@]}" \
    --spec-augment "$profile" \
    2>&1 | tee "experiments/parakeet/${name}.log" | awk '/val @|SpecAugment profile|saved last-step|Traceback|error:/'

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
run_arm c0-bpe512-spec-current-s0 current
run_arm c1-bpe512-spec-half-s0 half
run_arm c2-bpe512-spec-off-s0 off
echo "ALL_SPECAUGMENT_ARMS_DONE"
