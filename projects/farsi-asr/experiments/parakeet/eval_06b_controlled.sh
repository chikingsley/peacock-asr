#!/usr/bin/env bash
# Fixed fp32 scorecard for the first controlled 0.6B Farsi transfer.
set -euo pipefail
cd /home/simon/github/peacock-asr/projects/farsi-asr

UV=/home/simon/.local/bin/uv
NAME=d0-bpe512-spec-off-06b-ada-s0
MODEL="runs/parakeet/${NAME}/${NAME}_best-valloss.nemo"

mkdir -p experiments/parakeet/evals
for surface in fleurs_dev youtube_hf_dev_conv fleurs_test neyshekar_test worldspeech_test youtube_hf_test_conv; do
  echo "=== EVAL $NAME $surface ==="
  "$UV" run --no-sync farsi-parakeet-eval \
    --kind tdt \
    --model-name "$MODEL" \
    --manifest "data/parakeet/manifests/${surface}.jsonl" \
    --device cuda \
    --batch-size 8 \
    --sample-count 0 \
    --output-summary-json "experiments/parakeet/evals/${NAME}_${surface}.json" \
    2>&1 | tee "experiments/parakeet/evals/${NAME}_${surface}.log" \
    | awk '/normalized WER\/CER|raw WER\/CER|RTFx=|peak CUDA|Traceback|CUDA out of memory|error:/'
done
echo "CONTROLLED_06B_EVAL_DONE $(date +%H:%M:%S)"
