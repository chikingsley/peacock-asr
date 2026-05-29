#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/simon/github/peacock-asr/projects/persian-asr"
RUN_NAME="parakeet-ctc-109m-scribe-exact-match-20260525-bd700"
PARAKEET_SESSION="parakeet-exact-109m"
PARAKEET_MODEL="$ROOT/parakeet/runs/$RUN_NAME/2026-05-25_07-40-48/checkpoints/$RUN_NAME.nemo"
PARAKEET_LAUNCH_LOG="$ROOT/parakeet/runs/_launch_logs/$RUN_NAME.log"
HELDOUT_OUT="$ROOT/benchmarks/parakeet-ctc-109m-scribe-exact-match-20260525-heldout-test"
SUITE_NAME="canonical-tests-parakeet-scribe-exact-match-20260525"
OMNI_CONFIG="$ROOT/configs/omni/persian-asr-scribe-exact-match-20260525-ctc-300m-v2.yaml"
OMNI_OUT="$ROOT/runs/omni-ctc-300m-scribe-exact-match-20260525"
WAIT_SECONDS=3600

cd "$ROOT"

timestamp() {
  date '+%F %T %Z'
}

echo "[$(timestamp)] post-109m pipeline waiting for $PARAKEET_SESSION"
while tmux has-session -t "$PARAKEET_SESSION" 2>/dev/null || \
  pgrep -f "persian-finetune-parakeet-ctc --train-manifest data/training/scribe-verified/exact-match-20260525/manifests/train.jsonl" >/dev/null; do
  if [[ -s "$PARAKEET_MODEL" ]] && grep -q '`Trainer.fit` stopped' "$PARAKEET_LAUNCH_LOG"; then
    echo "[$(timestamp)] 109m fit completion marker found; terminating stale post-fit session/process"
    pkill -TERM -f "persian-finetune-parakeet-ctc --train-manifest data/training/scribe-verified/exact-match-20260525/manifests/train.jsonl" || true
    sleep 30
    tmux kill-session -t "$PARAKEET_SESSION" 2>/dev/null || true
    break
  fi
  echo "[$(timestamp)] 109m still running; next check in ${WAIT_SECONDS}s"
  sleep "$WAIT_SECONDS"
done

echo "[$(timestamp)] 109m appears finished"
if [[ ! -s "$PARAKEET_MODEL" ]]; then
  echo "[$(timestamp)] missing final .nemo: $PARAKEET_MODEL" >&2
  exit 1
fi

echo "[$(timestamp)] running held-out exact-match test evaluation"
.venv/bin/persian-benchmark-asr \
  --source manifest \
  --model nvidia \
  --manifest data/training/scribe-verified/exact-match-20260525/manifests/test.jsonl \
  --output-dir "$HELDOUT_OUT" \
  --batch-size 8 \
  --nvidia-model-name "$PARAKEET_MODEL" \
  --decoder-type ctc

echo "[$(timestamp)] running canonical benchmark suite"
.venv/bin/persian-benchmark-suite \
  --suite-name "$SUITE_NAME" \
  --models parakeet-exact-match-20260525 \
  --local-nvidia-model-name "$PARAKEET_MODEL" \
  --local-nvidia-model-label parakeet-exact-match-20260525 \
  --local-nvidia-batch-size 8 \
  --local-nvidia-decoder-type ctc

echo "[$(timestamp)] launching Omni 300M CTC exact-match training"
.venv/bin/persian-train-omni \
  --config-file "$OMNI_CONFIG" \
  --output-dir "$OMNI_OUT"

echo "[$(timestamp)] post-109m pipeline complete"
