#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/simon/github/peacock-asr/projects/P004-training-from-scratch"
REMOTE_ROOT="/home/simon/github/peacock-asr/projects/P004-training-from-scratch"
LOG_FILE="$PROJECT_ROOT/experiments/logs/watch_runpod_train_and_prepare_full_librispeech_$(date +%Y%m%d_%H%M%S).log"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <runpod-pod-id> <run-id>" >&2
  exit 1
fi

POD_ID="$1"
RUN_ID="$2"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_FILE"
}

ssh_info_json() {
  runpodctl ssh info "$POD_ID"
}

read_ssh_field() {
  local field="$1"
  python -c '
import json
import sys

payload = json.loads(sys.stdin.read())
field = sys.argv[1]
value = payload.get(field)
if value is None:
    raise SystemExit(1)
print(value)
' "$field"
}

main() {
  mkdir -p "$(dirname "$LOG_FILE")"
  : > "$LOG_FILE"

  log "watching remote run_id=$RUN_ID on pod_id=$POD_ID"
  local ssh_json host port key_path
  ssh_json="$(ssh_info_json)"
  host="$(printf '%s' "$ssh_json" | read_ssh_field ip)"
  port="$(printf '%s' "$ssh_json" | read_ssh_field port)"
  key_path="$(
    printf '%s' "$ssh_json" | python -c '
import json
import sys

payload = json.loads(sys.stdin.read())
ssh_key = payload.get("ssh_key") or {}
path = ssh_key.get("path")
if path is None:
    raise SystemExit(1)
print(path)
'
  )"
  local remote_output_dir="$REMOTE_ROOT/experiments/checkpoints/canonical_phone_ctc/$RUN_ID"

  while true; do
    if ssh \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -i "$key_path" \
      "root@${host}" \
      -p "$port" \
      "test -f '$remote_output_dir/report.json'"; then
      log "detected completed remote report: $remote_output_dir/report.json"
      break
    fi
    sleep 300
  done

  log "triggering full LibriSpeech preparation on the same pod"
  "$PROJECT_ROOT/scripts/prepare_runpod_full_librispeech.sh" "$POD_ID" >>"$LOG_FILE" 2>&1
  log "full LibriSpeech preparation completed"
}

main "$@"
