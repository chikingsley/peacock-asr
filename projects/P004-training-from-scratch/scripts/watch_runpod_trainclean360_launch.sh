#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/simon/github/peacock-asr/projects/P004-training-from-scratch"
REMOTE_ROOT="/home/simon/github/peacock-asr/projects/P004-training-from-scratch"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <runpod-pod-id> <local-launcher-pid>" >&2
  exit 1
fi

POD_ID="$1"
LAUNCHER_PID="$2"
LOG_FILE="$PROJECT_ROOT/experiments/logs/watch_runpod_trainclean360_launch_$(date +%Y%m%d_%H%M%S).log"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_FILE"
}

fetch_ssh_info() {
  runpodctl ssh info "$POD_ID"
}

read_field() {
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

read_key_path() {
  python -c '
import json
import sys

payload = json.loads(sys.stdin.read())
ssh_key = payload.get("ssh_key") or {}
path = ssh_key.get("path")
if path is None:
    raise SystemExit(1)
print(path)
'
}

remote_has_launcher() {
  local host="$1"
  local port="$2"
  local key_path="$3"
  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "bash -lc 'latest=\$(ls -dt \"$REMOTE_ROOT\"/experiments/checkpoints/canonical_phone_ctc/canonical_runpod_a100_trainclean360_conformer_h512_l12_b16_e5_* 2>/dev/null | head -1); if [[ -n \"\$latest\" && -f \"\$latest/launcher.pid\" ]]; then echo \"\$latest\"; fi'"
}

main() {
  mkdir -p "$(dirname "$LOG_FILE")"
  : > "$LOG_FILE"
  log "watching launcher pid=$LAUNCHER_PID for pod_id=$POD_ID"

  while kill -0 "$LAUNCHER_PID" 2>/dev/null; do
    sleep 60
  done
  log "launcher pid exited"

  local ssh_json host port key_path
  ssh_json="$(fetch_ssh_info)"
  host="$(printf '%s' "$ssh_json" | read_field ip)"
  port="$(printf '%s' "$ssh_json" | read_field port)"
  key_path="$(printf '%s' "$ssh_json" | read_key_path)"

  local launched_dir
  launched_dir="$(remote_has_launcher "$host" "$port" "$key_path")" || true
  if [[ -n "$launched_dir" ]]; then
    log "remote launcher already exists: $launched_dir"
    exit 0
  fi

  log "remote launcher missing; retrying launch script"
  bash "$PROJECT_ROOT/scripts/launch_runpod_trainclean360.sh" "$POD_ID" >> "$LOG_FILE" 2>&1

  launched_dir="$(remote_has_launcher "$host" "$port" "$key_path")" || true
  if [[ -z "$launched_dir" ]]; then
    log "retry completed but remote launcher is still missing"
    exit 1
  fi
  log "remote launcher created after retry: $launched_dir"
}

main "$@"
