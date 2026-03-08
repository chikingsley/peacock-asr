#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/workspace/p004-training-from-scratch"
REMOTE_CACHE_DIR="$REMOTE_ROOT/.cache/uv"
DEFAULT_SPLITS=(
  train_clean_100
  dev_clean
  train_other_500
  test_clean
  test_other
)

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <runpod-pod-id> [split ...]" >&2
  exit 1
fi

POD_ID="$1"
shift
if [[ $# -gt 0 ]]; then
  SPLITS=("$@")
else
  SPLITS=("${DEFAULT_SPLITS[@]}")
fi

RUN_TAG="prepare_librispeech_hf_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_ROOT/experiments/logs/${RUN_TAG}.log"

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

wait_for_ssh() {
  local host="$1"
  local port="$2"
  local key_path="$3"
  local attempt
  for attempt in $(seq 1 60); do
    if ssh \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=5 \
      -i "$key_path" \
      "root@${host}" \
      -p "$port" \
      'echo ready' >/dev/null 2>&1; then
      log "ssh ready on ${host}:${port}"
      return 0
    fi
    sleep 10
  done
  log "timed out waiting for ssh readiness"
  exit 1
}

sync_project_bits() {
  local host="$1"
  local port="$2"
  local key_path="$3"

  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "mkdir -p '$REMOTE_ROOT/experiments/data/lang_phone' '$REMOTE_ROOT/experiments/logs' '$REMOTE_ROOT/resources' '$REMOTE_CACHE_DIR'"

  if [[ -f "$HOME/.netrc" ]]; then
    scp \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -i "$key_path" \
      -P "$port" \
      "$HOME/.netrc" \
      "root@${host}:/root/.netrc"
  fi

  rsync -a --no-owner --no-group --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $key_path -p $port" \
    "$PROJECT_ROOT/README.md" \
    "$PROJECT_ROOT/pyproject.toml" \
    "$PROJECT_ROOT/uv.lock" \
    "$PROJECT_ROOT/.python-version" \
    "$PROJECT_ROOT/code" \
    "$PROJECT_ROOT/resources" \
    "root@${host}:$REMOTE_ROOT/"

  rsync -a --no-owner --no-group --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $key_path -p $port" \
    "$PROJECT_ROOT/experiments/data/lang_phone/" \
    "root@${host}:$REMOTE_ROOT/experiments/data/lang_phone/"
}

launch_remote_build() {
  local host="$1"
  local port="$2"
  local key_path="$3"
  local remote_log="$REMOTE_ROOT/experiments/logs/${RUN_TAG}.remote.log"
  local split_args=""
  local split
  for split in "${SPLITS[@]}"; do
    split_args+=" $(printf '%q' "$split")"
  done

  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "bash -lc 'cd \"$REMOTE_ROOT\" && mkdir -p experiments/data/manifests_phone_raw && nohup env P004_PROJECT_ROOT=\"$REMOTE_ROOT\" UV_CACHE_DIR=\"$REMOTE_CACHE_DIR\" uv run --script code/build_librispeech_phone_cuts.py --splits${split_args} > \"$remote_log\" 2>&1 < /dev/null & echo \$! > \"$REMOTE_ROOT/experiments/logs/${RUN_TAG}.pid\"'"

  log "remote HF prep launched: ${SPLITS[*]}"
  log "remote log: $remote_log"
}

main() {
  mkdir -p "$(dirname "$LOG_FILE")"
  : > "$LOG_FILE"

  log "fetching ssh info for pod_id=$POD_ID"
  local ssh_json
  ssh_json="$(ssh_info_json)"
  local host port key_path
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
  log "ssh endpoint ${host}:${port}"

  wait_for_ssh "$host" "$port" "$key_path"
  log "syncing project bits needed for HF prep"
  sync_project_bits "$host" "$port" "$key_path"
  log "sync complete"
  launch_remote_build "$host" "$port" "$key_path"
}

main
