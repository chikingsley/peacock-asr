#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/workspace/p004-training-from-scratch"
REMOTE_CACHE_DIR="/workspace/.cache/uv"
REMOTE_VENV="/workspace/venvs/p004-training-from-scratch"
TRAIN_MANIFEST="$PROJECT_ROOT/experiments/data/manifests_phone_raw/librispeech_cuts_train-clean-360.jsonl.gz"
DEV_MANIFEST="$PROJECT_ROOT/experiments/data/manifests_phone_raw/librispeech_cuts_dev-other.jsonl.gz"
TRAIN_AUDIO_DIR="$PROJECT_ROOT/experiments/data/manifests_phone_raw/audio/train_clean_360"
DEV_AUDIO_DIR="$PROJECT_ROOT/experiments/data/manifests_phone_raw/audio/dev_other"
LANG_DIR="$PROJECT_ROOT/experiments/data/lang_phone"
RUN_ID="canonical_runpod_a100_trainclean360_conformer_h512_l12_b16_e5_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_ROOT/experiments/logs/${RUN_ID}.log"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <runpod-pod-id>" >&2
  exit 1
fi

POD_ID="$1"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_FILE"
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    log "missing required path: $path"
    exit 1
  fi
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

sync_workspace() {
  local host="$1"
  local port="$2"
  local key_path="$3"
  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "mkdir -p '$REMOTE_ROOT/experiments/data/manifests_phone_raw/audio' '$REMOTE_ROOT/experiments/checkpoints/canonical_phone_ctc' '$REMOTE_ROOT/experiments/logs' '$REMOTE_ROOT/resources' '$REMOTE_CACHE_DIR' '$(dirname "$REMOTE_VENV")'"

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
    "$PROJECT_ROOT/src" \
    "root@${host}:$REMOTE_ROOT/"

  rsync -a --no-owner --no-group --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $key_path -p $port" \
    "$LANG_DIR/" \
    "root@${host}:$REMOTE_ROOT/experiments/data/lang_phone/"

  rsync -a --no-owner --no-group --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $key_path -p $port" \
    "$TRAIN_MANIFEST" \
    "$DEV_MANIFEST" \
    "root@${host}:$REMOTE_ROOT/experiments/data/manifests_phone_raw/"

  rsync -a --no-owner --no-group --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $key_path -p $port" \
    "$TRAIN_AUDIO_DIR/" \
    "root@${host}:$REMOTE_ROOT/experiments/data/manifests_phone_raw/audio/train_clean_360/"

  rsync -a --no-owner --no-group --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $key_path -p $port" \
    "$DEV_AUDIO_DIR/" \
    "root@${host}:$REMOTE_ROOT/experiments/data/manifests_phone_raw/audio/dev_other/"
}

prepare_remote_env() {
  local host="$1"
  local port="$2"
  local key_path="$3"
  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "bash -lc 'uv venv \"$REMOTE_VENV\" >/dev/null && cd \"$REMOTE_ROOT\" && export UV_CACHE_DIR=\"$REMOTE_CACHE_DIR\" VIRTUAL_ENV=\"$REMOTE_VENV\" PATH=\"$REMOTE_VENV/bin:\$PATH\" && P004_PROJECT_ROOT=\"$REMOTE_ROOT\" uv sync --active --group dev --group canonical'"
}

launch_remote_run() {
  local host="$1"
  local port="$2"
  local key_path="$3"
  local remote_output_dir="$REMOTE_ROOT/experiments/checkpoints/canonical_phone_ctc/$RUN_ID"
  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "bash -lc 'cd \"$REMOTE_ROOT\" && mkdir -p \"$remote_output_dir\" && export UV_CACHE_DIR=\"$REMOTE_CACHE_DIR\" VIRTUAL_ENV=\"$REMOTE_VENV\" PATH=\"$REMOTE_VENV/bin:\$PATH\" && nohup env P004_PROJECT_ROOT=\"$REMOTE_ROOT\" uv run --active p004-canonical-train --run-id \"$RUN_ID\" --train-manifest experiments/data/manifests_phone_raw/librispeech_cuts_train-clean-360.jsonl.gz --dev-manifest experiments/data/manifests_phone_raw/librispeech_cuts_dev-other.jsonl.gz --train-limit 0 --dev-limit 0 --epochs 5 --batch-size 16 --hidden-dim 512 --encoder-layers 12 --attention-heads 8 > \"$remote_output_dir/launcher.stdout.log\" 2>&1 < /dev/null & echo \$! > \"$remote_output_dir/launcher.pid\"'"
  log "remote run launched: $RUN_ID"
}

main() {
  mkdir -p "$(dirname "$LOG_FILE")"
  : > "$LOG_FILE"

  require_path "$TRAIN_MANIFEST"
  require_path "$DEV_MANIFEST"
  require_path "$TRAIN_AUDIO_DIR"
  require_path "$DEV_AUDIO_DIR"
  require_path "$LANG_DIR"

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
  log "syncing workspace and manifests"
  sync_workspace "$host" "$port" "$key_path"
  log "sync complete"
  log "syncing uv env on pod"
  prepare_remote_env "$host" "$port" "$key_path"
  launch_remote_run "$host" "$port" "$key_path"
}

main "$@"
