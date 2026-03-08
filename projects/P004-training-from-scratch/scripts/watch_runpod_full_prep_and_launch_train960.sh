#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/workspace/p004-training-from-scratch"
REMOTE_CACHE_DIR="$REMOTE_ROOT/.cache/uv"
REMOTE_VENV="/workspace/venvs/p004-training-from-scratch"
LOG_FILE="$PROJECT_ROOT/experiments/logs/watch_runpod_full_prep_and_launch_train960_$(date +%Y%m%d_%H%M%S).log"
RUN_ID="canonical_runpod_a100_train960_conformer_h512_l12_b16_e5_$(date +%Y%m%d_%H%M%S)"
REMOTE_OUTPUT_DIR="$REMOTE_ROOT/experiments/checkpoints/canonical_phone_ctc/$RUN_ID"
REQUIRED_MANIFESTS=(
  "librispeech_cuts_train-clean-100.jsonl.gz"
  "librispeech_cuts_train-clean-360.jsonl.gz"
  "librispeech_cuts_train-other-500.jsonl.gz"
  "librispeech_cuts_dev-clean.jsonl.gz"
  "librispeech_cuts_dev-other.jsonl.gz"
  "librispeech_cuts_test-clean.jsonl.gz"
  "librispeech_cuts_test-other.jsonl.gz"
)

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <runpod-pod-id>" >&2
  exit 1
fi

POD_ID="$1"

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

read_ssh_key_path() {
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

  log "timed out waiting for ssh readiness on ${host}:${port}"
  exit 1
}

fetch_connection() {
  local ssh_json host port key_path

  ssh_json="$(ssh_info_json)"
  host="$(printf '%s' "$ssh_json" | read_ssh_field ip)"
  port="$(printf '%s' "$ssh_json" | read_ssh_field port)"
  key_path="$(printf '%s' "$ssh_json" | read_ssh_key_path)"
  printf '%s %s %s\n' "$host" "$port" "$key_path"
}

remote_file_exists() {
  local host="$1"
  local port="$2"
  local key_path="$3"
  local path="$4"

  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "test -s '$path'"
}

wait_for_required_manifests() {
  local manifest
  log "waiting for full manifest set on pod_id=$POD_ID" >&2
  while true; do
    local connection host port key_path
    connection="$(fetch_connection)"
    host="$(awk '{print $1}' <<<"$connection")"
    port="$(awk '{print $2}' <<<"$connection")"
    key_path="$(awk '{print $3}' <<<"$connection")"
    local missing=0
    for manifest in "${REQUIRED_MANIFESTS[@]}"; do
      if ! remote_file_exists \
        "$host" \
        "$port" \
        "$key_path" \
        "$REMOTE_ROOT/experiments/data/manifests_phone_raw/$manifest"; then
        missing=1
        break
      fi
    done
    if [[ $missing -eq 0 ]]; then
      log "all required split manifests exist" >&2
      printf '%s\n' "$connection"
      return 0
    fi
    sleep 120
  done
}

sync_training_workspace() {
  local host="$1"
  local port="$2"
  local key_path="$3"

  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "mkdir -p '$REMOTE_ROOT/code' '$REMOTE_ROOT/src' '$REMOTE_ROOT/experiments/data/lang_phone' '$REMOTE_ROOT/experiments/checkpoints/canonical_phone_ctc' '$REMOTE_ROOT/experiments/logs' '$REMOTE_ROOT/resources' '$REMOTE_CACHE_DIR' '$(dirname "$REMOTE_VENV")'"

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
    "$PROJECT_ROOT/resources" \
    "root@${host}:$REMOTE_ROOT/"

  rsync -a --no-owner --no-group --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $key_path -p $port" \
    "$PROJECT_ROOT/code/merge_cut_manifests.py" \
    "root@${host}:$REMOTE_ROOT/code/"

  rsync -a --no-owner --no-group --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $key_path -p $port" \
    "$PROJECT_ROOT/experiments/data/lang_phone/" \
    "root@${host}:$REMOTE_ROOT/experiments/data/lang_phone/"
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

merge_full_manifests() {
  local host="$1"
  local port="$2"
  local key_path="$3"

  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "bash -lc 'cd \"$REMOTE_ROOT\" && export UV_CACHE_DIR=\"$REMOTE_CACHE_DIR\" VIRTUAL_ENV=\"$REMOTE_VENV\" PATH=\"$REMOTE_VENV/bin:\$PATH\" && P004_PROJECT_ROOT=\"$REMOTE_ROOT\" uv run --active --script code/merge_cut_manifests.py --inputs experiments/data/manifests_phone_raw/librispeech_cuts_train-clean-100.jsonl.gz experiments/data/manifests_phone_raw/librispeech_cuts_train-clean-360.jsonl.gz experiments/data/manifests_phone_raw/librispeech_cuts_train-other-500.jsonl.gz --output experiments/data/manifests_phone_raw/librispeech_cuts_train-960.jsonl.gz && P004_PROJECT_ROOT=\"$REMOTE_ROOT\" uv run --active --script code/merge_cut_manifests.py --inputs experiments/data/manifests_phone_raw/librispeech_cuts_dev-clean.jsonl.gz experiments/data/manifests_phone_raw/librispeech_cuts_dev-other.jsonl.gz --output experiments/data/manifests_phone_raw/librispeech_cuts_dev-all.jsonl.gz && P004_PROJECT_ROOT=\"$REMOTE_ROOT\" uv run --active --script code/merge_cut_manifests.py --inputs experiments/data/manifests_phone_raw/librispeech_cuts_test-clean.jsonl.gz experiments/data/manifests_phone_raw/librispeech_cuts_test-other.jsonl.gz --output experiments/data/manifests_phone_raw/librispeech_cuts_test-all.jsonl.gz'"
}

remote_train_already_started() {
  local host="$1"
  local port="$2"
  local key_path="$3"

  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "bash -lc 'ls \"$REMOTE_ROOT\"/experiments/checkpoints/canonical_phone_ctc/canonical_runpod_a100_train960_conformer_h512_l12_b16_e5_*/launcher.pid >/dev/null 2>&1 || pgrep -f \"p004-canonical-train.*librispeech_cuts_train-960.jsonl.gz\" >/dev/null 2>&1'"
}

launch_remote_train() {
  local host="$1"
  local port="$2"
  local key_path="$3"

  if remote_train_already_started "$host" "$port" "$key_path"; then
    log "train-960 launch already present on pod; skipping duplicate launch"
    return 0
  fi

  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "bash -lc 'cd \"$REMOTE_ROOT\" && mkdir -p \"$REMOTE_OUTPUT_DIR\" && export UV_CACHE_DIR=\"$REMOTE_CACHE_DIR\" VIRTUAL_ENV=\"$REMOTE_VENV\" PATH=\"$REMOTE_VENV/bin:\$PATH\" && nohup env P004_PROJECT_ROOT=\"$REMOTE_ROOT\" uv run --active p004-canonical-train --run-id \"$RUN_ID\" --train-manifest experiments/data/manifests_phone_raw/librispeech_cuts_train-960.jsonl.gz --dev-manifest experiments/data/manifests_phone_raw/librispeech_cuts_dev-all.jsonl.gz --train-limit 0 --dev-limit 0 --epochs 5 --batch-size 16 --hidden-dim 512 --encoder-layers 12 --attention-heads 8 > \"$REMOTE_OUTPUT_DIR/launcher.stdout.log\" 2>&1 < /dev/null & echo \$! > \"$REMOTE_OUTPUT_DIR/launcher.pid\"'"

  log "launched remote train-960 run_id=$RUN_ID"
  log "remote output dir: $REMOTE_OUTPUT_DIR"
}

main() {
  mkdir -p "$(dirname "$LOG_FILE")"
  : > "$LOG_FILE"

  log "starting full prep -> train-960 watcher for pod_id=$POD_ID"

  local connection host port key_path
  connection="$(fetch_connection)"
  host="$(awk '{print $1}' <<<"$connection")"
  port="$(awk '{print $2}' <<<"$connection")"
  key_path="$(awk '{print $3}' <<<"$connection")"
  log "ssh endpoint ${host}:${port}"

  wait_for_ssh "$host" "$port" "$key_path"
  connection="$(wait_for_required_manifests)"
  host="$(awk '{print $1}' <<<"$connection")"
  port="$(awk '{print $2}' <<<"$connection")"
  key_path="$(awk '{print $3}' <<<"$connection")"

  log "syncing training workspace to pod"
  sync_training_workspace "$host" "$port" "$key_path"
  log "sync complete"

  log "syncing uv environment on pod"
  prepare_remote_env "$host" "$port" "$key_path"
  log "uv environment ready"

  if remote_file_exists "$host" "$port" "$key_path" "$REMOTE_ROOT/experiments/data/manifests_phone_raw/librispeech_cuts_train-960.jsonl.gz" \
    && remote_file_exists "$host" "$port" "$key_path" "$REMOTE_ROOT/experiments/data/manifests_phone_raw/librispeech_cuts_dev-all.jsonl.gz" \
    && remote_file_exists "$host" "$port" "$key_path" "$REMOTE_ROOT/experiments/data/manifests_phone_raw/librispeech_cuts_test-all.jsonl.gz"; then
    log "merged full manifests already exist; skipping merge"
  else
    log "merging full manifests on pod"
    merge_full_manifests "$host" "$port" "$key_path"
    log "merge complete"
  fi

  launch_remote_train "$host" "$port" "$key_path"
}

main "$@"
