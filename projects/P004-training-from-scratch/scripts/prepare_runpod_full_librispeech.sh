#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/simon/github/peacock-asr/projects/P004-training-from-scratch"
REMOTE_ROOT="/workspace/p004-training-from-scratch"
REMOTE_CACHE_DIR="/workspace/.cache/uv"
REMOTE_VENV="/workspace/venvs/p004-training-from-scratch"
LANG_DIR="$PROJECT_ROOT/experiments/data/lang_phone"
RUN_ID="prepare_runpod_full_librispeech_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_ROOT/experiments/logs/${RUN_ID}.log"

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

sync_builder_files() {
  local host="$1"
  local port="$2"
  local key_path="$3"
  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "mkdir -p '$REMOTE_ROOT/code' '$REMOTE_ROOT/experiments/data/manifests_phone_raw' '$REMOTE_ROOT/experiments/data/lang_phone' '/workspace/.cache/huggingface' '$REMOTE_CACHE_DIR' '$(dirname "$REMOTE_VENV")'"

  if [[ -f "$HOME/.netrc" ]]; then
    scp \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -i "$key_path" \
      -P "$port" \
      "$HOME/.netrc" \
      "root@${host}:/root/.netrc"
  fi

  rsync -a --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $key_path -p $port" \
    "$PROJECT_ROOT/code/build_librispeech_phone_cuts.py" \
    "$PROJECT_ROOT/code/merge_cut_manifests.py" \
    "root@${host}:$REMOTE_ROOT/code/"

  rsync -a --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $key_path -p $port" \
    "$LANG_DIR/" \
    "root@${host}:$REMOTE_ROOT/experiments/data/lang_phone/"
}

run_remote_prepare() {
  local host="$1"
  local port="$2"
  local key_path="$3"
  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "$key_path" \
    "root@${host}" \
    -p "$port" \
    "bash -lc '
set -euo pipefail
cd \"$REMOTE_ROOT\"
uv venv \"$REMOTE_VENV\" >/dev/null
export UV_CACHE_DIR=\"$REMOTE_CACHE_DIR\"
export VIRTUAL_ENV=\"$REMOTE_VENV\"
export PATH=\"$REMOTE_VENV/bin:\$PATH\"
export HF_HOME=/workspace/.cache/huggingface
if [[ ! -f experiments/data/manifests_phone_raw/librispeech_cuts_train-other-500.jsonl.gz ]]; then
  uv run --script code/build_librispeech_phone_cuts.py \
    --splits train_other_500 test_clean test_other \
    --output-dir experiments/data/manifests_phone_raw \
    --vocab-json experiments/data/lang_phone/phone_list.txt
fi
uv run --script code/merge_cut_manifests.py \
  --inputs \
    experiments/data/manifests_phone_raw/librispeech_cuts_train-clean-100.jsonl.gz \
    experiments/data/manifests_phone_raw/librispeech_cuts_train-clean-360.jsonl.gz \
    experiments/data/manifests_phone_raw/librispeech_cuts_train-other-500.jsonl.gz \
  --output experiments/data/manifests_phone_raw/librispeech_cuts_train-960.jsonl.gz
uv run --script code/merge_cut_manifests.py \
  --inputs \
    experiments/data/manifests_phone_raw/librispeech_cuts_dev-clean.jsonl.gz \
    experiments/data/manifests_phone_raw/librispeech_cuts_dev-other.jsonl.gz \
  --output experiments/data/manifests_phone_raw/librispeech_cuts_dev-all.jsonl.gz
uv run --script code/merge_cut_manifests.py \
  --inputs \
    experiments/data/manifests_phone_raw/librispeech_cuts_test-clean.jsonl.gz \
    experiments/data/manifests_phone_raw/librispeech_cuts_test-other.jsonl.gz \
  --output experiments/data/manifests_phone_raw/librispeech_cuts_test-all.jsonl.gz
'"
}

main() {
  mkdir -p "$(dirname "$LOG_FILE")"
  : > "$LOG_FILE"

  log "fetching ssh info for pod_id=$POD_ID"
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
  log "ssh endpoint ${host}:${port}"

  wait_for_ssh "$host" "$port" "$key_path"
  log "syncing builder and language assets"
  sync_builder_files "$host" "$port" "$key_path"
  log "starting remote full LibriSpeech preparation"
  run_remote_prepare "$host" "$port" "$key_path"
  log "remote full LibriSpeech preparation finished"
}

main "$@"
