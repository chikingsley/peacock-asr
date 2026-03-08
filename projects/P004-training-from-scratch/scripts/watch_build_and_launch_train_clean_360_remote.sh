#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/simon/github/peacock-asr/projects/P004-training-from-scratch"
BUILD_PID_FILE="$PROJECT_ROOT/experiments/logs/build_train_clean_360_dev_other_20260307.pid"
BUILD_LOG_FILE="$PROJECT_ROOT/experiments/logs/build_train_clean_360_dev_other_20260307.log"
WATCH_LOG_FILE="$PROJECT_ROOT/experiments/logs/watch_build_and_launch_train_clean_360_remote_20260307.log"

TRAIN_MANIFEST="$PROJECT_ROOT/experiments/data/manifests_phone_raw/librispeech_cuts_train-clean-360.jsonl.gz"
DEV_MANIFEST="$PROJECT_ROOT/experiments/data/manifests_phone_raw/librispeech_cuts_dev-other.jsonl.gz"
TOKENS_PATH="$PROJECT_ROOT/experiments/data/lang_phone/tokens.txt"
TRAIN_AUDIO_DIR="$PROJECT_ROOT/experiments/data/manifests_phone_raw/audio/train_clean_360"
DEV_AUDIO_DIR="$PROJECT_ROOT/experiments/data/manifests_phone_raw/audio/dev_other"

GPU_NAME="RTX PRO 6000 S"
IMAGE="nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"
DISK_GB="300"
INSTANCE_LABEL="p004-canonical-trainclean360"
RUN_ID="canonical_remote_rtxpro6000_trainclean360_conformer_h512_l12_b16_e5_$(date +%Y%m%d_%H%M%S)"
REMOTE_ROOT="/home/simon/github/peacock-asr/projects/P004-training-from-scratch"
REMOTE_OUTPUT_DIR="experiments/checkpoints/canonical_phone_ctc/${RUN_ID}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$WATCH_LOG_FILE"
}

require_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    log "missing required file: $path"
    exit 1
  fi
}

wait_for_build() {
  require_file "$BUILD_PID_FILE"
  local build_pid
  build_pid="$(cat "$BUILD_PID_FILE")"
  log "waiting for manifest build pid=$build_pid"
  while kill -0 "$build_pid" 2>/dev/null; do
    sleep 30
  done
  log "manifest build process exited"
}

validate_manifests() {
  require_file "$TRAIN_MANIFEST"
  require_file "$DEV_MANIFEST"
  require_file "$TOKENS_PATH"
  if [[ ! -d "$TRAIN_AUDIO_DIR" ]]; then
    log "missing train audio dir: $TRAIN_AUDIO_DIR"
    exit 1
  fi
  if [[ ! -d "$DEV_AUDIO_DIR" ]]; then
    log "missing dev audio dir: $DEV_AUDIO_DIR"
    exit 1
  fi
  log "validated new manifests and audio dirs"
}

launch_instance() {
  local launch_json
  launch_json="$(mktemp)"
  if ! (
    cd "$PROJECT_ROOT"
    uv run p004-vast-launch-instance \
      --gpu-name "$GPU_NAME" \
      --image "$IMAGE" \
      --num-gpus 1 \
      --disk-gb "$DISK_GB" \
      --limit 3 \
      --order dph \
      --label "$INSTANCE_LABEL" \
      --query-clause 'reliability > 0.98' \
      --onstart-cmd 'bash -lc "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl rsync git build-essential && curl -LsSf https://astral.sh/uv/install.sh | sh"' \
      > "$launch_json"
  ); then
    log "vast launch command failed"
  fi
  local launch_status=0
  local launch_output
  if ! launch_output="$(python - <<'PY' "$launch_json"
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
if payload.get("success") and payload.get("instance_id") is not None:
    print(payload["instance_id"])
    raise SystemExit(0)
error = payload.get("error") or "unknown_error"
message = payload.get("message") or "launch failed"
print(f"{error}: {message}", file=sys.stderr)
raise SystemExit(1)
PY
  )"; then
    launch_status=$?
  fi
  if [[ $launch_status -ne 0 ]]; then
    log "launch failed: $(<"$launch_json")"
    rm -f "$launch_json"
    exit 1
  fi
  local instance_id="$launch_output"
  rm -f "$launch_json"
  echo "$instance_id"
}

poll_ssh_connection() {
  local instance_id="$1"
  local attempt
  for attempt in $(seq 1 60); do
    local connection
    connection="$(
      cd "$PROJECT_ROOT" && uv run python - <<'PY' "$instance_id"
import sys
from p004_training_from_scratch.vast import VastClient

instance_id = int(sys.argv[1])
client = VastClient.from_env()
for instance in client.show_instances():
    if instance.instance_id == instance_id and instance.ssh_connection is not None:
        conn = instance.ssh_connection
        print(f"{conn.host} {conn.port}")
        raise SystemExit(0)
raise SystemExit(1)
PY
    )" || true
    if [[ -n "$connection" ]]; then
      echo "$connection"
      return 0
    fi
    sleep 10
  done
  log "timed out waiting for ssh details for instance_id=$instance_id"
  exit 1
}

wait_for_ssh_ready() {
  local ssh_host="$1"
  local ssh_port="$2"
  local attempt
  for attempt in $(seq 1 60); do
    if ssh -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=5 \
      -p "$ssh_port" \
      "root@${ssh_host}" \
      'echo ready' >/dev/null 2>&1; then
      log "ssh is ready on ${ssh_host}:${ssh_port}"
      return 0
    fi
    sleep 10
  done
  log "timed out waiting for ssh readiness on ${ssh_host}:${ssh_port}"
  exit 1
}

sync_remote_workspace() {
  local ssh_host="$1"
  local ssh_port="$2"
  ssh -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -p "$ssh_port" \
    "root@${ssh_host}" \
    "mkdir -p '$REMOTE_ROOT/experiments/data/manifests_phone_raw/audio' '$REMOTE_ROOT/experiments/data' '$REMOTE_ROOT/experiments/checkpoints/canonical_phone_ctc'"

  scp -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -P "$ssh_port" \
    "$HOME/.netrc" \
    "root@${ssh_host}:/root/.netrc"

  rsync -az --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $ssh_port" \
    "$PROJECT_ROOT/README.md" \
    "$PROJECT_ROOT/pyproject.toml" \
    "$PROJECT_ROOT/uv.lock" \
    "$PROJECT_ROOT/.python-version" \
    "$PROJECT_ROOT/src" \
    "root@${ssh_host}:$REMOTE_ROOT/"

  rsync -az --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $ssh_port" \
    "$PROJECT_ROOT/experiments/data/lang_phone/" \
    "root@${ssh_host}:$REMOTE_ROOT/experiments/data/lang_phone/"

  rsync -az --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $ssh_port" \
    "$PROJECT_ROOT/experiments/data/manifests_phone_raw/librispeech_cuts_train-clean-360.jsonl.gz" \
    "$PROJECT_ROOT/experiments/data/manifests_phone_raw/librispeech_cuts_dev-other.jsonl.gz" \
    "root@${ssh_host}:$REMOTE_ROOT/experiments/data/manifests_phone_raw/"

  rsync -az --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $ssh_port" \
    "$TRAIN_AUDIO_DIR/" \
    "root@${ssh_host}:$REMOTE_ROOT/experiments/data/manifests_phone_raw/audio/train_clean_360/"

  rsync -az --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $ssh_port" \
    "$DEV_AUDIO_DIR/" \
    "root@${ssh_host}:$REMOTE_ROOT/experiments/data/manifests_phone_raw/audio/dev_other/"
}

prepare_remote_env() {
  local ssh_host="$1"
  local ssh_port="$2"
  ssh -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -p "$ssh_port" \
    "root@${ssh_host}" \
    "cd '$REMOTE_ROOT' && ~/.local/bin/uv sync --group dev --group canonical"
}

launch_remote_run() {
  local ssh_host="$1"
  local ssh_port="$2"
  ssh -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -p "$ssh_port" \
    "root@${ssh_host}" \
    "bash -lc 'cd \"$REMOTE_ROOT\" && mkdir -p \"$REMOTE_OUTPUT_DIR\" && nohup ~/.local/bin/uv run p004-canonical-train --run-id \"$RUN_ID\" --train-manifest experiments/data/manifests_phone_raw/librispeech_cuts_train-clean-360.jsonl.gz --dev-manifest experiments/data/manifests_phone_raw/librispeech_cuts_dev-other.jsonl.gz --train-limit 0 --dev-limit 0 --epochs 5 --batch-size 16 --hidden-dim 512 --encoder-layers 12 --attention-heads 8 > \"$REMOTE_OUTPUT_DIR/launcher.stdout.log\" 2>&1 < /dev/null & echo \$! > \"$REMOTE_OUTPUT_DIR/launcher.pid\"'"
}

main() {
  mkdir -p "$(dirname "$WATCH_LOG_FILE")"
  : > "$WATCH_LOG_FILE"
  log "watcher started"
  wait_for_build
  validate_manifests

  local instance_id
  instance_id="$(launch_instance)"
  log "launched instance_id=$instance_id"

  local connection ssh_host ssh_port
  connection="$(poll_ssh_connection "$instance_id")"
  ssh_host="$(awk '{print $1}' <<<"$connection")"
  ssh_port="$(awk '{print $2}' <<<"$connection")"
  log "ssh connection ${ssh_host}:${ssh_port}"

  wait_for_ssh_ready "$ssh_host" "$ssh_port"
  sync_remote_workspace "$ssh_host" "$ssh_port"
  prepare_remote_env "$ssh_host" "$ssh_port"
  launch_remote_run "$ssh_host" "$ssh_port"
  log "remote run launched: $RUN_ID on instance_id=$instance_id"
}

main "$@"
