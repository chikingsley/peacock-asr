#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-simon@Spark}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/simon/work/peacock-asr/projects/P014-hippo-reproduction}"
POLL_SECONDS="${P014_PIPELINE_POLL_SECONDS:-30}"

GOP_DIR_LOCAL="$PROJECT_ROOT/.cache/p014/features/gop"
SSL_DIR_LOCAL="$PROJECT_ROOT/.cache/p014/features/ssl_utt"
GOP_DIR_REMOTE="$REMOTE_PROJECT_ROOT/.cache/p014/features/gop"
SSL_DIR_REMOTE="$REMOTE_PROJECT_ROOT/.cache/p014/features/ssl_utt"

LOCAL_GOP_TRAIN="$GOP_DIR_LOCAL/train__shard_000_of_002.pt"
LOCAL_GOP_TEST="$GOP_DIR_LOCAL/test__shard_000_of_002.pt"
REMOTE_GOP_TRAIN="$GOP_DIR_REMOTE/train__shard_001_of_002.pt"
REMOTE_GOP_TEST="$GOP_DIR_REMOTE/test__shard_001_of_002.pt"

LOCAL_SSL_TRAIN="$SSL_DIR_LOCAL/train.pt"
REMOTE_SSL_TEST="$SSL_DIR_REMOTE/test.pt"

LOCAL_SSL_SESSION="p014_ssl_train_local"
REMOTE_SSL_SESSION="p014_ssl_test_remote"
TRAIN_SESSION="p014_train_read_aloud"
RUN_NAME="hippo_read_aloud_paperfix_20260422"


timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}


log() {
  printf "[%s] %s\n" "$(timestamp)" "$*"
}


local_tmux_exists() {
  local session_name="$1"
  tmux has-session -t "$session_name" 2>/dev/null
}


remote_tmux_exists() {
  local session_name="$1"
  ssh "$REMOTE_HOST" "tmux has-session -t '$session_name' 2>/dev/null"
}


remote_file_exists() {
  local path="$1"
  ssh "$REMOTE_HOST" "test -f '$path'"
}


sync_remote_runtime_files() {
  log "syncing CLI and .env to Spark"
  rsync -a "$PROJECT_ROOT/src/p014/cli.py" "$REMOTE_HOST:$REMOTE_PROJECT_ROOT/src/p014/cli.py"
  rsync -a "$PROJECT_ROOT/.env" "$REMOTE_HOST:$REMOTE_PROJECT_ROOT/.env"
}


launch_local_ssl_train() {
  if local_tmux_exists "$LOCAL_SSL_SESSION"; then
    return 0
  fi
  log "launching local SSL train extraction"
  tmux new-session -d -s "$LOCAL_SSL_SESSION" \
    "cd '$PROJECT_ROOT' && set -a && . ./.env && set +a && . .venv/bin/activate && p014 extract-ssl --split train --cache-dir .cache/p014 --device cuda > artifacts/logs/ssl_train_local.log 2>&1"
}


launch_remote_ssl_test() {
  if remote_tmux_exists "$REMOTE_SSL_SESSION"; then
    return 0
  fi
  log "launching Spark SSL test extraction"
  ssh "$REMOTE_HOST" \
    "tmux new-session -d -s '$REMOTE_SSL_SESSION' \"cd '$REMOTE_PROJECT_ROOT' && set -a && . ./.env && set +a && . .venv/bin/activate && p014 extract-ssl --split test --cache-dir .cache/p014 --device cuda > artifacts/logs/ssl_test_remote.log 2>&1\""
}


sync_remote_gop_artifacts() {
  log "syncing Spark GOP shard artifacts to local cache"
  rsync -a "$REMOTE_HOST:$REMOTE_GOP_TRAIN" "$GOP_DIR_LOCAL/"
  rsync -a "$REMOTE_HOST:$REMOTE_GOP_TEST" "$GOP_DIR_LOCAL/"
}


merge_gop_shards_if_ready() {
  if [[ -f "$GOP_DIR_LOCAL/train.pt" && -f "$GOP_DIR_LOCAL/test.pt" ]]; then
    return 0
  fi
  if [[ -f "$LOCAL_GOP_TRAIN" && -f "$LOCAL_GOP_TEST" && -f "$GOP_DIR_LOCAL/train__shard_001_of_002.pt" && -f "$GOP_DIR_LOCAL/test__shard_001_of_002.pt" ]]; then
    log "merging local GOP shard caches"
    (
      cd "$PROJECT_ROOT"
      set -a
      . ./.env
      set +a
      . .venv/bin/activate
      p014 merge-gop-shards --split train --cache-dir .cache/p014 --num-shards 2
      p014 merge-gop-shards --split test --cache-dir .cache/p014 --num-shards 2
    )
  fi
}


sync_remote_ssl_test() {
  log "syncing Spark SSL test cache to local cache"
  rsync -a "$REMOTE_HOST:$REMOTE_SSL_TEST" "$SSL_DIR_LOCAL/"
  rsync -a "$REMOTE_HOST:$SSL_DIR_REMOTE/test.json" "$SSL_DIR_LOCAL/"
  rsync -a "$REMOTE_HOST:$SSL_DIR_REMOTE/test_ids.json" "$SSL_DIR_LOCAL/"
}


launch_read_aloud_training() {
  if local_tmux_exists "$TRAIN_SESSION"; then
    return 0
  fi
  log "launching read-aloud training"
  tmux new-session -d -s "$TRAIN_SESSION" \
    "cd '$PROJECT_ROOT' && set -a && . ./.env && set +a && . .venv/bin/activate && p014 train --scenario read_aloud --cache-dir .cache/p014 --device cuda --run-name '$RUN_NAME' > artifacts/logs/${RUN_NAME}.log 2>&1"
}


main() {
  local remote_gop_synced=0
  local ssl_test_synced=0
  local merged_gop=0

  log "starting read-aloud continuation pipeline"
  sync_remote_runtime_files

  while true; do
    if [[ -f "$LOCAL_GOP_TEST" ]]; then
      launch_local_ssl_train
    fi

    if remote_file_exists "$REMOTE_GOP_TEST"; then
      launch_remote_ssl_test
      if [[ "$remote_gop_synced" -eq 0 ]]; then
        sync_remote_gop_artifacts
        remote_gop_synced=1
      fi
    fi

    if [[ "$merged_gop" -eq 0 ]]; then
      merge_gop_shards_if_ready
      if [[ -f "$GOP_DIR_LOCAL/train.pt" && -f "$GOP_DIR_LOCAL/test.pt" ]]; then
        merged_gop=1
        log "merged GOP caches are ready"
      fi
    fi

    if remote_file_exists "$REMOTE_SSL_TEST"; then
      if [[ "$ssl_test_synced" -eq 0 ]]; then
        sync_remote_ssl_test
        ssl_test_synced=1
      fi
    fi

    if [[ -f "$LOCAL_SSL_TRAIN" && -f "$SSL_DIR_LOCAL/test.pt" && -f "$GOP_DIR_LOCAL/train.pt" && -f "$GOP_DIR_LOCAL/test.pt" ]]; then
      launch_read_aloud_training
      log "read-aloud training session launched: $TRAIN_SESSION"
      break
    fi

    log "waiting: local_gop_test=$( [[ -f "$LOCAL_GOP_TEST" ]] && echo yes || echo no ) remote_gop_test=$( remote_file_exists "$REMOTE_GOP_TEST" && echo yes || echo no ) local_ssl_train=$( [[ -f "$LOCAL_SSL_TRAIN" ]] && echo yes || echo no ) remote_ssl_test=$( remote_file_exists "$REMOTE_SSL_TEST" && echo yes || echo no )"
    sleep "$POLL_SECONDS"
  done
}


main "$@"
