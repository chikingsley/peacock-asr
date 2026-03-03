#!/usr/bin/env bash
set -euo pipefail

# Bootstrap peacock-asr on a RunPod instance.
#
# Usage:
#   bash scripts/runpod_bootstrap.sh
#
# Optional env vars:
#   PEACOCK_REPO_URL   (default: https://github.com/chikingsley/peacock-asr.git)
#   PEACOCK_REPO_DIR   (default: /runpod/peacock-asr)
#   PEACOCK_GIT_REF    (optional: branch/tag/sha to checkout)

REPO_URL="${PEACOCK_REPO_URL:-https://github.com/chikingsley/peacock-asr.git}"
REPO_DIR="${PEACOCK_REPO_DIR:-/runpod/peacock-asr}"
GIT_REF="${PEACOCK_GIT_REF:-}"

echo "[bootstrap] repo: ${REPO_URL}"
echo "[bootstrap] dir:  ${REPO_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "[bootstrap] error: git is required but not installed" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] error: uv is required but not installed" >&2
  exit 1
fi

if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "[bootstrap] updating existing checkout"
  git -C "${REPO_DIR}" fetch --all --tags --prune
  if [[ -n "${GIT_REF}" ]]; then
    git -C "${REPO_DIR}" checkout "${GIT_REF}"
  else
    current_branch="$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)"
    git -C "${REPO_DIR}" checkout "${current_branch}"
    git -C "${REPO_DIR}" pull --ff-only
  fi
else
  echo "[bootstrap] cloning fresh checkout"
  git clone "${REPO_URL}" "${REPO_DIR}"
  if [[ -n "${GIT_REF}" ]]; then
    git -C "${REPO_DIR}" checkout "${GIT_REF}"
  fi
fi

cd "${REPO_DIR}"
echo "[bootstrap] syncing dependencies with uv"
uv sync

echo
echo "[bootstrap] done"
echo "[bootstrap] next: export runtime env and run jobs"
cat <<'EOF'
export MLFLOW_TRACKING_URI="https://mlflow.peacockery.studio"
export MLFLOW_EXPERIMENT_NAME="peacock-asr"
export PEACOCK_ASR_HF_CHECKPOINT_REPO="Peacockery/peacock-asr-checkpoints"
export PEACOCK_ASR_HF_CHECKPOINT_UPLOAD="true"
export HF_TOKEN="<your_hf_token>"

# Example:
# uv run peacock-asr batch --config runs/track05_phase1_baseline.yaml
EOF
