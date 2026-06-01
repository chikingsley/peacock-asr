#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p third_party artifacts

if [ ! -d third_party/zipa/.git ]; then
  git clone https://github.com/lingjzhu/zipa third_party/zipa
fi

if command -v hf >/dev/null 2>&1; then
  hf download anyspeech/zipa-large-crctc-ns-800k --local-dir artifacts/zipa-large-crctc-ns-800k
else
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "anyspeech/zipa-large-crctc-ns-800k",
    local_dir="artifacts/zipa-large-crctc-ns-800k",
)
PY
fi

echo "ZIPA repo: $ROOT/third_party/zipa"
echo "ZIPA model dir: $ROOT/artifacts/zipa-large-crctc-ns-800k"
