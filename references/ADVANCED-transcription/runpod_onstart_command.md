bash -lc '
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1        # optional: faster model pulls
export PIP_ROOT_USER_ACTION=ignore        # hide “pip as root” warning

# ── 1 · Update system packages and install nano ─────────────────────────────
apt-get update && apt-get install -y nano

# ── 2 · Configure git with environment variables ─────────────────────────────
if [ -n "${GIT_USER_NAME:-}" ]; then
    git config --global user.name "$GIT_USER_NAME"
fi
if [ -n "${GIT_USER_EMAIL:-}" ]; then
    git config --global user.email "$GIT_USER_EMAIL"
fi

# ── 3 · Bring your project up-to-date ────────────────────────────────────────
cd /workspace
if [ -d ADVANCED-transcription/.git ]; then
    git -C ADVANCED-transcription pull --ff-only
else
    git clone https://$GITHUB_PAT@github.com/TrelisResearch/ADVANCED-transcription.git ADVANCED-transcription
fi
cd ADVANCED-transcription

# ── 4 · Register a custom kernel in the CURRENT Python env ───────────────────
python -m ipykernel install --name ADVANCED-transcription \
                            --display-name "ADVANCED-transcription" \
                            --sys-prefix

# ── 5 · Hand off to RunPod’s standard entrypoint ────────────────────────────
exec /start.sh
'
