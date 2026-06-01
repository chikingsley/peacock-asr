#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MFA_DIR="$PROJECT_DIR/.mfa"
MICROMAMBA_BIN="$MFA_DIR/bin/micromamba"
MICROMAMBA_ROOT="$MFA_DIR/micromamba-root"
MFA_ENV="$MFA_DIR/env"
MFA_ROOT="$MFA_DIR/root"

mkdir -p "$MFA_DIR/bin" "$MFA_ROOT"

if [ ! -x "$MICROMAMBA_BIN" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "$MFA_DIR/bin" bin/micromamba --strip-components=1
fi

if [ ! -x "$MFA_ENV/bin/mfa" ]; then
  MAMBA_ROOT_PREFIX="$MICROMAMBA_ROOT" "$MICROMAMBA_BIN" create -y \
    -p "$MFA_ENV" \
    -c conda-forge \
    montreal-forced-aligner
fi

MFA_ROOT_DIR="$MFA_ROOT" "$MFA_ENV/bin/mfa" model download g2p russian_mfa

printf 'MFA_BIN=%s\n' "$MFA_ENV/bin/mfa"
printf 'MFA_ROOT_DIR=%s\n' "$MFA_ROOT"
