#!/usr/bin/env bash
# Refresh the YouTube cookies file the curator's downloader uses (yt-dlp's official anti-bot fix).
#
# Reads cookies from Safari on home-mac over SSH/Tailscale and writes a Netscape cookies.txt into
# the target project's data/ dir (gitignored). curate `download` picks it up automatically when the
# file exists (see COOKIES in <lang>_asr/curate.py).
#
# ONE-TIME macOS GRANT REQUIRED: Safari's cookie store is TCC-protected, so over SSH you get
# "Operation not permitted: …Cookies.binarycookies" until you grant Full Disk Access to the SSH
# login. On the Mac: System Settings → Privacy & Security → Full Disk Access → '+' → ⌘⇧G →
# /usr/sbin/sshd → enable, then toggle Remote Login off/on. (Alternative that skips TCC entirely:
# export youtube.com cookies with the "Get cookies.txt LOCALLY" browser extension and drop the
# file straight at the dest path below.)
#
# Usage: scripts/refresh-youtube-cookies.sh [dest_cookies_file]
#        MAC_HOST / MAC_USER env override the host (default = Tailscale IP / simonpeacocks).
set -euo pipefail

MAC_USER="${MAC_USER:-simonpeacocks}"
MAC_HOST="${MAC_HOST:-$(tailscale ip -4 home-mac 2>/dev/null || echo 100.73.198.100)}"
DEST="${1:-projects/tajik-asr/data/youtube_cookies.txt}"
SSH=(ssh -o ConnectTimeout=10 -o BatchMode=yes -o HostKeyAlias=home-mac "${MAC_USER}@${MAC_HOST}")

echo "extracting Safari cookies on ${MAC_USER}@${MAC_HOST} ..."
"${SSH[@]}" '~/yt-dlp --cookies-from-browser safari --cookies ~/yt_cookies.txt --skip-download \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ" >/dev/null 2>&1 || true
  [ -s ~/yt_cookies.txt ] || { echo "FAILED: Safari cookies unreadable — grant Full Disk Access to sshd"; exit 1; }'

mkdir -p "$(dirname "$DEST")"
scp -o HostKeyAlias=home-mac "${MAC_USER}@${MAC_HOST}:~/yt_cookies.txt" "$DEST"
echo "wrote $DEST  ($(grep -c "youtube.com" "$DEST" 2>/dev/null || echo 0) youtube cookie lines)"
