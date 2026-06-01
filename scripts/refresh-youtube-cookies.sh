#!/usr/bin/env bash
# Refresh the YouTube cookies file the curator's downloader uses (yt-dlp's official anti-bot fix).
#
# Pulls cookies from Safari on home-mac and writes a Netscape cookies.txt into the target project's
# data/ dir (gitignored). curate `download` picks it up automatically when the file exists (see
# COOKIES in <lang>_asr/curate.py).
#
# HOW IT WORKS / why not plain SSH: Safari's cookie store is TCC-protected, so reading it must run
# in a GUI app's context. sshd does NOT have Full Disk Access (granting it doesn't propagate), but
# Terminal.app does. So we write a tiny extractor on the Mac and launch it with `open -a Terminal`,
# which runs the read in Terminal's context, then pull the resulting plain cookies.txt back over SSH
# (a regular file = no TCC). One-time setup: yt-dlp must be at ~/yt-dlp on the Mac, and Terminal.app
# needs Full Disk Access (System Settings → Privacy & Security → Full Disk Access → Terminal).
#
# Usage: scripts/refresh-youtube-cookies.sh [dest_cookies_file]
#        MAC_HOST / MAC_USER env override the host (default = Tailscale IP / simonpeacocks).
set -euo pipefail

MAC_USER="${MAC_USER:-simonpeacocks}"
MAC_HOST="${MAC_HOST:-$(tailscale ip -4 home-mac 2>/dev/null || echo 100.73.198.100)}"
DEST="${1:-projects/tajik-asr/data/youtube_cookies.txt}"
SSH=(ssh -o ConnectTimeout=10 -o BatchMode=yes -o HostKeyAlias=home-mac "${MAC_USER}@${MAC_HOST}")

echo "launching Safari-cookie extractor in Terminal on ${MAC_USER}@${MAC_HOST} ..."
"${SSH[@]}" 'cat > ~/extract_cookies.sh && chmod +x ~/extract_cookies.sh && open -a Terminal ~/extract_cookies.sh' <<'SCRIPT'
#!/bin/bash
rm -f ~/yt_cookies.txt ~/cookie_extract.log
~/yt-dlp --cookies-from-browser safari --cookies ~/yt_cookies.txt --skip-download \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ" > ~/cookie_extract.log 2>&1 || true
[ -s ~/yt_cookies.txt ] && echo "COOKIES_OK" >> ~/cookie_extract.log || echo "COOKIES_EMPTY" >> ~/cookie_extract.log
SCRIPT

echo "waiting for the extraction to finish ..."
"${SSH[@]}" 'for _ in $(seq 1 30); do grep -q "COOKIES_" ~/cookie_extract.log 2>/dev/null && break; sleep 2; done
  grep -q COOKIES_OK ~/cookie_extract.log 2>/dev/null || {
    echo "FAILED — grant Terminal.app Full Disk Access on the Mac, then retry"; tail -3 ~/cookie_extract.log; exit 1; }'

mkdir -p "$(dirname "$DEST")"
scp -o HostKeyAlias=home-mac "${MAC_USER}@${MAC_HOST}:~/yt_cookies.txt" "$DEST"
echo "wrote $DEST  ($(grep -c "youtube.com" "$DEST" 2>/dev/null || echo 0) youtube cookies)"
