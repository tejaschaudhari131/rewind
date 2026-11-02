#!/usr/bin/env bash
# Serve REWIND to a phone (or laptop webcam) over HTTPS via a Cloudflare quick tunnel, behind an access token.
# Usage: scripts/serve_phone.sh [port]   -> prints the URL; the token lives in .phone_token (chmod 600, gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8077}"
CF="$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")"
if [ ! -x "$CF" ]; then
  mkdir -p "$HOME/.local/bin"
  curl -sSL -o "$HOME/.local/bin/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$HOME/.local/bin/cloudflared"; CF="$HOME/.local/bin/cloudflared"
fi
if [ ! -s .phone_token ]; then (umask 077; python3 -c 'import secrets;print(secrets.token_urlsafe(18))' > .phone_token); fi
chmod 600 .phone_token
TOKEN="$(cat .phone_token)"
[ -d frontend/dist ] || (cd frontend && npm run build)
mkdir -p .run
pkill -f "uvicorn backend.app.main:app --host 127.0.0.1 --port $PORT" 2>/dev/null || true
pkill -f "cloudflared tunnel --url http://127.0.0.1:$PORT" 2>/dev/null || true
REWIND_ACCESS_TOKEN="$TOKEN" nohup python3 -m uvicorn backend.app.main:app --host 127.0.0.1 --port "$PORT" > .run/server.log 2>&1 &
# --protocol http2 works through restrictive egress where QUIC/UDP is blocked
nohup "$CF" tunnel --no-autoupdate --protocol http2 --url "http://127.0.0.1:$PORT" > .run/tunnel.log 2>&1 &
for _ in $(seq 1 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' .run/tunnel.log | head -1 || true)"
  [ -n "$URL" ] && break; sleep 1
done
if [ -z "${URL:-}" ]; then echo "tunnel did not come up; see .run/tunnel.log" >&2; exit 1; fi
echo "$URL" > .run/phone_url
echo "REWIND phone URL: $URL"
echo "Open:  $URL/?t=<token from .phone_token>    (camera view: add &look=b&record=1&source=camera)"
echo "Logs:  .run/server.log  .run/tunnel.log"
