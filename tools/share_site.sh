#!/usr/bin/env bash
# Share the live website with the professor (option 3 of "1 + 3", 13 Nov 2026).
#
#   bash tools/share_site.sh            # Ctrl+C stops everything
#
# Starts `mmpquant serve --shared` (her uploads go to data/shared_uploads/ and
# runs/_shared/, never into the corpus), opens a free Cloudflare quick tunnel to
# it, and keeps the Mac awake while it runs. The link is printed below; it is
# NEW every time this script starts, so send her the fresh one.
#
# The Hugging Face page's "Mark your own video" button follows this link: it is
# published to the Space (live.json) on start and cleared on stop, so the
# professors only ever need the Hugging Face link (user's request, 13 Nov 2026).
#
# --protocol http2: this network blocks the tunnel's default UDP (QUIC) route.
# Only share the link AFTER her ratings are back (user, 12 Nov 2026).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"

if ! command -v cloudflared >/dev/null; then
  echo "cloudflared is not installed: brew install cloudflared" >&2
  exit 1
fi

.venv/bin/mmpquant serve --shared --port "$PORT" &
SERVER=$!
caffeinate -dims -w "$SERVER" &
LOG="$(mktemp -t mmpquant-tunnel)"
publish() {  # point (or clear) the Hugging Face button; never fatal
  .venv/bin/python tools/publish_live_link.py "$1" \
    || echo "  (could not update the Hugging Face button -- is \`hf auth login\` done?)"
}
cleanup() {
  kill "$SERVER" "${TUNNEL:-}" 2>/dev/null || true
  if [ -n "${URL:-}" ]; then publish ""; fi
}
trap cleanup EXIT INT TERM

sleep 3
cloudflared tunnel --no-autoupdate --protocol http2 --url "http://127.0.0.1:$PORT" >"$LOG" 2>&1 &
TUNNEL=$!

for _ in $(seq 1 30); do
  URL="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1 || true)"
  [ -n "$URL" ] && break
  sleep 2
done
if [ -z "${URL:-}" ]; then
  echo "The tunnel did not start; its log: $LOG" >&2
  exit 1
fi

publish "$URL"
echo
echo "  Live site for the professor:  $URL"
echo "  Hugging Face button updated:  https://huggingface.co/spaces/Frankyharsh08/mmp-quant"
echo "  (a new link each time this starts; keep this window open; Ctrl+C to stop)"
echo
wait "$SERVER"
