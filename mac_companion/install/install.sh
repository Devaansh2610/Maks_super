#!/bin/bash
# Sets up the Maks Mac companion service to run in the background and start
# automatically on login, using launchd. Run this ON THE MAC:
#   cd mac_companion/install && ./install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLIST_TEMPLATE="$SCRIPT_DIR/com.maks.companion.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.maks.companion.plist"

echo "==> Setting up Maks companion in $WORKDIR"
cd "$WORKDIR"

if [ ! -d venv ]; then
  echo "==> Creating virtualenv"
  python3 -m venv venv
fi

echo "==> Installing dependencies"
./venv/bin/pip install --quiet --upgrade pip
./venv/bin/pip install --quiet -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Generating .env with a random shared token"
  TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(24))")
  echo "MAC_COMPANION_TOKEN=$TOKEN" > .env
  echo ""
  echo "    IMPORTANT: copy this line into the Ubuntu box's .env as well:"
  echo "    MAC_COMPANION_TOKEN=$TOKEN"
  echo ""
else
  echo "==> .env already exists, leaving it as-is"
fi

VENV_PYTHON="$WORKDIR/venv/bin/python"

echo "==> Writing launchd job"
sed \
  -e "s|__VENV_PYTHON__|$VENV_PYTHON|g" \
  -e "s|__WORKDIR__|$WORKDIR|g" \
  "$PLIST_TEMPLATE" > "$PLIST_DEST"

echo "==> Loading launchd job"
launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load -w "$PLIST_DEST"

echo ""
echo "==> Done. The companion service should now be running on port 8765."
echo "    Check it with:  curl http://localhost:8765/health"
echo "    Logs:            $WORKDIR/companion.log / companion.error.log"
echo "    On the Ubuntu box, set MAC_COMPANION_URL to this Mac's LAN IP, e.g.:"
echo "    MAC_COMPANION_URL=http://$(ipconfig getifaddr en0 2>/dev/null || echo '<mac-ip>'):8765"
