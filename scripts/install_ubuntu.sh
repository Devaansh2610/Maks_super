#!/bin/bash
# One-time setup for Maks on Ubuntu: system packages, Python venv, and a
# check for Ollama. Run from the project root: bash scripts/install_ubuntu.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> Installing system packages (needs sudo)"
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-pip python3-dev \
  portaudio19-dev libasound2-dev \
  unzip curl build-essential

echo "==> Creating Python virtualenv"
python3 -m venv venv
./venv/bin/pip install --quiet --upgrade pip
./venv/bin/pip install --quiet -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example — edit it before first run"
  cp .env.example .env
else
  echo "==> .env already exists, leaving it as-is"
fi

# Ollama is only needed now for the local embedding model (the fast-path
# router) -- chat runs on Groq (cloud), so there's no chat model to pull.
if ! command -v ollama >/dev/null 2>&1; then
  echo ""
  echo "==> Ollama is not installed. Install it yourself with:"
  echo "    curl -fsSL https://ollama.com/install.sh | sh"
  echo "    (not run automatically here — review it first)"
  echo "    Then run: ollama pull all-minilm"
else
  MODEL=$(grep -E '^EMBEDDING_MODEL=' .env | cut -d= -f2- || echo "all-minilm")
  echo "==> Pulling Ollama embedding model: $MODEL"
  ollama pull "$MODEL"
fi

echo ""
echo "==> Next steps:"
echo "    1. Edit .env -- GROQ_API_KEY (free, console.groq.com), weather city,"
echo "       Fish Audio key, Mac companion URL/token, etc."
echo "    2. bash scripts/download_voice_models.sh"
echo "    3. Set up Google OAuth + Notion token — see README.md"
echo "    4. Run the Mac companion installer on your Mac (mac_companion/install/install.sh)"
echo "    5. ./venv/bin/python -m maks.main"
echo "       (or install scripts/maks.service to run it on boot)"
