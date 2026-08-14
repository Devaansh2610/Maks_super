#!/bin/bash
# Downloads the free/offline voice asset Maks needs: a small Vosk model for
# wake-word spotting. (TTS runs through Fish Audio's cloud API now -- no
# local voice binary to download for that.)
# Run once from the project root: bash scripts/download_voice_models.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"
mkdir -p "$MODELS_DIR"

VOSK_MODEL_NAME="vosk-model-small-en-us-0.15"
if [ ! -d "$MODELS_DIR/$VOSK_MODEL_NAME" ]; then
  echo "==> Downloading Vosk wake-word model ($VOSK_MODEL_NAME, ~40MB)"
  curl -L -o "$MODELS_DIR/vosk.zip" "https://alphacephei.com/vosk/models/${VOSK_MODEL_NAME}.zip"
  unzip -q "$MODELS_DIR/vosk.zip" -d "$MODELS_DIR"
  rm "$MODELS_DIR/vosk.zip"
else
  echo "==> Vosk model already present, skipping"
fi

echo ""
echo "==> Done. Make sure your .env has:"
echo "    VOSK_MODEL_PATH=./models/${VOSK_MODEL_NAME}"
echo "    FISH_AUDIO_API_KEY / FISH_AUDIO_MODEL / FISH_AUDIO_VOICE_ID for TTS"
