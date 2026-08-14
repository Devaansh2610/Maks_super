# Downloads the free/offline voice asset Maks needs: a small Vosk model for
# wake-word spotting. (TTS runs through Fish Audio's cloud API now -- no
# local voice binary to download for that.)
# Run once from the project root: .\scripts\download_voice_models.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$ModelsDir = ".\models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$VoskModelName = "vosk-model-small-en-us-0.15"
$VoskModelPath = Join-Path $ModelsDir $VoskModelName
if (-not (Test-Path $VoskModelPath)) {
    Write-Host "==> Downloading Vosk wake-word model ($VoskModelName, ~40MB)"
    $voskZip = Join-Path $ModelsDir "vosk.zip"
    Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/$VoskModelName.zip" -OutFile $voskZip
    Expand-Archive -Path $voskZip -DestinationPath $ModelsDir -Force
    Remove-Item $voskZip
} else {
    Write-Host "==> Vosk model already present, skipping"
}

Write-Host ""
Write-Host "==> Done. Make sure your .env has:"
Write-Host "    VOSK_MODEL_PATH=./models/$VoskModelName"
Write-Host "    FISH_AUDIO_API_KEY / FISH_AUDIO_MODEL / FISH_AUDIO_VOICE_ID for TTS"
