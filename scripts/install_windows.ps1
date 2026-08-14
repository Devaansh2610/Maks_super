# One-time setup for Maks on Windows: Python venv, dependencies, Ollama check.
# Run from the project root in PowerShell: .\scripts\install_windows.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "==> Creating Python virtualenv"
if (-not (Test-Path ".\venv")) {
    python -m venv venv
}

Write-Host "==> Installing dependencies"
& ".\venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
& ".\venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt

if (-not (Test-Path ".\.env")) {
    Write-Host "==> Creating .env from .env.example -- edit it before first run"
    Copy-Item ".\.env.example" ".\.env"
} else {
    Write-Host "==> .env already exists, leaving it as-is"
}

# Ollama is only needed now for the local embedding model (the fast-path
# router) -- chat runs on Groq (cloud), so there's no chat model to pull.
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Host ""
    Write-Host "==> Ollama is not installed. Download and install it from:"
    Write-Host "    https://ollama.com/download/windows"
    Write-Host "    Then run: ollama pull all-minilm"
} else {
    $model = "all-minilm"
    $envLine = Select-String -Path ".\.env" -Pattern "^EMBEDDING_MODEL=" -ErrorAction SilentlyContinue
    if ($envLine) {
        $model = ($envLine.Line -split "=", 2)[1].Trim()
    }
    Write-Host "==> Pulling Ollama embedding model: $model"
    ollama pull $model
}

Write-Host ""
Write-Host "==> Next steps:"
Write-Host "    1. Edit .env -- GROQ_API_KEY (free, console.groq.com), weather city,"
Write-Host "       Fish Audio key, Mac companion URL/token, etc."
Write-Host "    2. .\scripts\download_voice_models.ps1"
Write-Host "    3. Set up Google OAuth + Notion token -- see README.md"
Write-Host "    4. (optional, later) Mac companion install -- see README.md"
Write-Host "    5. .\venv\Scripts\python.exe -m maks.main"
Write-Host "       (or .\scripts\install_task.ps1 to run it automatically at logon)"
