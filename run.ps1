# Start the AI Receptionist. Creates the venv and installs deps on first run.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env - add your OPENAI_API_KEY, then run this again." -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting on http://127.0.0.1:8000 ..." -ForegroundColor Green
& .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
