$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

if (Test-Path "third_party\TimeSformer") {
    python -m pip install -e "third_party\TimeSformer"
}

Write-Host "Environment setup completed."