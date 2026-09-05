[CmdletBinding()]
param(
    [string]$Python = "",

    [switch]$Quick,

    [switch]$Full
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Benchmark = Join-Path $ProjectRoot "run_benchmark.py"

if (-not (Test-Path -LiteralPath $Benchmark -PathType Leaf)) {
    throw "Benchmark entry point not found: $Benchmark"
}

if ([string]::IsNullOrWhiteSpace($Python)) {
    $VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        $VenvPython
    } else {
        "python"
    }
}

function Invoke-BenchmarkStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Python -u $Benchmark @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

Push-Location $ProjectRoot
try {
    Invoke-BenchmarkStep -Name "K400 preflight" -Arguments @(
        "--preflight",
        "--device", "auto",
        "--dataset", "k400"
    )

    Invoke-BenchmarkStep -Name "K400 5-clip smoke test" -Arguments @(
        "--device", "auto",
        "--dataset", "k400",
        "--models", "uniformer-s",
        "--num-clips", "5",
        "--smoke-test",
        "--force"
    )

    if ($Quick) {
        Invoke-BenchmarkStep -Name "K400 100-clip quick benchmark" -Arguments @(
            "--device", "auto",
            "--dataset", "k400",
            "--num-clips", "100",
            "--quick",
            "--resume"
        )
    }

    if ($Full) {
        Invoke-BenchmarkStep -Name "K400 1000-clip resumable benchmark" -Arguments @(
            "--device", "auto",
            "--dataset", "k400",
            "--num-clips", "1000",
            "--resume"
        )
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Requested K400 validation sequence completed." -ForegroundColor Green
if (-not $Quick -and -not $Full) {
    Write-Host "Use -Quick, -Full, or both switches to continue beyond the smoke test."
}
