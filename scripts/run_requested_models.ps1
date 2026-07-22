[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [Alias("VideosDir")]
    [string]$DatasetDir,

    [string]$OutDir = "results_requested_models",

    [ValidateRange(1, [int]::MaxValue)]
    [int]$MaxClips = 1000,

    [ValidateRange(1, [int]::MaxValue)]
    [int]$Repeats = 1,

    [ValidateRange(1, [int]::MaxValue)]
    [int]$Frames = 32,

    [int]$Seed = 0,

    [ValidateSet("auto", "nvml", "nvidia-smi", "tegrastats", "none")]
    [string]$Power = "auto",

    [string]$DeviceName = "",

    [ValidateSet("k400", "ssv2")]
    [string]$Dataset = "k400",

    [string]$TorchDevice = "auto",

    [string]$Annotations = "",

    [string]$SplitName = "",

    [string]$Python = "",

    [switch]$NoCache,

    [switch]$AllowMissing
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Preflight = Join-Path $ProjectRoot "check_dataset_models.py"
$Benchmark = Join-Path $ProjectRoot "run_benchmark.py"

if ([string]::IsNullOrWhiteSpace($Python)) {
    $VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        $VenvPython
    } else {
        "python"
    }
}

if (-not (Test-Path -LiteralPath $DatasetDir -PathType Container)) {
    throw "Dataset directory does not exist: $DatasetDir"
}
$DatasetPath = (Resolve-Path -LiteralPath $DatasetDir).Path

$PreflightArgs = @($Preflight, "--dataset", $Dataset)
if ($AllowMissing) {
    $PreflightArgs += "--allow-missing"
}

$K400Models = @(
    "uniformer-b",
    "uniformer-s",
    "mvit-b-24-32x3",
    "mvit-v1-b",
    "videoswin-t",
    "videoswin-s",
    "videoswin-b",
    "timesformer-b",
    "video-focalnet-t",
    "video-focalnet-s",
    "video-focalnet-b",
    "dualformer-t",
    "omnivore-b",
    "videomae-b",
    "vtn-b"
)
$SSV2Models = @(
    "uniformer-b",
    "uniformer-s",
    "videoswin-b",
    "timesformer-b",
    "video-focalnet-b",
    "videomae-b"
)

Write-Host "Running $Dataset exact supervised-classifier asset preflight..."
& $Python @PreflightArgs
if ($LASTEXITCODE -ne 0) {
    throw "Dataset-model preflight failed. Supply the exact assets or rerun with -AllowMissing."
}
$Models = if ($Dataset -eq "k400") { $K400Models } else { $SSV2Models }

if ([System.IO.Path]::IsPathRooted($OutDir)) {
    $OutputPath = [System.IO.Path]::GetFullPath($OutDir)
} else {
    $OutputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutDir))
}
[void](New-Item -ItemType Directory -Path $OutputPath -Force)

$BenchmarkArgs = @(
    $Benchmark,
    "--videos-dir", $DatasetPath,
    "--max-clips", $MaxClips.ToString(),
    "--repeats", $Repeats.ToString(),
    "--frames", $Frames.ToString(),
    "--seed", $Seed.ToString(),
    "--power", $Power,
    "--dataset", $Dataset,
    "--device", $TorchDevice,
    "--output", "device_metrics.csv",
    "--models"
) + $Models

if ($AllowMissing) {
    $BenchmarkArgs += "--allow-skips"
}

if ($NoCache) {
    $BenchmarkArgs += "--no-cache"
}

if (-not [string]::IsNullOrWhiteSpace($DeviceName)) {
    $BenchmarkArgs += @("--device-name", $DeviceName)
}
if (-not [string]::IsNullOrWhiteSpace($Annotations)) {
    $BenchmarkArgs += @("--annotations", (Resolve-Path -LiteralPath $Annotations).Path)
}
if (-not [string]::IsNullOrWhiteSpace($SplitName)) {
    $BenchmarkArgs += @("--split-name", $SplitName)
}

Write-Host "Benchmarking $($Models.Count) requested registry keys for $Dataset."
Write-Host "Results directory: $OutputPath"
Push-Location $OutputPath
try {
    # Unbuffered output keeps long GPU runs observable model by model.
    & $Python -u @BenchmarkArgs
    if ($LASTEXITCODE -ne 0) {
        throw "run_benchmark.py failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
