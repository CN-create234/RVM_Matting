param(
    [Parameter(Mandatory=$true)]
    [string]$InputDir,

    [Parameter(Mandatory=$true)]
    [string]$OutputDir,

    [double]$KeypointThreshold = 0.25
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot "external\.venv-sapiens2-py310\Scripts\python.exe"
$poseRoot = Join-Path $projectRoot "external\sapiens2\sapiens\pose"
$detector = Join-Path $projectRoot "models\sapiens2\detector\detr-resnet-101-dc5"
$checkpoint = Join-Path $projectRoot "models\sapiens2\pose\sapiens2_0.4b_pose.safetensors"
$config = "configs\keypoints308\shutterstock_goliath_3po\sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"

foreach ($path in @($python, $poseRoot, $detector, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required Sapiens2 dependency is missing: $path"
    }
}

$inputPath = (Resolve-Path -LiteralPath $InputDir).Path
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$outputPath = (Resolve-Path -LiteralPath $OutputDir).Path

$env:PYTHONPATH = Join-Path $projectRoot "external\sapiens2"
Push-Location $poseRoot
try {
    & $python "tools\vis\vis_pose.py" `
        $detector `
        $config `
        $checkpoint `
        --input $inputPath `
        --output $outputPath `
        --kpt-thr $KeypointThreshold `
        --radius 4 `
        --thickness 4
    if ($LASTEXITCODE -ne 0) {
        throw "Sapiens2 pose inference failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
