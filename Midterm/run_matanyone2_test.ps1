$ErrorActionPreference = "Stop"

$midtermRoot = $PSScriptRoot
$repoRoot = Join-Path $midtermRoot "MatAnyone2"
$pythonExe = Join-Path $midtermRoot ".venv-matanyone2\Scripts\python.exe"
$caseRoot = Join-Path $midtermRoot "SOTA_Test_white_sequence_0160_0167"

$env:TORCH_HOME = Join-Path $midtermRoot "torch_cache"

Push-Location $repoRoot
try {
    & $pythonExe inference_matanyone2.py `
        -i (Join-Path $caseRoot "input") `
        -m (Join-Path $caseRoot "first_frame_mask.png") `
        -o (Join-Path $caseRoot "matanyone2_output") `
        -w 10 `
        -e 10 `
        -d 10 `
        --save_image
    if ($LASTEXITCODE -ne 0) {
        throw "MatAnyone 2 inference failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

& $pythonExe (Join-Path $midtermRoot "make_sota_comparison.py")
if ($LASTEXITCODE -ne 0) {
    throw "Comparison generation failed with exit code $LASTEXITCODE"
}
