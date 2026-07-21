Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSCommandPath
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Requirements = Join-Path $Root 'requirements_mvp.txt'
$Ffmpeg = Join-Path $Root 'engines\ffmpeg\ffmpeg.exe'
$Parakeet = Join-Path $Root 'models\sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8'

foreach ($required in @($Python, $Requirements, $Ffmpeg)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}

& $Python -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "MVP Python dependency installation failed. ExitCode=$LASTEXITCODE"
}

& $Python -c "import numpy, scipy, sounddevice, pyaudiowpatch, sherpa_onnx; import reazonspeech.k2.asr; print('LearningAccess audio and ASR modules: OK')"
if ($LASTEXITCODE -ne 0) {
    throw "LearningAccess module check failed. Run scripts\prototypes\setup_reazon_k2.ps1 for Japanese mode."
}

if (-not (Test-Path -LiteralPath (Join-Path $Parakeet 'encoder.int8.onnx'))) {
    Write-Warning "Parakeet model not found. English mode will be unavailable until the local model is installed: $Parakeet"
}

Write-Host "LearningAccess MVP setup complete."
Write-Host "Start with: .\run_otoweave.ps1"
