param(
    [Parameter(Mandatory = $false)]
    [string]$PythonCommand = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$VenvDir = Join-Path $Root '.venv'
$DownloadsDir = Join-Path $Root '_downloads'
$ReazonZip = Join-Path $DownloadsDir 'ReazonSpeech-master.zip'
$ReazonSourceDir = Join-Path $DownloadsDir 'ReazonSpeech-master'
$LogsDir = Join-Path $Root 'runs\setup_reazon_k2'
$LogFile = Join-Path $LogsDir 'setup.log'

New-Item -ItemType Directory -Force -Path $LogsDir, $DownloadsDir | Out-Null

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'), $Message
    $line | Tee-Object -FilePath $LogFile -Append
}

try {
    Write-Log "START setup_reazon_k2"
    Write-Log "Root=$Root"

    $pythonVersion = & $PythonCommand --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $PythonCommand"
    }
    Write-Log "Python=$pythonVersion"

    if (-not (Test-Path -LiteralPath $VenvDir)) {
        Write-Log "Creating venv: $VenvDir"
        & $PythonCommand -m venv $VenvDir 2>&1 | Tee-Object -FilePath $LogFile -Append
        if ($LASTEXITCODE -ne 0) {
            throw "venv creation failed."
        }
    }
    else {
        Write-Log "venv already exists: $VenvDir"
    }

    $VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        throw "venv python not found: $VenvPython"
    }

    Write-Log "Upgrading pip"
    & $VenvPython -m pip install --upgrade pip setuptools wheel 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed."
    }

    if (-not (Test-Path -LiteralPath $ReazonSourceDir -PathType Container)) {
        Write-Log "Downloading ReazonSpeech source zip"
        if (-not (Test-Path -LiteralPath $ReazonZip -PathType Leaf)) {
            curl.exe -L --fail --retry 3 --connect-timeout 20 -o $ReazonZip "https://github.com/reazon-research/ReazonSpeech/archive/refs/heads/master.zip" 2>&1 |
                Tee-Object -FilePath $LogFile -Append
            if ($LASTEXITCODE -ne 0) {
                throw "ReazonSpeech source download failed."
            }
        }

        Write-Log "Extracting ReazonSpeech source"
        Expand-Archive -LiteralPath $ReazonZip -DestinationPath $DownloadsDir -Force
    }

    $K2PackageDir = Join-Path $ReazonSourceDir 'pkg\k2-asr'
    if (-not (Test-Path -LiteralPath $K2PackageDir -PathType Container)) {
        throw "k2-asr package folder not found: $K2PackageDir"
    }

    Write-Log "Installing ReazonSpeech k2 package from official source checkout into local venv only"
    & $VenvPython -m pip install $K2PackageDir 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "ReazonSpeech pkg/k2-asr install failed. See $LogFile"
    }

    Write-Log "Verifying import"
    & $VenvPython -c "from reazonspeech.k2.asr import audio_from_path, load_model, transcribe; print('reazonspeech.k2.asr import OK')" 2>&1 |
        Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "ReazonSpeech import verification failed. See $LogFile"
    }

    Write-Log "END setup_reazon_k2 OK"
    Write-Host ""
    Write-Host "Setup complete."
    Write-Host "venv: $VenvDir"
    Write-Host "log:  $LogFile"
}
catch {
    Write-Log "FATAL $($_.Exception.Message)"
    Write-Error $_.Exception.Message
    exit 1
}
