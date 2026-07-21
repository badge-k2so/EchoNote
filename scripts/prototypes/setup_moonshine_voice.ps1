param(
    [Parameter(Mandatory = $false)]
    [string]$PythonCommand = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$VenvDir = Join-Path $Root '.venv'
$LogsDir = Join-Path $Root 'runs\setup_moonshine_voice'
$LogFile = Join-Path $LogsDir 'setup.log'

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'), $Message
    $line | Tee-Object -FilePath $LogFile -Append
}

try {
    Write-Log "START setup_moonshine_voice"
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

    Write-Log "Installing moonshine-voice into local venv only"
    & $VenvPython -m pip install moonshine-voice 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "moonshine-voice install failed. See $LogFile"
    }

    Write-Log "Downloading/verifying Japanese model"
    & $VenvPython -m moonshine_voice.download --language ja 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Moonshine Japanese model download failed. See $LogFile"
    }

    Write-Log "Verifying import and model lookup"
    & $VenvPython -c "import moonshine_voice; from moonshine_voice import Transcriber; path, arch = moonshine_voice.get_model_for_language('ja'); print(path); print(arch)" 2>&1 |
        Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Moonshine import/model verification failed. See $LogFile"
    }

    Write-Log "END setup_moonshine_voice OK"
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
