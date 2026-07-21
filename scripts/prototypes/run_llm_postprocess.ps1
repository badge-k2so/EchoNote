param(
    [Parameter(Mandatory = $true)]
    [string]$TranscriptFile,

    [Parameter(Mandatory = $false)]
    [string]$ModelFile = "",   # "" = auto-download Qwen2.5-1.5B-Instruct Q4_K_M

    [Parameter(Mandatory = $false)]
    [int]$NCtx = 4096,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokensFormat = 2048,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokensSummary = 512
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root       = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$PyScript   = Join-Path $Root 'scripts\prototypes\llm_postprocess.py'
$ModelsDir  = Join-Path $Root 'models'

$DefaultRepo     = 'Qwen/Qwen2.5-1.5B-Instruct-GGUF'
$DefaultFilename = 'qwen2.5-1.5b-instruct-q4_k_m.gguf'

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$InputBase = [IO.Path]::GetFileNameWithoutExtension($TranscriptFile)
$ResolvedModelFile = if ($ModelFile -eq "") { Join-Path $ModelsDir $DefaultFilename } else { $ModelFile }
$ModelLabel = [IO.Path]::GetFileNameWithoutExtension($ResolvedModelFile)
$ModelLabel = ($ModelLabel -replace '[^A-Za-z0-9._-]', '_').ToLowerInvariant()
$RunDir    = Join-Path $Root "runs\${Timestamp}_llm_${ModelLabel}_${InputBase}"
$OutputDir = Join-Path $RunDir 'output'
$LogsDir   = Join-Path $RunDir 'logs'
$LogFile   = Join-Path $LogsDir 'log.txt'

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'), $Message
    $line | Out-File -FilePath $LogFile -Append -Encoding utf8
}

function Assert-FileExists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Label not found: $Path" }
}

try {
    New-Item -ItemType Directory -Force -Path $OutputDir, $LogsDir | Out-Null

    Assert-FileExists -Path $TranscriptFile -Label 'Transcript'
    Assert-FileExists -Path $VenvPython     -Label '.venv Python'
    Assert-FileExists -Path $PyScript       -Label 'llm_postprocess.py'

    Write-Log "START run_llm_postprocess"
    Write-Log "TranscriptFile=$TranscriptFile"
    Write-Log "RunDir=$RunDir"

    # --- モデル解決 ---
    if ($ModelFile -eq "") {
        New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
        $ModelFile = $ResolvedModelFile

        if (-not (Test-Path -LiteralPath $ModelFile -PathType Leaf)) {
            Write-Host "Downloading $DefaultFilename from $DefaultRepo ..."
            Write-Log "model_download_start repo=$DefaultRepo file=$DefaultFilename"

            $dlScript = @"
from huggingface_hub import hf_hub_download
import sys
path = hf_hub_download(
    repo_id='$DefaultRepo',
    filename='$DefaultFilename',
    local_dir=r'$ModelsDir',
    local_dir_use_symlinks=False,
)
print(path)
"@
            $dlOut = & $VenvPython -c $dlScript 2>&1
            if ($LASTEXITCODE -ne 0) { throw "Model download failed: $dlOut" }
            Write-Log "model_download_done path=$ModelFile"
            Write-Host "Download complete: $ModelFile"
        } else {
            Write-Host "Model already cached: $ModelFile"
            Write-Log "model_cache_hit path=$ModelFile"
        }
    }

    Assert-FileExists -Path $ModelFile -Label 'GGUF model'
    Write-Log "ModelFile=$ModelFile"

    $totalStart = Get-Date

    # --- LLM実行 ---
    $pyArgs = @(
        $PyScript,
        '--input',              $TranscriptFile,
        '--output_dir',         $OutputDir,
        '--model',              $ModelFile,
        '--log',                $LogFile,
        '--n_ctx',              "$NCtx",
        '--max_tokens_format',  "$MaxTokensFormat",
        '--max_tokens_summary', "$MaxTokensSummary"
    )

    Write-Log "COMMAND python $($pyArgs -join ' ')"
    & $VenvPython @pyArgs 2>&1 | Tee-Object -FilePath (Join-Path $LogsDir 'llm_stdout.txt') | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "llm_postprocess.py failed. ExitCode=$LASTEXITCODE" }

    $totalSec = ((Get-Date) - $totalStart).TotalSeconds
    Write-Log ('total_seconds={0:N3}' -f $totalSec)
    Write-Log "END run_llm_postprocess OK"

    Write-Host ""
    Write-Host "Done."
    Write-Host "Run folder  : $RunDir"
    Write-Host "Formatted   : $(Join-Path $OutputDir 'formatted.txt')"
    Write-Host "Summary     : $(Join-Path $OutputDir 'summary.txt')"
    Write-Host "End-to-end  : $([math]::Round($totalSec,1)) sec"
}
catch {
    if (Test-Path -LiteralPath $LogsDir) { Write-Log "FATAL $($_.Exception.Message)" }
    Write-Error $_.Exception.Message
    exit 1
}
