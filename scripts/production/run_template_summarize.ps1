param(
    [Parameter(Mandatory = $true)]
    [Alias('CleanTranscriptFile')]
    [string]$SafeTranscriptFile,

    [Parameter(Mandatory = $false)]
    [ValidateSet('meeting_record', 'support_record', 'lesson_record', 'self_reflection', 'meeting_memo', 'interview_record')]
    [string]$Template = 'meeting_record',

    [Parameter(Mandatory = $false)]
    [string]$OutputDir = '',

    [Parameter(Mandatory = $false)]
    [string]$ModelFile = '',

    [Parameter(Mandatory = $false)]
    [int]$NCtx = 8192,

    [Parameter(Mandatory = $false)]
    [int]$MaxCharsPerBatch = 7000,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokensPart = 1000,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokensFinal = 1800
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$PyScript = Join-Path $Root 'scripts\production\template_summarize.py'
$DefaultModel = Join-Path $Root 'models\Qwen3.5-4B-Q4_K_M.gguf'

if ($ModelFile -eq '') {
    $ModelFile = $DefaultModel
}

if ($OutputDir -eq '') {
    $parent = Split-Path -Parent $SafeTranscriptFile
    if ($parent) {
        $OutputDir = $parent
    }
    else {
        $Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $OutputDir = Join-Path $Root "runs\${Timestamp}_template_${Template}\output"
    }
}

$LogsDir = Join-Path $OutputDir 'logs'
$LogFile = Join-Path $LogsDir "template_${Template}.log"

function Assert-FileExists {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}

try {
    New-Item -ItemType Directory -Force -Path $OutputDir, $LogsDir | Out-Null
    Assert-FileExists -Path $SafeTranscriptFile -Label 'safe_transcript.md'
    Assert-FileExists -Path $ModelFile -Label 'Model'
    Assert-FileExists -Path $VenvPython -Label '.venv Python'
    Assert-FileExists -Path $PyScript -Label 'template_summarize.py'

    $argsList = @(
        $PyScript,
        '--safe_transcript', $SafeTranscriptFile,
        '--output_dir', $OutputDir,
        '--model', $ModelFile,
        '--log', $LogFile,
        '--template', $Template,
        '--n_ctx', "$NCtx",
        '--max_chars_per_batch', "$MaxCharsPerBatch",
        '--max_tokens_part', "$MaxTokensPart",
        '--max_tokens_final', "$MaxTokensFinal"
    )

    & $VenvPython @argsList 2>&1 |
        Tee-Object -FilePath (Join-Path $LogsDir "template_${Template}_stdout.txt") |
        Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "template_summarize.py failed. ExitCode=$LASTEXITCODE"
    }

    Write-Host ''
    Write-Host 'Done.'
    Write-Host "Template        : $Template"
    Write-Host "Output directory: $OutputDir"
    Write-Host "Summaries       : $(Join-Path $OutputDir 'summaries')"
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
