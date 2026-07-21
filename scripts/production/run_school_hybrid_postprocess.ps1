param(
    [Parameter(Mandatory = $true)]
    [string]$TranscriptFile,

    [Parameter(Mandatory = $false)]
    [string]$ModelFile = "",

    [Parameter(Mandatory = $false)]
    [int]$NCtx = 8192,

    [Parameter(Mandatory = $false)]
    [int]$MaxCharsPerBatch = 6000,

    [Parameter(Mandatory = $false)]
    [int]$MaxBatches = 0,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokensPart = 1400,

    [Parameter(Mandatory = $false)]
    [int]$NThreads = 4,

    [Parameter(Mandatory = $false)]
    [int]$NBatch = 256,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokensSummary = 700,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokensFinal = 1800,

    [Parameter(Mandatory = $false)]
    [double]$Temperature = 0.0,

    [Parameter(Mandatory = $false)]
    [double]$TopP = 0.95,

    [Parameter(Mandatory = $false)]
    [int]$TopK = 40,

    [Parameter(Mandatory = $false)]
    [switch]$Stage1Only
    ,
    [Parameter(Mandatory = $false)]
    [ValidateSet('safe', 'llm')]
    [string]$CleanMode = 'safe'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$PyScript = Join-Path $Root 'scripts\production\school_hybrid_postprocess.py'
$PromptFile = Join-Path $Root 'prompts\school_hybrid_format.md'
$DefaultModel = Join-Path $Root 'models\Qwen3.5-4B-Q4_K_M.gguf'

if ($ModelFile -eq "") {
    $ModelFile = $DefaultModel
}

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$InputBase = [IO.Path]::GetFileNameWithoutExtension($TranscriptFile)
$RunDir = Join-Path $Root "runs\${Timestamp}_school_postprocess_${InputBase}"
$OutputDir = Join-Path $RunDir 'output'
$LogsDir = Join-Path $RunDir 'logs'
$LogFile = Join-Path $LogsDir 'log.txt'

function Assert-FileExists {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}

try {
    New-Item -ItemType Directory -Force -Path $OutputDir, $LogsDir | Out-Null
    Assert-FileExists -Path $TranscriptFile -Label 'Transcript'
    Assert-FileExists -Path $ModelFile -Label 'Model'
    Assert-FileExists -Path $VenvPython -Label '.venv Python'
    Assert-FileExists -Path $PyScript -Label 'school_hybrid_postprocess.py'
    Assert-FileExists -Path $PromptFile -Label 'school_hybrid_format.md'

    $argsList = @(
        $PyScript,
        '--input', $TranscriptFile,
        '--output_dir', $OutputDir,
        '--model', $ModelFile,
        '--log', $LogFile,
        '--prompt_file', $PromptFile,
        '--n_ctx', "$NCtx",
        '--max_chars_per_batch', "$MaxCharsPerBatch",
        '--max_batches', "$MaxBatches",
        '--max_tokens_part', "$MaxTokensPart",
        '--n_threads', "$NThreads",
        '--n_batch', "$NBatch",
        '--clean_mode', $CleanMode,
        '--max_tokens_summary', "$MaxTokensSummary",
        '--max_tokens_final', "$MaxTokensFinal",
        '--temperature', "$Temperature",
        '--top_p', "$TopP",
        '--top_k', "$TopK"
    )
    if ($Stage1Only) {
        $argsList += '--stage1_only'
    }

    & $VenvPython @argsList 2>&1 |
        Tee-Object -FilePath (Join-Path $LogsDir 'school_postprocess_stdout.txt') |
        Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "school_hybrid_postprocess.py failed. ExitCode=$LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "Done."
    Write-Host "Run folder    : $RunDir"
    if ($Stage1Only) {
        Write-Host "Safe transcript: $(Join-Path $OutputDir 'safe_transcript.md')"
        if ($CleanMode -eq 'llm') {
            Write-Host "AI readable   : $(Join-Path $OutputDir 'ai_readable_transcript.md')"
        }
        Write-Host "Legacy clean  : $(Join-Path $OutputDir 'clean_transcript.md')"
        Write-Host "Review flags  : $(Join-Path $OutputDir 'review_flags.md')"
    }
    else {
        Write-Host "School record : $(Join-Path $OutputDir 'school_record.md')"
        Write-Host "Part records  : $(Join-Path $OutputDir 'partial_records.md')"
        Write-Host "Part summaries: $(Join-Path $OutputDir 'part_summaries.md')"
    }
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
