param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [Parameter(Mandatory = $false)]
    [int]$MaxDurationSeconds = 0,

    [Parameter(Mandatory = $false)]
    [int]$StartSeconds = 0,

    [Parameter(Mandatory = $false)]
    [int]$MaxPatchChunks = 240,

    [Parameter(Mandatory = $false)]
    [int]$ShortTextChars = 16,

    [Parameter(Mandatory = $false)]
    [double]$MinCandidateSeconds = 2.0,

    [Parameter(Mandatory = $false)]
    [string]$PatchLanguage = "en"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$ReazonRunner = Join-Path $Root 'scripts\prototypes\run_reazon_k2_vad_test.ps1'
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$AlternatingScript = Join-Path $Root 'scripts\prototypes\ja_en_alternating_patch.py'

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$DurLabel = if ($MaxDurationSeconds -gt 0) { "${MaxDurationSeconds}sec" } else { 'full' }
if ($StartSeconds -gt 0) { $DurLabel = "${StartSeconds}s_${DurLabel}" }
$RunDir = Join-Path $Root "runs\${Timestamp}_ja_en_alternating_${DurLabel}"
$OutputDir = Join-Path $RunDir 'output'
$LogsDir = Join-Path $RunDir 'logs'
$ArtifactsDir = Join-Path $RunDir 'artifacts'
$LogFile = Join-Path $LogsDir 'log.txt'

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'), $Message
    $line | Out-File -FilePath $LogFile -Append -Encoding utf8
}

function Assert-FileExists {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}

function Get-NewestRun {
    param(
        [Parameter(Mandatory = $true)][datetime]$Since,
        [Parameter(Mandatory = $true)][string]$NamePattern
    )
    $run = Get-ChildItem -LiteralPath (Join-Path $Root 'runs') -Directory |
        Where-Object { $_.Name -like $NamePattern -and $_.LastWriteTime -ge $Since.AddSeconds(-2) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $run) {
        throw "Could not find generated run folder matching: $NamePattern"
    }
    return $run.FullName
}

try {
    New-Item -ItemType Directory -Force -Path $OutputDir, $LogsDir, $ArtifactsDir | Out-Null
    Assert-FileExists -Path $InputFile -Label 'Input audio'
    Assert-FileExists -Path $ReazonRunner -Label 'ReazonSpeech runner'
    Assert-FileExists -Path $VenvPython -Label '.venv Python'
    Assert-FileExists -Path $AlternatingScript -Label 'ja_en_alternating_patch.py'

    $started = Get-Date
    Write-Log "START run_pipeline_ja_en_alternating"
    Write-Log "InputFile=$InputFile"
    Write-Log "StartSeconds=$StartSeconds"
    Write-Log "MaxDurationSeconds=$MaxDurationSeconds MaxPatchChunks=$MaxPatchChunks ShortTextChars=$ShortTextChars MinCandidateSeconds=$MinCandidateSeconds PatchLanguage=$PatchLanguage"
    Write-Log "RunDir=$RunDir"

    $asrStart = Get-Date
    & $ReazonRunner -InputFile $InputFile -StartSeconds $StartSeconds -MaxDurationSeconds $MaxDurationSeconds 2>&1 |
        Tee-Object -FilePath (Join-Path $LogsDir 'reazon_runner_stdout.txt') |
        Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "ReazonSpeech ASR runner failed. ExitCode=$LASTEXITCODE"
    }
    $reazonRunDir = Get-NewestRun -Since $asrStart -NamePattern '*_reazon_k2_vad_*'
    Write-Log "ReazonRunDir=$reazonRunDir"

    $altArgs = @(
        $AlternatingScript,
        '--reazon_run_dir', $reazonRunDir,
        '--output_dir', $OutputDir,
        '--log', $LogFile,
        '--model', 'small',
        '--compute_type', 'int8',
        '--language', $PatchLanguage,
        '--short_text_chars', "$ShortTextChars",
        '--min_candidate_seconds', "$MinCandidateSeconds",
        '--max_patch_chunks', "$MaxPatchChunks"
    )

    & $VenvPython @altArgs 2>&1 |
        Tee-Object -FilePath (Join-Path $LogsDir 'alternating_patch_stdout.txt') |
        Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Japanese/English alternating patch failed. ExitCode=$LASTEXITCODE"
    }

    Copy-Item -LiteralPath (Join-Path $reazonRunDir 'output\full_transcript.txt') -Destination (Join-Path $OutputDir 'ja_base_full_transcript.txt') -Force
    Copy-Item -LiteralPath (Join-Path $reazonRunDir 'output\summary.json') -Destination (Join-Path $ArtifactsDir 'reazon_summary.json') -Force
    Copy-Item -LiteralPath (Join-Path $reazonRunDir 'chunks_vad\vad_splits.json') -Destination (Join-Path $ArtifactsDir 'vad_splits.json') -Force

    $summary = Get-Content -LiteralPath (Join-Path $OutputDir 'alternating_summary.json') -Raw | ConvertFrom-Json
    $totalSeconds = ((Get-Date) - $started).TotalSeconds
    Write-Log ('pipeline_seconds={0:N3}' -f $totalSeconds)
    Write-Log "END run_pipeline_ja_en_alternating OK"

    Write-Host ''
    Write-Host 'Done.'
    Write-Host "Run folder          : $RunDir"
    Write-Host "Reazon run          : $reazonRunDir"
    Write-Host "Candidates          : $($summary.candidates)"
    Write-Host "Patched chunks      : $($summary.selected_patch_chunks)"
    Write-Host "Patch ASR seconds   : $($summary.patch_asr_seconds)"
    Write-Host "Language counts     : $($summary.lang_counts | ConvertTo-Json -Compress)"
    Write-Host "Pair candidates     : $($summary.pair_candidates)"
    Write-Host "Pipeline seconds    : $([math]::Round($totalSeconds, 1))"
    Write-Host "Review transcript   : $(Join-Path $OutputDir 'alternating_review_transcript.txt')"
    Write-Host "Pairs               : $(Join-Path $OutputDir 'english_japanese_pairs.md')"
    Write-Host "Best effort         : $(Join-Path $OutputDir 'alternating_best_effort.txt')"
}
catch {
    if (Test-Path -LiteralPath $LogsDir) {
        Write-Log "FATAL $($_.Exception.Message)"
    }
    Write-Error $_.Exception.Message
    exit 1
}
