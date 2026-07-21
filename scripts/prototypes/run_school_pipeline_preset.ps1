param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [Parameter(Mandatory = $false)]
    [ValidateSet('Standard')]
    [string]$Mode = 'Standard',

    [Parameter(Mandatory = $false)]
    [ValidateSet('Japanese', 'EnglishMixed', 'English')]
    [string]$ContentType = 'Japanese',

    [Parameter(Mandatory = $false)]
    [int]$MaxDurationSeconds = 0,

    [Parameter(Mandatory = $false)]
    [int]$StartSeconds = 0,

    [Parameter(Mandatory = $false)]
    [switch]$SkipPostprocess,

    [Parameter(Mandatory = $false)]
    [switch]$RunStage2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$RunsRoot = Join-Path $Root 'runs'
$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$RunDir = Join-Path $RunsRoot "${Timestamp}_school_pipeline_${Mode}_${ContentType}"
$LogsDir = Join-Path $RunDir 'logs'
$LogFile = Join-Path $LogsDir 'log.txt'
$ManifestPath = Join-Path $RunDir 'pipeline_manifest.json'

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'), $Message
    $line | Out-File -FilePath $LogFile -Append -Encoding utf8
}

function Assert-FileExists {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}

function Get-NewestRun {
    param(
        [Parameter(Mandatory = $true)][datetime]$Since,
        [Parameter(Mandatory = $true)][string]$NamePattern
    )
    $run = Get-ChildItem -LiteralPath $RunsRoot -Directory |
        Where-Object { $_.Name -like $NamePattern -and $_.LastWriteTime -ge $Since.AddSeconds(-2) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $run) {
        throw "Could not find generated run folder matching: $NamePattern"
    }
    return $run.FullName
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Log "START_STEP $Name"
    Write-Log ('COMMAND: "{0}" {1}' -f $Command, (($Arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '))
    $powerShellExe = (Get-Process -Id $PID).Path
    $invokeArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Command) + $Arguments
    & $powerShellExe @invokeArgs 2>&1 |
        Tee-Object -FilePath (Join-Path $LogsDir "${Name}_stdout.txt") |
        Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed. ExitCode=$LASTEXITCODE"
    }
    Write-Log "END_STEP $Name OK"
}

function Get-Preset {
    param(
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][string]$ContentType
    )

    $preset = [ordered]@{
        mode = $Mode
        content_type = $ContentType
        target_ram_gb = 16
        asr_kind = 'hybrid'
        asr_model = 'reazon-k2-v2-int8 + faster-whisper-small-int8 patch'
        llm_enabled = $true
        llm_model = 'Qwen3.5-4B-Q4_K_M'
        n_ctx = 8192
        max_chars_per_batch = 5000
        max_tokens_part = 1200
        max_tokens_final = 1800
        max_patch_chunks = 8
        short_text_chars = 8
        min_candidate_seconds = 4.0
        patch_language = 'None'
        note = '標準構成。日本語中心はReazon K2を主に使い、短い崩れだけfaster-whisperで補助し、Qwen3.5-4Bで整形する。'
    }

    if ($ContentType -eq 'EnglishMixed') {
        $preset.asr_kind = 'ja_en_alternating'
        $preset.asr_model = 'reazon-k2-v2-int8 + faster-whisper-small-int8 en patch'
        $preset.max_patch_chunks = 120
        $preset.short_text_chars = 8
        $preset.min_candidate_seconds = 2.0
        $preset.patch_language = 'en'
        $preset.note = '英語授業・日英交互向け。英語らしい区間だけfaster-whisperで補助する。'
    }
    elseif ($ContentType -eq 'English') {
        $preset.asr_kind = 'faster_whisper'
        $preset.asr_model = 'faster-whisper-small-int8-auto'
        $preset.patch_language = 'None'
        $preset.note = '英語主体向け。Reazon先行処理を使わず、faster-whisper autoで処理する。'
    }

    return [pscustomobject]$preset
}

try {
    New-Item -ItemType Directory -Force -Path $RunDir, $LogsDir | Out-Null
    Assert-FileExists -Path $InputFile -Label 'Input audio'

    $started = Get-Date
    $preset = Get-Preset -Mode $Mode -ContentType $ContentType
    if ($SkipPostprocess) {
        $preset.llm_enabled = $false
    }

    Write-Log "START run_school_pipeline_preset"
    Write-Log "InputFile=$InputFile"
    Write-Log "Mode=$Mode ContentType=$ContentType StartSeconds=$StartSeconds MaxDurationSeconds=$MaxDurationSeconds SkipPostprocess=$($SkipPostprocess.IsPresent) RunStage2=$($RunStage2.IsPresent)"
    Write-Log "RunDir=$RunDir"
    Write-Log ("Preset=" + ($preset | ConvertTo-Json -Compress))

    $asrStart = Get-Date
    $asrRunDir = ''
    $transcriptFile = ''

    if ($preset.asr_kind -eq 'faster_whisper') {
        $fwRunner = Join-Path $Root 'scripts\prototypes\run_faster_whisper_test.ps1'
        Assert-FileExists -Path $fwRunner -Label 'faster-whisper runner'
        $fwModel = 'small'
        Invoke-Step -Name 'asr_faster_whisper' -Command $fwRunner -Arguments @(
            '-InputFile', $InputFile,
            '-Model', $fwModel,
            '-Language', 'None',
            '-ComputeType', 'int8',
            '-StartSeconds', "$StartSeconds",
            '-MaxDurationSeconds', "$MaxDurationSeconds"
        )
        $asrRunDir = Get-NewestRun -Since $asrStart -NamePattern "*_fw_${fwModel}_auto_*"
        $transcriptFile = Join-Path $asrRunDir 'output\full_transcript.txt'
    }
    elseif ($preset.asr_kind -eq 'ja_en_alternating') {
        $altRunner = Join-Path $Root 'scripts\prototypes\run_pipeline_ja_en_alternating.ps1'
        Assert-FileExists -Path $altRunner -Label 'Japanese/English alternating runner'
        Invoke-Step -Name 'asr_ja_en_alternating' -Command $altRunner -Arguments @(
            '-InputFile', $InputFile,
            '-StartSeconds', "$StartSeconds",
            '-MaxDurationSeconds', "$MaxDurationSeconds",
            '-MaxPatchChunks', "$($preset.max_patch_chunks)",
            '-ShortTextChars', "$($preset.short_text_chars)",
            '-MinCandidateSeconds', "$($preset.min_candidate_seconds)",
            '-PatchLanguage', "$($preset.patch_language)"
        )
        $asrRunDir = Get-NewestRun -Since $asrStart -NamePattern '*_ja_en_alternating_*'
        $transcriptFile = Join-Path $asrRunDir 'output\alternating_review_transcript.txt'
    }
    else {
        $hybridRunner = Join-Path $Root 'scripts\prototypes\run_pipeline_hybrid.ps1'
        Assert-FileExists -Path $hybridRunner -Label 'Hybrid runner'
        Invoke-Step -Name 'asr_hybrid_standard' -Command $hybridRunner -Arguments @(
            '-InputFile', $InputFile,
            '-StartSeconds', "$StartSeconds",
            '-MaxDurationSeconds', "$MaxDurationSeconds",
            '-MaxPatchChunks', "$($preset.max_patch_chunks)",
            '-ShortTextChars', "$($preset.short_text_chars)",
            '-MinCandidateSeconds', "$($preset.min_candidate_seconds)",
            '-PatchLanguage', "$($preset.patch_language)"
        )
        $asrRunDir = Get-NewestRun -Since $asrStart -NamePattern '*_hybrid_ja_fw_*'
        $transcriptFile = Join-Path $asrRunDir 'output\hybrid_review_transcript.txt'
    }

    Assert-FileExists -Path $transcriptFile -Label 'ASR transcript'
    Write-Log "AsrRunDir=$asrRunDir"
    Write-Log "TranscriptFile=$transcriptFile"

    $postprocessRunDir = ''
    if ($preset.llm_enabled) {
        $postStart = Get-Date
        $postRunner = Join-Path $Root 'scripts\production\run_school_hybrid_postprocess.ps1'
        Assert-FileExists -Path $postRunner -Label 'School postprocess runner'
        $postArgs = @(
            '-TranscriptFile', $transcriptFile,
            '-NCtx', "$($preset.n_ctx)",
            '-MaxCharsPerBatch', "$($preset.max_chars_per_batch)",
            '-MaxTokensPart', "$($preset.max_tokens_part)",
            '-MaxTokensFinal', "$($preset.max_tokens_final)"
        )
        if (-not $RunStage2) {
            $postArgs += '-Stage1Only'
        }
        Invoke-Step -Name 'llm_stage1_cleaning' -Command $postRunner -Arguments $postArgs
        $inputBase = [IO.Path]::GetFileNameWithoutExtension($transcriptFile)
        $postprocessRunDir = Get-NewestRun -Since $postStart -NamePattern "*_school_postprocess_${inputBase}"
        Write-Log "PostprocessRunDir=$postprocessRunDir"
    }
    else {
        Write-Log "PostprocessSkipped=true"
    }

    $totalSeconds = ((Get-Date) - $started).TotalSeconds
    $manifest = [ordered]@{
        input = $InputFile
        mode = $Mode
        content_type = $ContentType
        max_duration_seconds = $MaxDurationSeconds
        start_seconds = $StartSeconds
        preset = $preset
        coordinator_run_dir = $RunDir
        asr_run_dir = $asrRunDir
        transcript_file = $transcriptFile
        postprocess_run_dir = $postprocessRunDir
        total_seconds = [math]::Round($totalSeconds, 3)
    }
    $manifest | ConvertTo-Json -Depth 6 | Out-File -FilePath $ManifestPath -Encoding utf8
    Write-Log ('total_seconds={0:N3}' -f $totalSeconds)
    Write-Log "END run_school_pipeline_preset OK"

    Write-Host ''
    Write-Host 'Done.'
    Write-Host "Preset run       : $RunDir"
    Write-Host "Mode             : $Mode"
    Write-Host "Content type     : $ContentType"
    Write-Host "Target RAM       : $($preset.target_ram_gb)GB"
    Write-Host "ASR run          : $asrRunDir"
    Write-Host "Transcript       : $transcriptFile"
    if ($postprocessRunDir) {
        Write-Host "Postprocess run  : $postprocessRunDir"
        Write-Host "Raw transcript   : $(Join-Path $postprocessRunDir 'output\raw_transcript.txt')"
        Write-Host "Safe transcript  : $(Join-Path $postprocessRunDir 'output\safe_transcript.md')"
        Write-Host "Review flags     : $(Join-Path $postprocessRunDir 'output\review_flags.md')"
        if ($RunStage2) {
            Write-Host "School record    : $(Join-Path $postprocessRunDir 'output\school_record.md')"
            Write-Host "Summary          : $(Join-Path $postprocessRunDir 'output\summary.md')"
        }
    }
    else {
        Write-Host "Postprocess run  : skipped"
    }
    Write-Host "Manifest         : $ManifestPath"
}
catch {
    if (Test-Path -LiteralPath $LogsDir) {
        Write-Log "FATAL $($_.Exception.Message)"
    }
    Write-Error $_.Exception.Message
    exit 1
}
