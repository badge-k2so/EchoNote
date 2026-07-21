param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [Parameter(Mandatory = $false)]
    [ValidateSet('ja', 'auto', 'en')]
    [string]$AsrMode = 'ja',

    # 0 = full audio. Use 600 for a 10-minute smoke test.
    [Parameter(Mandatory = $false)]
    [int]$MaxDurationSeconds = 0,

    [Parameter(Mandatory = $false)]
    [string]$LlmModelFile = "",

    [Parameter(Mandatory = $false)]
    [int]$NCtx = 4096,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokensFormat = 2048,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokensSummary = 512
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$ReazonRunner = Join-Path $Root 'scripts\prototypes\run_reazon_k2_vad_test.ps1'
$FasterWhisperRunner = Join-Path $Root 'scripts\prototypes\run_faster_whisper_test.ps1'
$LlmRunner = Join-Path $Root 'scripts\prototypes\run_llm_postprocess.ps1'
$DefaultLlmModel = Join-Path $Root 'models\Qwen3.5-4B-Q4_K_M.gguf'

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$DurLabel = if ($MaxDurationSeconds -gt 0) { "${MaxDurationSeconds}sec" } else { 'full' }
$RunDir = Join-Path $Root "runs\${Timestamp}_cpu_pipeline_${AsrMode}_${DurLabel}"
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

function Copy-IfExists {
    param([string]$Source, [string]$Destination)
    if (Test-Path -LiteralPath $Source -PathType Leaf) {
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

try {
    New-Item -ItemType Directory -Force -Path $OutputDir, $LogsDir, $ArtifactsDir | Out-Null

    Assert-FileExists -Path $InputFile -Label 'Input audio'
    Assert-FileExists -Path $ReazonRunner -Label 'ReazonSpeech runner'
    Assert-FileExists -Path $FasterWhisperRunner -Label 'faster-whisper runner'
    Assert-FileExists -Path $LlmRunner -Label 'LLM postprocess runner'

    if ($LlmModelFile -eq '') {
        $LlmModelFile = $DefaultLlmModel
    }
    Assert-FileExists -Path $LlmModelFile -Label 'LLM GGUF model'

    $pipelineStart = Get-Date
    Write-Log "START run_pipeline_cpu"
    Write-Log "InputFile=$InputFile"
    Write-Log "AsrMode=$AsrMode"
    Write-Log "MaxDurationSeconds=$MaxDurationSeconds"
    Write-Log "LlmModelFile=$LlmModelFile"
    Write-Log "RunDir=$RunDir"

    $asrStart = Get-Date
    if ($AsrMode -eq 'ja') {
        Write-Log "ASR selected: ReazonSpeech-k2-v2 + VAD"
        & $ReazonRunner -InputFile $InputFile -MaxDurationSeconds $MaxDurationSeconds 2>&1 |
            Tee-Object -FilePath (Join-Path $LogsDir 'asr_runner_stdout.txt') |
            Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "ReazonSpeech ASR runner failed. ExitCode=$LASTEXITCODE"
        }
        $asrRunDir = Get-NewestRun -Since $asrStart -NamePattern '*_reazon_k2_vad_*'
        $asrEngine = 'reazon_k2_vad'
    }
    else {
        $fwLanguage = if ($AsrMode -eq 'auto') { 'None' } else { 'en' }
        Write-Log "ASR selected: faster-whisper small/int8 Language=$fwLanguage"
        & $FasterWhisperRunner -InputFile $InputFile -Model 'small' -Language $fwLanguage -ComputeType 'int8' -MaxDurationSeconds $MaxDurationSeconds 2>&1 |
            Tee-Object -FilePath (Join-Path $LogsDir 'asr_runner_stdout.txt') |
            Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "faster-whisper ASR runner failed. ExitCode=$LASTEXITCODE"
        }
        $langLabel = if ($AsrMode -eq 'auto') { 'auto' } else { 'en' }
        $asrRunDir = Get-NewestRun -Since $asrStart -NamePattern "*_fw_small_${langLabel}_*"
        $asrEngine = 'faster_whisper_small_int8'
    }

    $asrOutput = Join-Path $asrRunDir 'output'
    $asrTranscript = Join-Path $asrOutput 'full_transcript.txt'
    Assert-FileExists -Path $asrTranscript -Label 'ASR transcript'
    Write-Log "AsrRunDir=$asrRunDir"
    Write-Log "AsrTranscript=$asrTranscript"

    Copy-Item -LiteralPath $asrTranscript -Destination (Join-Path $OutputDir 'raw_transcript.txt') -Force
    Copy-IfExists -Source (Join-Path $asrOutput 'summary.json') -Destination (Join-Path $ArtifactsDir 'asr_summary.json')
    Copy-IfExists -Source (Join-Path $asrRunDir 'logs\log.txt') -Destination (Join-Path $ArtifactsDir 'asr_log.txt')

    $llmStart = Get-Date
    & $LlmRunner `
        -TranscriptFile (Join-Path $OutputDir 'raw_transcript.txt') `
        -ModelFile $LlmModelFile `
        -NCtx $NCtx `
        -MaxTokensFormat $MaxTokensFormat `
        -MaxTokensSummary $MaxTokensSummary 2>&1 |
        Tee-Object -FilePath (Join-Path $LogsDir 'llm_runner_stdout.txt') |
        Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "LLM postprocess runner failed. ExitCode=$LASTEXITCODE"
    }

    $llmRunDir = Get-NewestRun -Since $llmStart -NamePattern '*_llm_*'
    $llmOutput = Join-Path $llmRunDir 'output'
    Assert-FileExists -Path (Join-Path $llmOutput 'formatted.txt') -Label 'Formatted output'
    Assert-FileExists -Path (Join-Path $llmOutput 'summary.txt') -Label 'Summary output'
    Write-Log "LlmRunDir=$llmRunDir"

    Copy-Item -LiteralPath (Join-Path $llmOutput 'formatted.txt') -Destination (Join-Path $OutputDir 'formatted.txt') -Force
    Copy-Item -LiteralPath (Join-Path $llmOutput 'summary.txt') -Destination (Join-Path $OutputDir 'summary.txt') -Force
    Copy-IfExists -Source (Join-Path $llmOutput 'llm_result.json') -Destination (Join-Path $ArtifactsDir 'llm_result.json')
    Copy-IfExists -Source (Join-Path $llmRunDir 'logs\log.txt') -Destination (Join-Path $ArtifactsDir 'llm_log.txt')

    $asrSummaryPath = Join-Path $ArtifactsDir 'asr_summary.json'
    $llmResultPath = Join-Path $ArtifactsDir 'llm_result.json'
    $asrSummary = if (Test-Path -LiteralPath $asrSummaryPath) { Get-Content -LiteralPath $asrSummaryPath -Raw | ConvertFrom-Json } else { $null }
    $llmResult = if (Test-Path -LiteralPath $llmResultPath) { Get-Content -LiteralPath $llmResultPath -Raw | ConvertFrom-Json } else { $null }

    $pipelineSeconds = ((Get-Date) - $pipelineStart).TotalSeconds
    $metadata = [ordered]@{
        pipeline = 'cpu_stt_llm'
        created_at = (Get-Date).ToString('o')
        input_file = $InputFile
        asr_mode = $AsrMode
        asr_engine = $asrEngine
        asr_run_dir = $asrRunDir
        llm_model = $LlmModelFile
        llm_run_dir = $llmRunDir
        max_duration_seconds = $MaxDurationSeconds
        pipeline_seconds = [math]::Round($pipelineSeconds, 3)
        outputs = [ordered]@{
            raw_transcript = (Join-Path $OutputDir 'raw_transcript.txt')
            formatted = (Join-Path $OutputDir 'formatted.txt')
            summary = (Join-Path $OutputDir 'summary.txt')
            metadata = (Join-Path $OutputDir 'metadata.json')
        }
        asr_summary = $asrSummary
        llm_result = $llmResult
        english_mixed_operation = [ordered]@{
            default = 'Use -AsrMode ja for Japanese-heavy recordings.'
            english_class = 'Use -AsrMode auto or -AsrMode en for English classes or English-heavy recordings.'
            mixed_note = 'Japanese STT is fastest and best for Japanese domain terms; faster-whisper is better for English names and English-heavy speech.'
        }
        safety_note = 'AI outputs are drafts. A human must review important records before use.'
    }

    $metadata | ConvertTo-Json -Depth 10 | Out-File -FilePath (Join-Path $OutputDir 'metadata.json') -Encoding utf8
    Write-Log ('pipeline_seconds={0:N3}' -f $pipelineSeconds)
    Write-Log "END run_pipeline_cpu OK"

    Write-Host ''
    Write-Host 'Done.'
    Write-Host "Run folder      : $RunDir"
    Write-Host "ASR mode        : $AsrMode"
    Write-Host "ASR run         : $asrRunDir"
    Write-Host "LLM run         : $llmRunDir"
    Write-Host "Raw transcript  : $(Join-Path $OutputDir 'raw_transcript.txt')"
    Write-Host "Formatted       : $(Join-Path $OutputDir 'formatted.txt')"
    Write-Host "Summary         : $(Join-Path $OutputDir 'summary.txt')"
    Write-Host "Metadata        : $(Join-Path $OutputDir 'metadata.json')"
    Write-Host "Pipeline seconds: $([math]::Round($pipelineSeconds, 1))"
}
catch {
    if (Test-Path -LiteralPath $LogsDir) {
        Write-Log "FATAL $($_.Exception.Message)"
    }
    Write-Error $_.Exception.Message
    exit 1
}
