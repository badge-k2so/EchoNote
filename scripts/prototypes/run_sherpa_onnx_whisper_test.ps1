param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [Parameter(Mandatory = $false)]
    [string]$ModelDir = "models\sherpa-onnx-whisper-tiny.en",

    [Parameter(Mandatory = $false)]
    [string]$ModelPrefix = "tiny.en",

    [Parameter(Mandatory = $false)]
    [int]$NumThreads = 2,

    [Parameter(Mandatory = $false)]
    [double]$MaxChunkSeconds = 28.0,

    [Parameter(Mandatory = $false)]
    [string]$SilenceNoise = "-30dB",

    [Parameter(Mandatory = $false)]
    [double]$SilenceDuration = 0.3,

    [Parameter(Mandatory = $false)]
    [int]$MaxDurationSeconds = 600,

    [Parameter(Mandatory = $false)]
    [int]$StartSeconds = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root       = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$FfmpegExe  = Join-Path $Root 'engines\ffmpeg\ffmpeg.exe'
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$VadScript  = Join-Path $Root 'scripts\prototypes\vad_chunk.py'
$AsrScript  = Join-Path $Root 'scripts\prototypes\sherpa_onnx_whisper_transcribe.py'

if (-not [System.IO.Path]::IsPathRooted($ModelDir)) {
    $ModelDir = Join-Path $Root $ModelDir
}

$Encoder = Join-Path $ModelDir "$ModelPrefix-encoder.int8.onnx"
$Decoder = Join-Path $ModelDir "$ModelPrefix-decoder.int8.onnx"
$Tokens  = Join-Path $ModelDir "$ModelPrefix-tokens.txt"

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$DurLabel  = if ($MaxDurationSeconds -gt 0) { "${MaxDurationSeconds}sec" } else { 'full' }
if ($StartSeconds -gt 0) { $DurLabel = "${StartSeconds}s_${DurLabel}" }
$ModelLabel = Split-Path -Leaf $ModelDir
$RunDir    = Join-Path $Root "runs\${Timestamp}_sherpa_onnx_${ModelLabel}_${DurLabel}"
$SourceDir = Join-Path $RunDir 'source'
$ChunksDir = Join-Path $RunDir 'chunks_vad'
$OutputDir = Join-Path $RunDir 'output'
$LogsDir   = Join-Path $RunDir 'logs'
$LogFile   = Join-Path $LogsDir 'log.txt'
$ResourceLog = Join-Path $LogsDir 'system_resources.csv'
$MonitorJob = $null

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'), $Message
    $line | Out-File -FilePath $LogFile -Append -Encoding utf8
}

function Assert-FileExists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}

function Invoke-LoggedProcess {
    param([string]$Exe, [string[]]$Arguments, [string]$OutputLog, [string]$FailureMessage)
    Write-Log ('COMMAND: "{0}" {1}' -f $Exe, (($Arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '))
    & $Exe @Arguments 2>&1 | Tee-Object -FilePath $OutputLog | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage ExitCode=$LASTEXITCODE See=$OutputLog"
    }
}

try {
    New-Item -ItemType Directory -Force -Path $SourceDir, $ChunksDir, $OutputDir, $LogsDir | Out-Null

    Assert-FileExists -Path $InputFile  -Label 'Input audio'
    Assert-FileExists -Path $FfmpegExe  -Label 'ffmpeg.exe'
    Assert-FileExists -Path $VenvPython -Label '.venv Python'
    Assert-FileExists -Path $VadScript  -Label 'VAD chunk script'
    Assert-FileExists -Path $AsrScript  -Label 'sherpa-onnx ASR script'
    Assert-FileExists -Path $Encoder    -Label 'Whisper encoder'
    Assert-FileExists -Path $Decoder    -Label 'Whisper decoder'
    Assert-FileExists -Path $Tokens     -Label 'Whisper tokens'

    $totalStart = Get-Date
    Write-Log "START run_sherpa_onnx_whisper_test"
    Write-Log "InputFile=$InputFile"
    Write-Log "ModelDir=$ModelDir ModelPrefix=$ModelPrefix NumThreads=$NumThreads"
    Write-Log "StartSeconds=$StartSeconds MaxDurationSeconds=$MaxDurationSeconds MaxChunkSeconds=$MaxChunkSeconds SilenceNoise=$SilenceNoise SilenceDuration=$SilenceDuration"
    Write-Log "RunDir=$RunDir"

    $wavLabel = if ($MaxDurationSeconds -gt 0) { "test_${MaxDurationSeconds}sec.wav" } else { 'test_full.wav' }
    $testWav = Join-Path $SourceDir $wavLabel
    $convArgs = @('-y')
    if ($StartSeconds -gt 0) { $convArgs += @('-ss', "$StartSeconds") }
    $convArgs += @('-i', $InputFile)
    if ($MaxDurationSeconds -gt 0) { $convArgs += @('-t', "$MaxDurationSeconds") }
    $convArgs += @('-ar', '16000', '-ac', '1', '-sample_fmt', 's16', $testWav)

    Invoke-LoggedProcess -Exe $FfmpegExe -Arguments $convArgs `
        -OutputLog (Join-Path $LogsDir 'ffmpeg_convert.txt') -FailureMessage 'WAV conversion failed.'

    $vadArgs = @(
        $VadScript,
        '--input', $testWav,
        '--output_dir', $ChunksDir,
        '--ffmpeg', $FfmpegExe,
        "--max_seconds=$MaxChunkSeconds",
        "--silence_noise=$SilenceNoise",
        "--silence_duration=$SilenceDuration"
    )
    Invoke-LoggedProcess -Exe $VenvPython -Arguments $vadArgs `
        -OutputLog (Join-Path $LogsDir 'vad_chunk_stdout.txt') -FailureMessage 'VAD chunking failed.'

    $chunkCount = @(Get-ChildItem -LiteralPath $ChunksDir -Filter '*.wav' | Sort-Object Name).Count
    Write-Log "chunk_count=$chunkCount"
    if ($chunkCount -eq 0) {
        throw "No VAD chunks were created."
    }

    $MonitorJob = Start-Job -ArgumentList $ResourceLog -ScriptBlock {
        param([string]$CsvPath)
        'timestamp,total_mb,available_mb,used_mb,python_working_set_mb,python_cpu_seconds' |
            Out-File -FilePath $CsvPath -Encoding utf8
        while ($true) {
            $os = Get-CimInstance Win32_OperatingSystem
            $total = [math]::Round($os.TotalVisibleMemorySize / 1024, 1)
            $avail = [math]::Round($os.FreePhysicalMemory / 1024, 1)
            $python = @(Get-Process -Name 'python' -ErrorAction SilentlyContinue)
            $pyMb = [math]::Round((($python | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB), 1)
            $pyCpu = [math]::Round((($python | Measure-Object -Property CPU -Sum).Sum), 3)
            '{0},{1},{2},{3},{4},{5}' -f (Get-Date -Format 'o'), $total, $avail, ($total - $avail), $pyMb, $pyCpu |
                Out-File -FilePath $CsvPath -Append -Encoding utf8
            Start-Sleep -Milliseconds 500
        }
    }

    $pyArgs = @(
        $AsrScript,
        '--chunks_dir', $ChunksDir,
        '--output_dir', $OutputDir,
        '--log', $LogFile,
        '--encoder', $Encoder,
        '--decoder', $Decoder,
        '--tokens', $Tokens,
        '--language', 'en',
        '--task', 'transcribe',
        '--num_threads', "$NumThreads",
        '--provider', 'cpu'
    )
    Invoke-LoggedProcess -Exe $VenvPython -Arguments $pyArgs `
        -OutputLog (Join-Path $LogsDir 'sherpa_onnx_stdout.txt') -FailureMessage 'sherpa-onnx Whisper failed.'

    if ($MonitorJob) {
        Stop-Job -Job $MonitorJob -ErrorAction SilentlyContinue
        Remove-Job -Job $MonitorJob -Force -ErrorAction SilentlyContinue
        $MonitorJob = $null
    }

    if (Test-Path -LiteralPath $ResourceLog) {
        $rows = @(Import-Csv -LiteralPath $ResourceLog)
        if ($rows.Count -gt 0) {
            Write-Log "resource_peak_python_mb=$(($rows | Measure-Object python_working_set_mb -Maximum).Maximum)"
            Write-Log "resource_max_python_cpu_seconds=$(($rows | Measure-Object python_cpu_seconds -Maximum).Maximum)"
        }
    }

    $summary = Get-Content -LiteralPath (Join-Path $OutputDir 'summary.json') -Raw | ConvertFrom-Json
    $totalSec = ((Get-Date) - $totalStart).TotalSeconds
    Write-Log ('total_seconds={0:N3}' -f $totalSec)
    Write-Log "END run_sherpa_onnx_whisper_test OK"

    Write-Host ""
    Write-Host "Done."
    Write-Host "Run folder    : $RunDir"
    Write-Host "Chunks (VAD)  : $chunkCount"
    Write-Host "Success chunks: $($summary.success)"
    Write-Host "Failed chunks : $($summary.failed)"
    Write-Host "Model load    : $($summary.model_load_seconds) sec"
    Write-Host "ASR total     : $($summary.asr_total_seconds) sec"
    Write-Host "End-to-end    : $([math]::Round($totalSec,3)) sec"
    Write-Host "Transcript    : $(Join-Path $OutputDir 'full_transcript.txt')"
}
catch {
    if ($MonitorJob) {
        Stop-Job -Job $MonitorJob -ErrorAction SilentlyContinue
        Remove-Job -Job $MonitorJob -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $LogsDir) { Write-Log "FATAL $($_.Exception.Message)" }
    Write-Error $_.Exception.Message
    exit 1
}
