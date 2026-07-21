param(
    [Parameter(Mandatory = $false)]
    [string]$InputFile = "C:\Users\$env:USERNAME\Downloads\2026-05-22 13_30_39.ogg",

    [Parameter(Mandatory = $false)]
    [string]$Device = "cpu",

    [Parameter(Mandatory = $false)]
    [string]$Precision = "int8",

    [Parameter(Mandatory = $false)]
    [string]$Language = "ja",

    [Parameter(Mandatory = $false)]
    [double]$MaxChunkSeconds = 28.0,

    [Parameter(Mandatory = $false)]
    [string]$SilenceNoise = "-30dB",

    [Parameter(Mandatory = $false)]
    [double]$SilenceDuration = 0.3,

    # 0 = no limit (full audio); positive value = limit in seconds
    [Parameter(Mandatory = $false)]
    [int]$MaxDurationSeconds = 600,

    [Parameter(Mandatory = $false)]
    [int]$StartSeconds = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root          = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$FfmpegExe     = Join-Path $Root 'engines\ffmpeg\ffmpeg.exe'
$VenvPython    = Join-Path $Root '.venv\Scripts\python.exe'
$VadScript     = Join-Path $Root 'scripts\prototypes\vad_chunk.py'
$AsrScript     = Join-Path $Root 'scripts\prototypes\reazon_k2_transcribe.py'
$FilenameScript = Join-Path $Root 'scripts\production\record_filename.py'

$Timestamp   = Get-Date -Format 'yyyyMMdd_HHmmss'
$DurLabel    = if ($MaxDurationSeconds -gt 0) { "${MaxDurationSeconds}sec" } else { 'full' }
if ($StartSeconds -gt 0) { $DurLabel = "${StartSeconds}s_${DurLabel}" }
$RunDir      = Join-Path $Root "runs\${Timestamp}_reazon_k2_vad_${DurLabel}"
$SourceDir   = Join-Path $RunDir 'source'
$ChunksDir   = Join-Path $RunDir 'chunks_vad'
$OutputDir   = Join-Path $RunDir 'output'
$LogsDir     = Join-Path $RunDir 'logs'
$LogFile     = Join-Path $LogsDir 'log.txt'
$ResourceLog = Join-Path $LogsDir 'system_resources.csv'
$MonitorJob  = $null

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

function Assert-DirectoryExists {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label not found: $Path"
    }
}

function Invoke-LoggedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$OutputLog,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )
    Write-Log ('COMMAND: "{0}" {1}' -f $Exe, (($Arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '))
    & $Exe @Arguments 2>&1 | Tee-Object -FilePath $OutputLog | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$FailureMessage ExitCode=$exitCode See=$OutputLog"
    }
}

try {
    New-Item -ItemType Directory -Force -Path $SourceDir, $ChunksDir, $OutputDir, $LogsDir | Out-Null

    Assert-FileExists -Path $InputFile  -Label 'Input audio'
    Assert-FileExists -Path $FfmpegExe  -Label 'ffmpeg.exe'
    Assert-FileExists -Path $VenvPython -Label '.venv Python'
    Assert-FileExists -Path $VadScript  -Label 'VAD chunk script'
    Assert-FileExists -Path $AsrScript  -Label 'ReazonSpeech ASR script'
    Assert-FileExists -Path $FilenameScript -Label 'Filename suggestion script'
    Assert-DirectoryExists -Path (Join-Path $Root '.venv') -Label '.venv'

    $totalStart = Get-Date
    Write-Log "START run_reazon_k2_vad_test"
    Write-Log "ComputerName=$env:COMPUTERNAME"
    Write-Log "OS=$((Get-CimInstance Win32_OperatingSystem).Caption) $((Get-CimInstance Win32_OperatingSystem).Version)"
    Write-Log "PythonVersion=$(& $VenvPython --version)"
    Write-Log "InputFile=$InputFile"
    Write-Log "Device=$Device Precision=$Precision Language=$Language"
    Write-Log "StartSeconds=$StartSeconds MaxDurationSeconds=$MaxDurationSeconds MaxChunkSeconds=$MaxChunkSeconds SilenceNoise=$SilenceNoise SilenceDuration=$SilenceDuration"
    Write-Log "RunDir=$RunDir"

    # --- Step 1: convert to 16kHz mono WAV (full or limited) ---
    $wavLabel    = if ($MaxDurationSeconds -gt 0) { "test_${MaxDurationSeconds}sec.wav" } else { 'test_full.wav' }
    $testWav     = Join-Path $SourceDir $wavLabel
    $ffmpegConvArgs = @('-y')
    if ($StartSeconds -gt 0) { $ffmpegConvArgs += @('-ss', "$StartSeconds") }
    $ffmpegConvArgs += @('-i', $InputFile)
    if ($MaxDurationSeconds -gt 0) { $ffmpegConvArgs += @('-t', "$MaxDurationSeconds") }
    $ffmpegConvArgs += @('-ar', '16000', '-ac', '1', '-sample_fmt', 's16', $testWav)

    $convertStart = Get-Date
    Invoke-LoggedProcess `
        -Exe $FfmpegExe `
        -Arguments $ffmpegConvArgs `
        -OutputLog (Join-Path $LogsDir 'ffmpeg_convert.txt') `
        -FailureMessage 'WAV conversion failed.'
    $convertSeconds = ((Get-Date) - $convertStart).TotalSeconds
    Write-Log ('convert_seconds={0:N3}' -f $convertSeconds)

    # --- Step 2: VAD-based chunking ---
    $vadArgs = @(
        $VadScript,
        '--input',                    $testWav,
        '--output_dir',               $ChunksDir,
        '--ffmpeg',                   $FfmpegExe,
        "--max_seconds=$MaxChunkSeconds",
        "--silence_noise=$SilenceNoise",
        "--silence_duration=$SilenceDuration"
    )

    $vadStart = Get-Date
    Invoke-LoggedProcess `
        -Exe $VenvPython `
        -Arguments $vadArgs `
        -OutputLog (Join-Path $LogsDir 'vad_chunk_stdout.txt') `
        -FailureMessage 'VAD chunking failed.'
    $vadSeconds = ((Get-Date) - $vadStart).TotalSeconds
    $chunkCount = @(Get-ChildItem -LiteralPath $ChunksDir -Filter '*.wav' | Sort-Object Name).Count
    Write-Log ('vad_chunk_seconds={0:N3}' -f $vadSeconds)
    Write-Log "chunk_count=$chunkCount"

    if ($chunkCount -eq 0) {
        throw "No VAD chunks were created."
    }

    # --- Resource monitor ---
    $MonitorJob = Start-Job -ArgumentList $ResourceLog -ScriptBlock {
        param([string]$CsvPath)
        'timestamp,total_mb,available_mb,used_mb,python_working_set_mb,python_cpu_seconds,ffmpeg_working_set_mb,ffmpeg_cpu_seconds' |
            Out-File -FilePath $CsvPath -Encoding utf8
        while ($true) {
            $os        = Get-CimInstance Win32_OperatingSystem
            $totalMb   = [math]::Round($os.TotalVisibleMemorySize / 1024, 1)
            $availMb   = [math]::Round($os.FreePhysicalMemory / 1024, 1)
            $usedMb    = [math]::Round($totalMb - $availMb, 1)
            $python    = @(Get-Process -Name 'python'  -ErrorAction SilentlyContinue)
            $ffmpeg    = @(Get-Process -Name 'ffmpeg'  -ErrorAction SilentlyContinue)
            $pyMb      = [math]::Round((($python | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB), 1)
            $pyCpu     = [math]::Round((($python | Measure-Object -Property CPU -Sum).Sum), 3)
            $fMb       = [math]::Round((($ffmpeg | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB), 1)
            $fCpu      = [math]::Round((($ffmpeg | Measure-Object -Property CPU -Sum).Sum), 3)
            '{0},{1},{2},{3},{4},{5},{6},{7}' -f (Get-Date -Format 'o'), $totalMb, $availMb, $usedMb, $pyMb, $pyCpu, $fMb, $fCpu |
                Out-File -FilePath $CsvPath -Append -Encoding utf8
            Start-Sleep -Milliseconds 500
        }
    }
    Write-Log "resource_monitor=$ResourceLog"

    # --- Step 3: ReazonSpeech ASR ---
    $asrArgs = @(
        $AsrScript,
        '--chunks_dir', $ChunksDir,
        '--output_dir', $OutputDir,
        '--log',        $LogFile,
        '--device',     $Device,
        '--precision',  $Precision,
        '--language',   $Language
    )

    $asrStart = Get-Date
    Invoke-LoggedProcess `
        -Exe $VenvPython `
        -Arguments $asrArgs `
        -OutputLog (Join-Path $LogsDir 'reazon_k2_stdout.txt') `
        -FailureMessage 'ReazonSpeech k2 transcription failed.'
    $asrWallSeconds = ((Get-Date) - $asrStart).TotalSeconds
    Write-Log ('python_asr_wall_seconds={0:N3}' -f $asrWallSeconds)

    if ($MonitorJob) {
        Stop-Job   -Job $MonitorJob -ErrorAction SilentlyContinue
        Remove-Job -Job $MonitorJob -Force -ErrorAction SilentlyContinue
        $MonitorJob = $null
    }

    if (Test-Path -LiteralPath $ResourceLog) {
        $rows = @(Import-Csv -LiteralPath $ResourceLog)
        if ($rows.Count -gt 0) {
            Write-Log "resource_peak_python_working_set_mb=$(($rows | Measure-Object python_working_set_mb -Maximum).Maximum)"
            Write-Log "resource_peak_system_used_mb=$(($rows | Measure-Object used_mb -Maximum).Maximum)"
            Write-Log "resource_min_available_mb=$(($rows | Measure-Object available_mb -Minimum).Minimum)"
            Write-Log "resource_max_python_cpu_seconds=$(($rows | Measure-Object python_cpu_seconds -Maximum).Maximum)"
        }
    }

    $transcriptPath = Join-Path $OutputDir 'full_transcript.txt'
    $filenameSuggestionPath = Join-Path $OutputDir 'filename_suggestion.json'
    & $VenvPython $FilenameScript `
        --audio $InputFile `
        --transcript $transcriptPath `
        --output $filenameSuggestionPath 2>&1 |
        Tee-Object -FilePath (Join-Path $LogsDir 'filename_suggestion_stdout.txt') |
        Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Log "WARNING filename_suggestion_failed exit_code=$LASTEXITCODE"
    }
    else {
        Write-Log "filename_suggestion=$filenameSuggestionPath"
    }

    $summaryPath = Join-Path $OutputDir 'summary.json'
    $summary     = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $totalSeconds = ((Get-Date) - $totalStart).TotalSeconds
    Write-Log ('total_seconds={0:N3}' -f $totalSeconds)
    Write-Log "END run_reazon_k2_vad_test OK"

    Write-Host ""
    Write-Host "Done."
    Write-Host "Run folder    : $RunDir"
    Write-Host "Chunks (VAD)  : $chunkCount"
    Write-Host "Success chunks: $($summary.success)"
    Write-Host "Failed chunks : $($summary.failed)"
    Write-Host "Model load    : $($summary.model_load_seconds) sec"
    Write-Host "ASR total     : $($summary.asr_total_seconds) sec"
    Write-Host "Transcript    : $transcriptPath"
    Write-Host "Filename idea : $filenameSuggestionPath"
}
catch {
    if ($MonitorJob) {
        Stop-Job   -Job $MonitorJob -ErrorAction SilentlyContinue
        Remove-Job -Job $MonitorJob -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $LogsDir) {
        Write-Log "FATAL $($_.Exception.Message)"
    }
    Write-Error $_.Exception.Message
    exit 1
}
