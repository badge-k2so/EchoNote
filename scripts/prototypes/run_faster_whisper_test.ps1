param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [Parameter(Mandatory = $false)]
    [string]$Model = "small",

    [Parameter(Mandatory = $false)]
    [string]$Language = "None",      # "None" = auto-detect, "ja", "en", etc.

    [Parameter(Mandatory = $false)]
    [string]$ComputeType = "int8",

    [Parameter(Mandatory = $false)]
    [int]$MaxDurationSeconds = 600,  # 0 = full audio

    [Parameter(Mandatory = $false)]
    [int]$StartSeconds = 0,

    [Parameter(Mandatory = $false)]
    [string]$ModelDir = ""           # "" = HuggingFace default cache
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root       = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$FfmpegExe  = Join-Path $Root 'engines\ffmpeg\ffmpeg.exe'
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$PyScript   = Join-Path $Root 'scripts\prototypes\faster_whisper_transcribe.py'

$Timestamp  = Get-Date -Format 'yyyyMMdd_HHmmss'
$LangLabel  = if ($Language -eq "None") { "auto" } else { $Language }
$DurLabel   = if ($MaxDurationSeconds -gt 0) { "${MaxDurationSeconds}sec" } else { "full" }
if ($StartSeconds -gt 0) { $DurLabel = "${StartSeconds}s_${DurLabel}" }
$RunDir     = Join-Path $Root "runs\${Timestamp}_fw_${Model}_${LangLabel}_${DurLabel}"
$SourceDir  = Join-Path $RunDir 'source'
$OutputDir  = Join-Path $RunDir 'output'
$LogsDir    = Join-Path $RunDir 'logs'
$LogFile    = Join-Path $LogsDir 'log.txt'
$ResourceLog = Join-Path $LogsDir 'system_resources.csv'
$MonitorJob = $null

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'), $Message
    $line | Out-File -FilePath $LogFile -Append -Encoding utf8
}

function Assert-FileExists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Label not found: $Path" }
}

function Invoke-LoggedProcess {
    param([string]$Exe, [string[]]$Arguments, [string]$OutputLog, [string]$FailureMessage)
    Write-Log ('COMMAND: "{0}" {1}' -f $Exe, (($Arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '))
    & $Exe @Arguments 2>&1 | Tee-Object -FilePath $OutputLog | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "$FailureMessage ExitCode=$LASTEXITCODE See=$OutputLog" }
}

try {
    New-Item -ItemType Directory -Force -Path $SourceDir, $OutputDir, $LogsDir | Out-Null

    Assert-FileExists -Path $InputFile  -Label 'Input audio'
    Assert-FileExists -Path $FfmpegExe  -Label 'ffmpeg.exe'
    Assert-FileExists -Path $VenvPython -Label '.venv Python'
    Assert-FileExists -Path $PyScript   -Label 'faster_whisper_transcribe.py'

    $totalStart = Get-Date
    Write-Log "START run_faster_whisper_test"
    Write-Log "InputFile=$InputFile"
    Write-Log "Model=$Model ComputeType=$ComputeType Language=$Language StartSeconds=$StartSeconds MaxDurationSeconds=$MaxDurationSeconds"
    Write-Log "RunDir=$RunDir"

    # --- Step 1: WAV 変換 ---
    $wavLabel = if ($MaxDurationSeconds -gt 0) { "test_${MaxDurationSeconds}sec.wav" } else { "test_full.wav" }
    $testWav  = Join-Path $SourceDir $wavLabel
    $convArgs = @('-y')
    if ($StartSeconds -gt 0) { $convArgs += @('-ss', "$StartSeconds") }
    $convArgs += @('-i', $InputFile)
    if ($MaxDurationSeconds -gt 0) { $convArgs += @('-t', "$MaxDurationSeconds") }
    $convArgs += @('-ar', '16000', '-ac', '1', '-sample_fmt', 's16', $testWav)

    $convertStart = Get-Date
    Invoke-LoggedProcess -Exe $FfmpegExe -Arguments $convArgs `
        -OutputLog (Join-Path $LogsDir 'ffmpeg_convert.txt') -FailureMessage 'WAV conversion failed.'
    Write-Log ('convert_seconds={0:N3}' -f ((Get-Date) - $convertStart).TotalSeconds)

    # --- Resource monitor ---
    $MonitorJob = Start-Job -ArgumentList $ResourceLog -ScriptBlock {
        param([string]$CsvPath)
        'timestamp,total_mb,available_mb,used_mb,python_working_set_mb,python_cpu_seconds' |
            Out-File -FilePath $CsvPath -Encoding utf8
        while ($true) {
            $os     = Get-CimInstance Win32_OperatingSystem
            $total  = [math]::Round($os.TotalVisibleMemorySize / 1024, 1)
            $avail  = [math]::Round($os.FreePhysicalMemory / 1024, 1)
            $python = @(Get-Process -Name 'python' -ErrorAction SilentlyContinue)
            $pyMb   = [math]::Round((($python | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB), 1)
            $pyCpu  = [math]::Round((($python | Measure-Object -Property CPU -Sum).Sum), 3)
            '{0},{1},{2},{3},{4},{5}' -f (Get-Date -Format 'o'), $total, $avail, ($total - $avail), $pyMb, $pyCpu |
                Out-File -FilePath $CsvPath -Append -Encoding utf8
            Start-Sleep -Milliseconds 500
        }
    }

    # --- Step 2: faster-whisper 転写 ---
    $pyArgs = @(
        $PyScript,
        '--input',        $testWav,
        '--output_dir',   $OutputDir,
        '--log',          $LogFile,
        '--model',        $Model,
        '--language',     $Language,
        '--compute_type', $ComputeType
    )
    if ($ModelDir -ne "") { $pyArgs += @('--model_dir', $ModelDir) }

    $asrStart = Get-Date
    Invoke-LoggedProcess -Exe $VenvPython -Arguments $pyArgs `
        -OutputLog (Join-Path $LogsDir 'faster_whisper_stdout.txt') -FailureMessage 'faster-whisper failed.'
    Write-Log ('asr_wall_seconds={0:N3}' -f ((Get-Date) - $asrStart).TotalSeconds)

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

    $summary  = Get-Content -LiteralPath (Join-Path $OutputDir 'summary.json') -Raw | ConvertFrom-Json
    $totalSec = ((Get-Date) - $totalStart).TotalSeconds
    Write-Log ('total_seconds={0:N3}' -f $totalSec)
    Write-Log "END run_faster_whisper_test OK"

    Write-Host ""
    Write-Host "Done."
    Write-Host "Run folder       : $RunDir"
    Write-Host "Detected language: $($summary.detected_language) (p=$($summary.detected_language_probability))"
    Write-Host "Model load       : $($summary.model_load_seconds) sec"
    Write-Host "ASR total        : $($summary.asr_seconds) sec"
    Write-Host "End-to-end       : $([math]::Round($totalSec,3)) sec"
    Write-Host "Segments         : $($summary.segments)"
    Write-Host "Transcript       : $(Join-Path $OutputDir 'full_transcript.txt')"
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
