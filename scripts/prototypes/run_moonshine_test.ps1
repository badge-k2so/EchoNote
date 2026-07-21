param(
    [Parameter(Mandatory = $false)]
    [string]$InputFile = "C:\Users\$env:USERNAME\Downloads\2026-05-22 13_30_39.ogg",

    [Parameter(Mandatory = $false)]
    [string]$Language = "ja"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$FfmpegExe = Join-Path $Root 'engines\ffmpeg\ffmpeg.exe'
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$PythonScript = Join-Path $Root 'scripts\prototypes\moonshine_transcribe.py'

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$RunDir = Join-Path $Root "runs\${Timestamp}_moonshine_ja"
$SourceDir = Join-Path $RunDir 'source'
$ChunksDir = Join-Path $RunDir 'chunks_30sec'
$OutputDir = Join-Path $RunDir 'output'
$LogsDir = Join-Path $RunDir 'logs'
$LogFile = Join-Path $LogsDir 'log.txt'
$ResourceLog = Join-Path $LogsDir 'system_resources.csv'
$MonitorJob = $null

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
    Assert-FileExists -Path $InputFile -Label 'Input audio'
    Assert-FileExists -Path $FfmpegExe -Label 'ffmpeg.exe'
    Assert-FileExists -Path $VenvPython -Label '.venv Python'
    Assert-FileExists -Path $PythonScript -Label 'Moonshine Python script'

    $totalStart = Get-Date
    Write-Log "START run_moonshine_test"
    Write-Log "ComputerName=$env:COMPUTERNAME"
    Write-Log "OS=$((Get-CimInstance Win32_OperatingSystem).Caption) $((Get-CimInstance Win32_OperatingSystem).Version)"
    Write-Log "PythonVersion=$(& $VenvPython --version)"
    Write-Log "InputFile=$InputFile"
    Write-Log "Language=$Language"
    Write-Log "RunDir=$RunDir"

    $testWav = Join-Path $SourceDir 'test_10min.wav'
    $convertStart = Get-Date
    Invoke-LoggedProcess `
        -Exe $FfmpegExe `
        -Arguments @('-y', '-i', $InputFile, '-t', '600', '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', $testWav) `
        -OutputLog (Join-Path $LogsDir 'ffmpeg_10min.txt') `
        -FailureMessage '10-minute WAV conversion failed.'
    Write-Log ('convert_10min_seconds={0:N3}' -f ((Get-Date) - $convertStart).TotalSeconds)

    $chunkPattern = Join-Path $ChunksDir 'chunk_%03d.wav'
    $chunkStart = Get-Date
    Invoke-LoggedProcess `
        -Exe $FfmpegExe `
        -Arguments @('-y', '-i', $testWav, '-f', 'segment', '-segment_time', '30', '-reset_timestamps', '1', $chunkPattern) `
        -OutputLog (Join-Path $LogsDir 'ffmpeg_chunks.txt') `
        -FailureMessage '30-second chunk creation failed.'
    $chunkCount = @(Get-ChildItem -LiteralPath $ChunksDir -Filter '*.wav' | Sort-Object Name).Count
    Write-Log ('split_30sec_seconds={0:N3}' -f ((Get-Date) - $chunkStart).TotalSeconds)
    Write-Log "chunk_count=$chunkCount"
    if ($chunkCount -eq 0) {
        throw "No 30-second chunks were created."
    }

    $MonitorJob = Start-Job -ArgumentList $ResourceLog -ScriptBlock {
        param([string]$CsvPath)
        'timestamp,total_mb,available_mb,used_mb,python_working_set_mb,python_cpu_seconds,ffmpeg_working_set_mb,ffmpeg_cpu_seconds' |
            Out-File -FilePath $CsvPath -Encoding utf8
        while ($true) {
            $os = Get-CimInstance Win32_OperatingSystem
            $totalMb = [math]::Round($os.TotalVisibleMemorySize / 1024, 1)
            $availableMb = [math]::Round($os.FreePhysicalMemory / 1024, 1)
            $usedMb = [math]::Round($totalMb - $availableMb, 1)
            $python = @(Get-Process -Name 'python' -ErrorAction SilentlyContinue)
            $ffmpeg = @(Get-Process -Name 'ffmpeg' -ErrorAction SilentlyContinue)
            $pythonMb = [math]::Round((($python | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB), 1)
            $pythonCpu = [math]::Round((($python | Measure-Object -Property CPU -Sum).Sum), 3)
            $ffmpegMb = [math]::Round((($ffmpeg | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB), 1)
            $ffmpegCpu = [math]::Round((($ffmpeg | Measure-Object -Property CPU -Sum).Sum), 3)
            '{0},{1},{2},{3},{4},{5},{6},{7}' -f (Get-Date -Format 'o'), $totalMb, $availableMb, $usedMb, $pythonMb, $pythonCpu, $ffmpegMb, $ffmpegCpu |
                Out-File -FilePath $CsvPath -Append -Encoding utf8
            Start-Sleep -Milliseconds 500
        }
    }
    Write-Log "resource_monitor=$ResourceLog"

    $asrStart = Get-Date
    Invoke-LoggedProcess `
        -Exe $VenvPython `
        -Arguments @($PythonScript, '--chunks_dir', $ChunksDir, '--output_dir', $OutputDir, '--log', $LogFile, '--language', $Language) `
        -OutputLog (Join-Path $LogsDir 'moonshine_stdout.txt') `
        -FailureMessage 'Moonshine transcription failed.'
    Write-Log ('python_asr_wall_seconds={0:N3}' -f ((Get-Date) - $asrStart).TotalSeconds)

    if ($MonitorJob) {
        Stop-Job -Job $MonitorJob -ErrorAction SilentlyContinue
        Remove-Job -Job $MonitorJob -Force -ErrorAction SilentlyContinue
        $MonitorJob = $null
    }

    if (Test-Path -LiteralPath $ResourceLog) {
        $resourceRows = @(Import-Csv -LiteralPath $ResourceLog)
        if ($resourceRows.Count -gt 0) {
            Write-Log "resource_peak_python_working_set_mb=$(($resourceRows | Measure-Object python_working_set_mb -Maximum).Maximum)"
            Write-Log "resource_peak_system_used_mb=$(($resourceRows | Measure-Object used_mb -Maximum).Maximum)"
            Write-Log "resource_min_available_mb=$(($resourceRows | Measure-Object available_mb -Minimum).Minimum)"
            Write-Log "resource_max_python_cpu_seconds=$(($resourceRows | Measure-Object python_cpu_seconds -Maximum).Maximum)"
        }
    }

    $summary = Get-Content -LiteralPath (Join-Path $OutputDir 'summary.json') -Raw | ConvertFrom-Json
    Write-Log ('total_seconds={0:N3}' -f ((Get-Date) - $totalStart).TotalSeconds)
    Write-Log "END run_moonshine_test OK"

    Write-Host ""
    Write-Host "Done."
    Write-Host "Run folder: $RunDir"
    Write-Host "Chunks: $chunkCount"
    Write-Host "Success chunks: $($summary.success)"
    Write-Host "Failed chunks: $($summary.failed)"
    Write-Host "Model load seconds: $($summary.model_load_seconds)"
    Write-Host "ASR total seconds: $($summary.asr_total_seconds)"
    Write-Host "Transcript: $(Join-Path $OutputDir 'full_transcript.txt')"
}
catch {
    if ($MonitorJob) {
        Stop-Job -Job $MonitorJob -ErrorAction SilentlyContinue
        Remove-Job -Job $MonitorJob -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $LogsDir) {
        Write-Log "FATAL $($_.Exception.Message)"
    }
    Write-Error $_.Exception.Message
    exit 1
}
