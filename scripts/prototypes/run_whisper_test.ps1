<#
Windows PowerShell test runner for ffmpeg + whisper.cpp.

What it does:
- Converts the input audio to 16 kHz mono 16-bit PCM WAV chunks.
- Splits audio into 10-minute chunks.
- Runs whisper-cli.exe sequentially with the base model.
- Writes txt, srt, combined transcript, and timing logs.
#>
param(
    [Parameter(Mandatory = $false)]
    [string]$InputFile = "C:\Users\$env:USERNAME\Downloads\2026-05-22 13_30_39.ogg",

    [Parameter(Mandatory = $false)]
    [bool]$TEST_ONLY = $true,

    [Parameter(Mandatory = $false)]
    [ValidateSet('base', 'small', 'medium')]
    [string]$Model = 'base',

    [Parameter(Mandatory = $false)]
    [string]$Language = 'ja'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$FfmpegExe = Join-Path $Root 'engines\ffmpeg\ffmpeg.exe'
$WhisperExe = Join-Path $Root 'engines\whisper\whisper-cli.exe'
$ModelPath = Join-Path $Root "engines\whisper\models\ggml-$Model.bin"
$ChunkSeconds = 600

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$RunDir = Join-Path $Root "runs\${Timestamp}_$Model"
$ChunksDir = Join-Path $RunDir 'chunks'
$OutputDir = Join-Path $RunDir 'output'
$LogsDir = Join-Path $RunDir 'logs'
$LogFile = Join-Path $LogsDir 'log.txt'
$MemoryLogFile = Join-Path $LogsDir 'memory.csv'
$MemoryMonitorJob = $null

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
    New-Item -ItemType Directory -Force -Path $ChunksDir, $OutputDir, $LogsDir | Out-Null

    Assert-FileExists -Path $InputFile -Label 'Input audio'
    Assert-FileExists -Path $FfmpegExe -Label 'ffmpeg.exe'
    Assert-FileExists -Path $WhisperExe -Label 'whisper-cli.exe'
    Assert-FileExists -Path $ModelPath -Label 'Whisper model'

    $totalStart = Get-Date
    Write-Log "START"
    Write-Log "InputFile=$InputFile"
    Write-Log "TEST_ONLY=$TEST_ONLY"
    Write-Log "Model=$Model"
    Write-Log "Language=$Language"
    Write-Log "ModelPath=$ModelPath"
    Write-Log "RunDir=$RunDir"

    $MemoryMonitorJob = Start-Job -ArgumentList $MemoryLogFile -ScriptBlock {
        param([string]$CsvPath)

        'timestamp,total_mb,available_mb,used_mb,ffmpeg_working_set_mb,whisper_working_set_mb,total_target_process_mb' |
            Out-File -FilePath $CsvPath -Encoding utf8

        while ($true) {
            $os = Get-CimInstance Win32_OperatingSystem
            $totalMb = [math]::Round($os.TotalVisibleMemorySize / 1024, 1)
            $availableMb = [math]::Round($os.FreePhysicalMemory / 1024, 1)
            $usedMb = [math]::Round($totalMb - $availableMb, 1)

            $ffmpegMb = [math]::Round(((Get-Process -Name 'ffmpeg' -ErrorAction SilentlyContinue |
                Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB), 1)
            $whisperMb = [math]::Round(((Get-Process -Name 'whisper-cli' -ErrorAction SilentlyContinue |
                Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB), 1)
            $targetMb = [math]::Round($ffmpegMb + $whisperMb, 1)

            '{0},{1},{2},{3},{4},{5},{6}' -f `
                (Get-Date -Format 'o'), $totalMb, $availableMb, $usedMb, $ffmpegMb, $whisperMb, $targetMb |
                Out-File -FilePath $CsvPath -Append -Encoding utf8

            Start-Sleep -Milliseconds 500
        }
    }
    Write-Log "memory_monitor=$MemoryLogFile"

    $chunkPattern = Join-Path $ChunksDir 'chunk_%04d.wav'
    $ffmpegArgs = @(
        '-y',
        '-i', $InputFile
    )

    if ($TEST_ONLY) {
        $ffmpegArgs += @('-t', "$ChunkSeconds")
    }

    $ffmpegArgs += @(
        '-ar', '16000',
        '-ac', '1',
        '-sample_fmt', 's16',
        '-f', 'segment',
        '-segment_time', "$ChunkSeconds",
        '-reset_timestamps', '1',
        $chunkPattern
    )

    $ffmpegStart = Get-Date
    Invoke-LoggedProcess `
        -Exe $FfmpegExe `
        -Arguments $ffmpegArgs `
        -OutputLog (Join-Path $LogsDir 'ffmpeg.txt') `
        -FailureMessage 'ffmpeg failed.'
    $ffmpegDuration = (Get-Date) - $ffmpegStart
    Write-Log ('ffmpeg_seconds={0:N3}' -f $ffmpegDuration.TotalSeconds)

    $chunks = @(Get-ChildItem -LiteralPath $ChunksDir -Filter '*.wav' | Sort-Object Name)
    if ($chunks.Count -eq 0) {
        throw "ffmpeg produced no chunks."
    }

    $chunkTimings = @()
    for ($i = 0; $i -lt $chunks.Count; $i++) {
        $chunk = $chunks[$i]
        $chunkBase = [IO.Path]::GetFileNameWithoutExtension($chunk.Name)
        $outputBase = Join-Path $OutputDir $chunkBase
        $whisperLog = Join-Path $LogsDir "$chunkBase.whisper.txt"

        Write-Log "chunk_start index=$i file=$($chunk.FullName)"

        $whisperArgs = @(
            '-m', $ModelPath,
            '-f', $chunk.FullName,
            '-l', $Language,
            '-otxt',
            '-osrt',
            '-of', $outputBase
        )

        $whisperStart = Get-Date
        try {
            Invoke-LoggedProcess `
                -Exe $WhisperExe `
                -Arguments $whisperArgs `
                -OutputLog $whisperLog `
                -FailureMessage "whisper failed on chunk $i ($($chunk.FullName))."
        }
        catch {
            Write-Log "ERROR chunk=$i file=$($chunk.FullName) message=$($_.Exception.Message)"
            throw
        }

        $whisperDuration = (Get-Date) - $whisperStart
        $chunkTimings += [PSCustomObject]@{
            Index = $i
            File = $chunk.FullName
            Seconds = $whisperDuration.TotalSeconds
        }

        Write-Log ('chunk_done index={0} whisper_seconds={1:N3}' -f $i, $whisperDuration.TotalSeconds)

        Assert-FileExists -Path "$outputBase.txt" -Label "Whisper txt output for chunk $i"
        Assert-FileExists -Path "$outputBase.srt" -Label "Whisper srt output for chunk $i"
    }

    $fullTranscript = Join-Path $OutputDir 'full_transcript.txt'
    if (Test-Path -LiteralPath $fullTranscript) {
        Remove-Item -LiteralPath $fullTranscript -Force
    }

    $txtFiles = @(Get-ChildItem -LiteralPath $OutputDir -Filter 'chunk_*.txt' | Sort-Object Name)
    foreach ($txt in $txtFiles) {
        "===== $($txt.BaseName) =====" | Out-File -FilePath $fullTranscript -Append -Encoding utf8
        Get-Content -LiteralPath $txt.FullName | Out-File -FilePath $fullTranscript -Append -Encoding utf8
        "" | Out-File -FilePath $fullTranscript -Append -Encoding utf8
    }
    Write-Log "full_transcript=$fullTranscript"

    $totalDuration = (Get-Date) - $totalStart
    $totalWhisperSeconds = ($chunkTimings | Measure-Object -Property Seconds -Sum).Sum
    if ($null -eq $totalWhisperSeconds) {
        $totalWhisperSeconds = 0
    }

    Write-Log ('total_whisper_seconds={0:N3}' -f $totalWhisperSeconds)
    Write-Log ('total_seconds={0:N3}' -f $totalDuration.TotalSeconds)
    Write-Log "chunks_processed=$($chunkTimings.Count)"

    if ($MemoryMonitorJob) {
        Stop-Job -Job $MemoryMonitorJob -ErrorAction SilentlyContinue
        Remove-Job -Job $MemoryMonitorJob -Force -ErrorAction SilentlyContinue
        $MemoryMonitorJob = $null
    }

    if (Test-Path -LiteralPath $MemoryLogFile) {
        $memoryRows = @(Import-Csv -LiteralPath $MemoryLogFile)
        if ($memoryRows.Count -gt 0) {
            $peakSystemUsedMb = ($memoryRows | Measure-Object -Property used_mb -Maximum).Maximum
            $peakFfmpegMb = ($memoryRows | Measure-Object -Property ffmpeg_working_set_mb -Maximum).Maximum
            $peakWhisperMb = ($memoryRows | Measure-Object -Property whisper_working_set_mb -Maximum).Maximum
            $peakTargetMb = ($memoryRows | Measure-Object -Property total_target_process_mb -Maximum).Maximum
            Write-Log "memory_peak_system_used_mb=$peakSystemUsedMb"
            Write-Log "memory_peak_ffmpeg_working_set_mb=$peakFfmpegMb"
            Write-Log "memory_peak_whisper_working_set_mb=$peakWhisperMb"
            Write-Log "memory_peak_target_process_mb=$peakTargetMb"
        }
    }

    Write-Log "END"

    Write-Host ""
    Write-Host "Done."
    Write-Host "Run folder: $RunDir"
    Write-Host "Log: $LogFile"
    Write-Host "Transcript: $fullTranscript"
}
catch {
    if ($MemoryMonitorJob) {
        Stop-Job -Job $MemoryMonitorJob -ErrorAction SilentlyContinue
        Remove-Job -Job $MemoryMonitorJob -Force -ErrorAction SilentlyContinue
    }

    if (Test-Path -LiteralPath $LogsDir) {
        Write-Log "FATAL $($_.Exception.Message)"
    }

    Write-Error $_.Exception.Message
    exit 1
}
