param(
    [Parameter(Mandatory = $false)]
    [ValidateSet('study', 'english', 'chat')]
    [string]$Mode = 'study',

    [Parameter(Mandatory = $false)]
    [string]$Language = 'ja',         # ja / en / None(auto)

    [Parameter(Mandatory = $false)]
    [string]$ModelFile = '',          # "" = Qwen3.5-4B デフォルト

    [Parameter(Mandatory = $false)]
    [string]$AsrModel = 'small',

    [Parameter(Mandatory = $false)]
    [float]$SilenceThreshold = 0.015,

    [Parameter(Mandatory = $false)]
    [float]$SilenceDuration = 1.2,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokens = 150,

    [Parameter(Mandatory = $false)]
    [string]$VoicevoxUrl = 'http://localhost:50021',

    [Parameter(Mandatory = $false)]
    [int]$SpeakerId = 1    # 1=ずんだもん  3=ずんだもん(囁き)  8=春日部つむぎ 等
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root       = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$PyScript   = Join-Path $Root 'scripts\prototypes\voice_chat.py'
$ModelsDir  = Join-Path $Root 'models'
$DefaultModel = Join-Path $ModelsDir 'Qwen3.5-4B-Q4_K_M.gguf'

function Assert-FileExists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Label not found: $Path" }
}

Assert-FileExists -Path $VenvPython  -Label '.venv Python'
Assert-FileExists -Path $PyScript    -Label 'voice_chat.py'

if ($ModelFile -eq '') { $ModelFile = $DefaultModel }
Assert-FileExists -Path $ModelFile -Label 'GGUF model'

Write-Host ""
Write-Host "=== Voice Chat ==="
Write-Host "Mode       : $Mode"
Write-Host "Language   : $Language"
Write-Host "LLM        : $ModelFile"
Write-Host "ASR        : $AsrModel"
Write-Host "VOICEVOX   : $VoicevoxUrl  speaker=$SpeakerId"
Write-Host ""

$pyArgs = @(
    $PyScript,
    '--llm_model',         $ModelFile,
    '--asr_model',         $AsrModel,
    '--mode',              $Mode,
    '--language',          $Language,
    '--silence_threshold', "$SilenceThreshold",
    '--silence_duration',  "$SilenceDuration",
    '--max_tokens',        "$MaxTokens",
    '--voicevox_url',      $VoicevoxUrl,
    '--speaker_id',        "$SpeakerId"
)

& $VenvPython @pyArgs
