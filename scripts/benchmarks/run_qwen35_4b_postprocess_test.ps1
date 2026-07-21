param(
    [Parameter(Mandatory = $false)]
    [string]$TranscriptFile = ".\runs\sample_transcript.txt",

    [Parameter(Mandatory = $false)]
    [int]$NCtx = 12288,

    [Parameter(Mandatory = $false)]
    [int]$MaxCharsPerBatch = 5000,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokensPart = 1200,

    [Parameter(Mandatory = $false)]
    [int]$MaxTokensFinal = 1800
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$Runner = Join-Path $Root 'scripts\production\run_school_hybrid_postprocess.ps1'
$Model = Join-Path $Root 'models\Qwen3.5-4B-Q4_K_M.gguf'

if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Qwen3.5-4B model not found: $Model"
}

& $Runner `
    -TranscriptFile $TranscriptFile `
    -ModelFile $Model `
    -NCtx $NCtx `
    -MaxCharsPerBatch $MaxCharsPerBatch `
    -MaxTokensPart $MaxTokensPart `
    -MaxTokensFinal $MaxTokensFinal
