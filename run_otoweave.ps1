param(
    [Parameter(Mandatory = $false)]
    [string]$DataRoot = "",

    [Parameter(Mandatory = $false)]
    [switch]$Demo
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSCommandPath
$Python = Join-Path $Root '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}

$arguments = @('-m', 'otoweave_app.main')
if ($DataRoot) { $arguments += @('--data-root', $DataRoot) }
if ($Demo) { $arguments += '--demo' }

Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $Root -WindowStyle Hidden
