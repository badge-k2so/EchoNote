param(
    [Parameter(Mandatory = $false)]
    [string]$InitialDirectory = "",

    [Parameter(Mandatory = $false)]
    [string]$TestPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Windows.Forms

if ($TestPath) {
    if (-not (Test-Path -LiteralPath $TestPath -PathType Leaf)) {
        throw "Test audio file not found"
    }
    [Console]::WriteLine([System.IO.Path]::GetFullPath($TestPath))
    exit 0
}

$dialog = [System.Windows.Forms.OpenFileDialog]::new()
$dialog.AutoUpgradeEnabled = $false
$dialog.Title = 'Select an audio recording'
$dialog.Filter = 'Audio files|*.ogg;*.wav;*.mp3;*.m4a;*.opus;*.flac;*.aac;*.wma|All files|*.*'
$dialog.CheckFileExists = $true
$dialog.CheckPathExists = $true
$dialog.Multiselect = $false
$dialog.RestoreDirectory = $true
if ($InitialDirectory -and (Test-Path -LiteralPath $InitialDirectory -PathType Container)) {
    $dialog.InitialDirectory = $InitialDirectory
}

if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::WriteLine($dialog.FileName)
}
