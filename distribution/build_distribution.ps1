# OtoWeave プロトタイプ配布パッケージの組み立て（開発機で実行・要ネット接続）
# 出力: dist\OtoWeave_ProtoTest_<日付>\ … このフォルダごとUSB等でテスト機へコピーする
param(
    [string]$OutputRoot = "dist",
    # Python インストーラーは既定で同梱する（オフラインのテスト機で
    # Python 3.12 が無いと詰むため）。除外したい場合のみ指定する。
    [switch]$SkipPythonInstaller,
    [switch]$SkipWheels,
    [switch]$SkipModels,
    # USBメモリ等の容量を節約したい場合の選択肢: 要約用の4Bを同梱しない。
    # 配布は標準版1本でよい（setup_test_pc.ps1 が端末のメモリ・空き容量を
    # 自動判定し、条件を満たさない端末では4Bを削除してLite相当にする）。
    # 4Bが無い環境ではAI要約の生成UIが自動的に「準備中」表示になる
    # （2Bでの要約生成は行わない）。2Bはチャット用として同梱を続ける。
    [switch]$Lite
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$DistName = "OtoWeave_ProtoTest_" + (Get-Date -Format 'yyyyMMdd')
if ($Lite) { $DistName += "_Lite" }
$Dist = Join-Path (Join-Path $ProjectRoot $OutputRoot) $DistName
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

function Write-Step([string]$Message) {
    Write-Host "== $Message" -ForegroundColor Cyan
}

Write-Step "出力先: $Dist"
New-Item -ItemType Directory -Force $Dist | Out-Null

# ------------------------------------------------------------------
# 1. アプリ本体（コード）
# ------------------------------------------------------------------
Write-Step "アプリコードをコピー"
robocopy "$ProjectRoot\otoweave_app" "$Dist\otoweave_app" /MIR /XD __pycache__ /NFL /NDL /NJH /NJS | Out-Null
New-Item -ItemType Directory -Force "$Dist\scripts" | Out-Null
foreach ($file in @(
    'production\windows_audio_file_dialog.ps1',
    'production\record_filename.py',
    'production\template_summarize.py',
    'production\school_hybrid_postprocess.py',
    'prototypes\diarization_prototype.py'
)) {
    Copy-Item "$ProjectRoot\scripts\$file" "$Dist\scripts\" -Force
}
if (Test-Path "$ProjectRoot\prompts") {
    robocopy "$ProjectRoot\prompts" "$Dist\prompts" /MIR /NFL /NDL /NJH /NJS | Out-Null
}

# ------------------------------------------------------------------
# 2. ffmpeg
# ------------------------------------------------------------------
Write-Step "ffmpeg をコピー"
New-Item -ItemType Directory -Force "$Dist\engines\ffmpeg" | Out-Null
Copy-Item "$ProjectRoot\engines\ffmpeg\ffmpeg.exe" "$Dist\engines\ffmpeg\" -Force

# ------------------------------------------------------------------
# 3. モデル（配布対象のみ / モデル構成.md 参照）
# ------------------------------------------------------------------
if (-not $SkipModels) {
    Write-Step "モデルをコピー（約5.3GB・数分かかります）"
    New-Item -ItemType Directory -Force "$Dist\models" | Out-Null
    foreach ($dir in @(
        'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8',
        'speechbrain-lang-id-voxlingua107-ecapa-onnx',
        'diarization'
    )) {
        robocopy "$ProjectRoot\models\$dir" "$Dist\models\$dir" /MIR /NFL /NDL /NJH /NJS | Out-Null
    }
    $ggufFiles = @('Qwen3.5-2B-Q4_K_M.gguf')
    if (-not $Lite) { $ggufFiles += 'Qwen3.5-4B-Q4_K_M.gguf' }
    foreach ($file in $ggufFiles) {
        Copy-Item "$ProjectRoot\models\$file" "$Dist\models\" -Force
    }

    Write-Step "ReazonSpeech K2 (HFキャッシュ) をコピー"
    $HfSource = "$env:USERPROFILE\.cache\huggingface\hub\models--reazon-research--reazonspeech-k2-v2"
    if (-not (Test-Path $HfSource)) { throw "ReazonSpeech のキャッシュが見つかりません: $HfSource" }
    robocopy $HfSource "$Dist\hf-cache\hub\models--reazon-research--reazonspeech-k2-v2" /MIR /NFL /NDL /NJH /NJS | Out-Null
}

# ------------------------------------------------------------------
# 4. Python ライブラリ（wheel をオフライン同梱）
# ------------------------------------------------------------------
if (-not $SkipWheels) {
    Write-Step "wheel をダウンロード（要ネット接続）"
    New-Item -ItemType Directory -Force "$Dist\wheels" | Out-Null
    & $VenvPython -m pip download `
        -r "$ProjectRoot\distribution\requirements_dist.txt" `
        --only-binary=:all: `
        --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cpu" `
        -d "$Dist\wheels" --quiet
    if ($LASTEXITCODE -ne 0) { throw "wheel のダウンロードに失敗しました。" }

    Write-Step "ReazonSpeech k2-asr パッケージを wheel 化"
    & $VenvPython -m pip wheel `
        "$ProjectRoot\_downloads\ReazonSpeech-master\pkg\k2-asr" `
        --no-deps -w "$Dist\wheels" --quiet
    if ($LASTEXITCODE -ne 0) { throw "reazonspeech-k2-asr の wheel 化に失敗しました。" }
}

# ------------------------------------------------------------------
# 5. セットアップ・起動・検証・ドキュメント
# ------------------------------------------------------------------
Write-Step "セットアップ/起動スクリプトとドキュメントをコピー"
Copy-Item "$ProjectRoot\distribution\setup_test_pc.ps1" $Dist -Force
Copy-Item "$ProjectRoot\distribution\setup.bat" $Dist -Force
Copy-Item "$ProjectRoot\distribution\OtoWeaveを起動.bat" $Dist -Force
Copy-Item "$ProjectRoot\distribution\verify_setup.py" $Dist -Force
Copy-Item "$ProjectRoot\distribution\verify_offline.ps1" $Dist -Force
Copy-Item "$ProjectRoot\distribution\requirements_dist.txt" "$Dist\requirements.txt" -Force
Copy-Item "$ProjectRoot\distribution\はじめにお読みください.txt" $Dist -Force
New-Item -ItemType Directory -Force "$Dist\docs" | Out-Null
Copy-Item "$ProjectRoot\distribution\docs\*" "$Dist\docs\" -Force

# ------------------------------------------------------------------
# 6. Python インストーラー同梱（既定で同梱 / -SkipPythonInstaller で除外）
#    テスト機はオフラインのことがあるため、Python 3.12 が入っていない
#    端末でも配布フォルダだけでセットアップを完了できるようにする。
# ------------------------------------------------------------------
if (-not $SkipPythonInstaller) {
    Write-Step "Python 3.12 インストーラーを同梱"
    $PyUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
    $PyInstallerName = Split-Path $PyUrl -Leaf
    $PyCacheDir = Join-Path $ProjectRoot '_downloads'
    $PyCached = Join-Path $PyCacheDir $PyInstallerName

    if (-not (Test-Path $PyCached)) {
        Write-Host "   ローカルキャッシュに無いためダウンロードします: $PyUrl"
        try {
            New-Item -ItemType Directory -Force $PyCacheDir | Out-Null
            Invoke-WebRequest $PyUrl -OutFile $PyCached
        } catch {
            if (Test-Path $PyCached) { Remove-Item $PyCached -Force }
            throw (
                "Python 3.12 インストーラーを用意できませんでした（配布に必須です）。`n" +
                "  入手元: $PyUrl`n" +
                "  置き場所: $PyCached`n" +
                "上記URLから手動でダウンロードして置き場所に保存するか、ネット接続を確認して再実行してください。`n" +
                "（同梱せずにビルドする場合のみ -SkipPythonInstaller を指定できますが、`n" +
                "  Python 3.12 が無いテスト機ではセットアップできなくなります）`n" +
                "元のエラー: $($_.Exception.Message)"
            )
        }
    }
    Copy-Item $PyCached (Join-Path $Dist $PyInstallerName) -Force
    Write-Host "   同梱: $PyInstallerName"
}

# ------------------------------------------------------------------
# サイズ報告
# ------------------------------------------------------------------
$size = (Get-ChildItem $Dist -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ""
Write-Host ("完成: {0}  ({1:N1} GB)" -f $Dist, ($size / 1GB)) -ForegroundColor Green
Write-Host "このフォルダごとUSBメモリ等でテスト機へコピーし、"
Write-Host "テスト機で setup_test_pc.ps1 を実行してください。"
