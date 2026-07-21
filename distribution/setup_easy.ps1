# OtoWeave 「git clone から誰でも試せる」簡単セットアップ
#
# 使い方（PowerShellで）:
#   git clone https://github.com/badge-k2so/otoweave.git
#   cd otoweave
#   .\distribution\setup_easy.ps1
#
# このスクリプトは、リポジトリ直下（このフォルダの一つ上）に対して:
#   1) Python 3.12 を確認（無ければ winget でインストールを試みる）
#   2) リポジトリ直下に .venv を作成
#   3) 実行に必要なライブラリを pip でインストール（要インターネット接続）
#   4) この端末のメモリ・空き容量を確認（AI要約4Bモデルの要否を自動判定）
#   5) AIモデルをインターネットから自動ダウンロード（数GB・時間がかかります）
#   6) verify_setup.py でセットアップを検証
#   7) 起動方法を案内
# を行います。初回はモデルのダウンロードで時間がかかりますが、
# 一度完了すれば「run_otoweave.ps1」だけで起動できます。
#
# 途中で失敗しても、このスクリプトをもう一度実行すれば、
# 済んだ部分はスキップして続きから再開します（ダウンロード済みファイルは
# サイズを確認してから再利用し、壊れていた場合だけ再取得します）。
#
# 既存の「オフライン配布パッケージ」（setup.bat / setup_test_pc.ps1）は、
# インターネットに繋がらないテスト機向けの別経路として、そのまま使えます。
# このスクリプトはそれとは別の「ネット接続がある人向けの簡単セットアップ」です。
#
# オプション:
#   -Lite       : AI要約用の4Bモデルをダウンロードしない（メモリ判定に関係なく）
#   -SkipModels : モデルのダウンロードをスキップする（ライブラリのみ準備・検証用）
#   -SkipVerify : 最後の verify_setup.py を実行しない
#   -Yes        : ダウンロード量の確認プロンプトを省略して進める
param(
    [switch]$Lite,
    [switch]$SkipModels,
    [switch]$SkipVerify,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$DistDir = Split-Path -Parent $PSCommandPath
$RepoRoot = Split-Path -Parent $DistDir
Set-Location $RepoRoot

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "== $Message" -ForegroundColor Cyan
}

function Write-Info([string]$Message) {
    Write-Host "   $Message"
}

function Format-GB([double]$bytes) {
    return ("{0:N2}GB" -f ($bytes / 1GB))
}

# ------------------------------------------------------------------
# 0. curl.exe / tar.exe の確認（どちらも Windows 10 1803 以降に標準搭載）
# ------------------------------------------------------------------
Write-Step "必要なコマンドを確認しています"
$CurlCmd = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $CurlCmd) {
    throw "curl.exe が見つかりません（Windows 10 1803以降には標準搭載されています。最新のWindows Updateを適用してください）。"
}
$TarCmd = Get-Command tar.exe -ErrorAction SilentlyContinue
if (-not $TarCmd) {
    throw "tar.exe が見つかりません（Windows 10 1803以降には標準搭載されています。最新のWindows Updateを適用してください）。"
}
Write-Info "curl.exe / tar.exe を確認しました。"

# ------------------------------------------------------------------
# 1. Python 3.12 を確認（無ければ winget → 見つからなければ手動案内）
# ------------------------------------------------------------------
Write-Step "Python 3.12 を確認しています"

function Find-Python312 {
    foreach ($candidate in @(
        @('py', @('-3.12', '-c', 'import sys;print(sys.executable)')),
        @('python', @('-c', 'import sys;print(sys.executable)'))
    )) {
        try {
            $exe = & $candidate[0] @($candidate[1]) 2>$null | Select-Object -First 1
            if ($exe -and (Test-Path $exe)) {
                $version = & $exe -c "import sys;print('{0}.{1}'.format(*sys.version_info))" 2>$null
                if ($version -eq '3.12') { return $exe }
            }
        } catch { }
    }
    foreach ($fallback in @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "C:\Program Files\Python312\python.exe"
    )) {
        if (Test-Path $fallback) {
            $version = & $fallback -c "import sys;print('{0}.{1}'.format(*sys.version_info))" 2>$null
            if ($version -eq '3.12') { return $fallback }
        }
    }
    return $null
}

$Python = Find-Python312
if (-not $Python) {
    $WingetCmd = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($WingetCmd) {
        Write-Info "Python 3.12 が見つからないため、winget でインストールを試みます。"
        try {
            & winget.exe install --id Python.Python.3.12 -e --silent `
                --accept-package-agreements --accept-source-agreements
        } catch {
            Write-Info ("winget でのインストールに失敗しました: {0}" -f $_.Exception.Message)
        }
        $Python = Find-Python312
    }
}
if (-not $Python) {
    throw (
        "Python 3.12 が見つかりません。以下のいずれかの方法でインストールしてから、" +
        "もう一度このスクリプトを実行してください。`n" +
        "  - https://www.python.org/downloads/ から Python 3.12 をダウンロードしてインストール" +
        "（『Add python.exe to PATH』にチェック）`n" +
        "  - または管理者権限のPowerShellで: winget install --id Python.Python.3.12 -e"
    )
}
Write-Info "Python: $Python"

# ------------------------------------------------------------------
# 2. リポジトリ直下に仮想環境を作成
# ------------------------------------------------------------------
Write-Step "アプリ専用のPython環境を作成しています（$RepoRoot\.venv）"
if (-not (Test-Path "$RepoRoot\.venv\Scripts\python.exe")) {
    & $Python -m venv "$RepoRoot\.venv"
    if ($LASTEXITCODE -ne 0) { throw "仮想環境の作成に失敗しました。" }
} else {
    Write-Info "既に作成済みです。"
}
$VenvPython = "$RepoRoot\.venv\Scripts\python.exe"

# ------------------------------------------------------------------
# 3. ライブラリを pip でインストール（要インターネット接続）
# ------------------------------------------------------------------
Write-Step "Pythonライブラリをインストールしています（初回は数分かかります）"
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install `
    -r "$DistDir\requirements_dist.txt" `
    --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cpu" `
    --quiet
if ($LASTEXITCODE -ne 0) { throw "ライブラリのインストールに失敗しました（インターネット接続を確認し、再実行してください）。" }

# reazonspeech-k2-asr（日本語文字起こしの補助パッケージ）は PyPI に無いため、
# このプロジェクトのGitHub Releaseからwheelを取得してインストールする。
Write-Info "reazonspeech-k2-asr パッケージを確認しています"
$ReazonWheelInstalled = $false
try {
    & $VenvPython -c "import reazonspeech.k2.asr" 2>$null
    if ($LASTEXITCODE -eq 0) { $ReazonWheelInstalled = $true }
} catch { }
if (-not $ReazonWheelInstalled) {
    $ReazonWheelUrl = "https://github.com/badge-k2so/otoweave/releases/download/v0.1.0-beta/reazonspeech_k2_asr-3.0.0-py3-none-any.whl"
    $ReazonWheelPath = Join-Path $env:TEMP "reazonspeech_k2_asr-3.0.0-py3-none-any.whl"
    Write-Info "ダウンロード中: $ReazonWheelUrl"
    & curl.exe -L --fail --retry 5 --retry-delay 5 -o "$ReazonWheelPath" "$ReazonWheelUrl"
    if ($LASTEXITCODE -ne 0) {
        throw (
            "reazonspeech-k2-asr パッケージのダウンロードに失敗しました。`n" +
            "  入手元: $ReazonWheelUrl`n" +
            "  ネット接続を確認し、もう一度このスクリプトを実行してください。"
        )
    }
    & $VenvPython -m pip install --no-deps "$ReazonWheelPath" --quiet
    if ($LASTEXITCODE -ne 0) { throw "reazonspeech-k2-asr パッケージのインストールに失敗しました。" }
}
Write-Info "ライブラリの準備が完了しました。"

# ------------------------------------------------------------------
# 4. この端末のメモリ・空き容量を確認（AI要約4Bモデルの要否を自動判定）
#    ※ この閾値は otoweave_app/llm_chat.py の
#       _LOW_MEMORY_THRESHOLD_BYTES = int(11.5 * 1024**3) と合わせている。
# ------------------------------------------------------------------
Write-Step "この端末のスペックを確認しています"
$RamBytes = [long]0
try {
    $RamBytes = [long](Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
} catch { }
$RamGB = [Math]::Round($RamBytes / 1GB, 1)
$RamThresholdBytes = [long](11.5 * 1GB)
$RamEnoughFor4B = ($RamBytes -gt $RamThresholdBytes)
if ($RamBytes -gt 0) {
    Write-Info ("搭載メモリ: {0:N1}GB（AI要約4Bモデルの目安: 11.5GB超）" -f $RamGB)
} else {
    Write-Info "搭載メモリを取得できませんでした（安全のため4Bモデルはダウンロードしません）。"
}
$Download4B = (-not $Lite) -and $RamEnoughFor4B
if ($Lite) {
    Write-Info "-Lite の指定により、AI要約4Bモデルはダウンロードしません（AIチャットの2Bのみ使用します）。"
} elseif (-not $RamEnoughFor4B) {
    Write-Info "この端末はメモリが11.5GB以下のため、AI要約4Bモデルはダウンロードしません（AI要約は『じゅんび中』表示になります。チャットは2Bで使えます）。"
} else {
    Write-Info "この端末はメモリが十分なため、AI要約4Bモデルもダウンロードします。"
}

$freeGB = -1.0
try {
    $drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($RepoRoot))
    $freeGB = $drive.AvailableFreeSpace / 1GB
} catch { }
if ($freeGB -ge 0) {
    Write-Info ("ドライブの空き容量: {0:N1}GB" -f $freeGB)
}

# ------------------------------------------------------------------
# 5. AIモデルのダウンロード
#    既存ファイルはサイズを検証してスキップする（再実行安全・途中再開OK）。
# ------------------------------------------------------------------

function Test-ValidFile([string]$Path, [long]$MinBytes) {
    if (-not (Test-Path $Path -PathType Leaf)) { return $false }
    return ((Get-Item $Path).Length -ge $MinBytes)
}

function Invoke-Download([string]$Url, [string]$DestPath, [long]$MinBytes, [string]$Label) {
    if (Test-ValidFile $DestPath $MinBytes) {
        Write-Info "[済] $Label （既存ファイルを使用）"
        return
    }
    New-Item -ItemType Directory -Force (Split-Path -Parent $DestPath) | Out-Null
    $PartPath = "$DestPath.part"
    Write-Info "取得中: $Label"
    & curl.exe -L --fail --retry 5 --retry-delay 5 -C - -o "$PartPath" "$Url"
    if ($LASTEXITCODE -ne 0) {
        throw (
            "$Label のダウンロードに失敗しました（curl.exe 終了コード: $LASTEXITCODE）。`n" +
            "  入手元: $Url`n" +
            "  ネット接続を確認し、もう一度このスクリプトを実行してください（続きから再開します）。"
        )
    }
    if (-not (Test-ValidFile $PartPath $MinBytes)) {
        $gotBytes = 0
        if (Test-Path $PartPath) { $gotBytes = (Get-Item $PartPath).Length }
        throw (
            ("$Label のダウンロードが不完全です（{0:N1}MB / 想定{1:N1}MB以上）。`n" -f ($gotBytes / 1MB), ($MinBytes / 1MB)) +
            "  もう一度このスクリプトを実行すると続きから再開します。"
        )
    }
    Move-Item $PartPath $DestPath -Force
    Write-Info ("完了: {0} （{1:N1}MB）" -f $Label, ((Get-Item $DestPath).Length / 1MB))
}

# ダウンロード計画（未取得分のみ集計して事前表示する）
$ModelsDir = Join-Path $RepoRoot 'models'
$ParakeetDir = Join-Path $ModelsDir 'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8'
$SpeechBrainDir = Join-Path $ModelsDir 'speechbrain-lang-id-voxlingua107-ecapa-onnx'
$DiarizationDir = Join-Path $ModelsDir 'diarization'
$EngineFfmpegDir = Join-Path $RepoRoot 'engines\ffmpeg'

$Plan = @()
$Plan += @{ Label = 'ffmpeg.exe';                     Path = "$EngineFfmpegDir\ffmpeg.exe";                MinBytes = 50MB;  Required = $true;  Kind = 'ffmpeg' }
$Plan += @{ Label = '日本語ASR ReazonSpeech K2 v2';    Path = '__reazonspeech__';                            MinBytes = 150MB; Required = $true;  Kind = 'reazonspeech' }
$Plan += @{ Label = '英語ASR Parakeet (encoder)';      Path = "$ParakeetDir\encoder.int8.onnx";              MinBytes = 600MB; Required = $true;  Kind = 'curl'; Url = 'https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/encoder.int8.onnx' }
$Plan += @{ Label = '英語ASR Parakeet (decoder)';      Path = "$ParakeetDir\decoder.int8.onnx";              MinBytes = 5MB;   Required = $true;  Kind = 'curl'; Url = 'https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/decoder.int8.onnx' }
$Plan += @{ Label = '英語ASR Parakeet (joiner)';       Path = "$ParakeetDir\joiner.int8.onnx";               MinBytes = 1MB;   Required = $true;  Kind = 'curl'; Url = 'https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/joiner.int8.onnx' }
$Plan += @{ Label = '英語ASR Parakeet (tokens)';       Path = "$ParakeetDir\tokens.txt";                     MinBytes = 1KB;   Required = $true;  Kind = 'curl'; Url = 'https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/tokens.txt' }
$Plan += @{ Label = '言語判定 SpeechBrain (ONNX一式)';  Path = "$SpeechBrainDir\lang-id-ecapa.onnx.data";     MinBytes = 80MB;  Required = $true;  Kind = 'speechbrain' }
$Plan += @{ Label = '話者分離 pyannote segmentation';  Path = "$DiarizationDir\sherpa-onnx-pyannote-segmentation-3-0\model.onnx"; MinBytes = 5MB;   Required = $false; Kind = 'diarization_seg' }
$Plan += @{ Label = '話者分離 3D-Speaker embedding';   Path = "$DiarizationDir\3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"; MinBytes = 30MB; Required = $false; Kind = 'curl'; Url = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx' }
$Plan += @{ Label = 'AIチャット Qwen3.5-2B Q4_K_M';    Path = "$ModelsDir\Qwen3.5-2B-Q4_K_M.gguf";           MinBytes = 1200MB; Required = $false; Kind = 'curl'; Url = 'https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf' }
if ($Download4B) {
    $Plan += @{ Label = 'AI要約 Qwen3.5-4B Q4_K_M';    Path = "$ModelsDir\Qwen3.5-4B-Q4_K_M.gguf";           MinBytes = 2600MB; Required = $false; Kind = 'curl'; Url = 'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf' }
}

if (-not $SkipModels) {
    # 概算サイズ表（実際のファイルサイズが取れない項目は MinBytes を目安として使う）
    $KnownSizes = @{
        'ffmpeg.exe' = 110MB
        '日本語ASR ReazonSpeech K2 v2' = 153MB
        '英語ASR Parakeet (encoder)' = 622MB
        '英語ASR Parakeet (decoder)' = 7MB
        '英語ASR Parakeet (joiner)' = 2MB
        '英語ASR Parakeet (tokens)' = 1MB
        '言語判定 SpeechBrain (ONNX一式)' = 76MB
        '話者分離 pyannote segmentation' = 7MB
        '話者分離 3D-Speaker embedding' = 38MB
        'AIチャット Qwen3.5-2B Q4_K_M' = 1193MB
        'AI要約 Qwen3.5-4B Q4_K_M' = 2552MB
    }

    Write-Step "ダウンロードするAIモデルを確認しています"
    $TotalBytes = 0.0
    $ToDownload = @()
    foreach ($item in $Plan) {
        $already = $false
        if ($item.Kind -eq 'reazonspeech') {
            # 後段で存在確認するため、ここでは概算のみ（HFキャッシュを直接見る）
            $reazonSnapshot = Get-ChildItem "$RepoRoot\hf-cache\hub\models--reazon-research--reazonspeech-k2-v2\snapshots" `
                -Filter 'encoder-epoch-99-avg-1.int8.onnx' -Recurse -ErrorAction SilentlyContinue
            $already = [bool]$reazonSnapshot
        } elseif ($item.Kind -eq 'speechbrain') {
            $already = (Test-ValidFile $item.Path $item.MinBytes) -and
                (Test-ValidFile "$SpeechBrainDir\lang-id-ecapa.onnx" 500KB) -and
                (Test-ValidFile "$SpeechBrainDir\labels.json" 100)
        } elseif ($item.Kind -eq 'diarization_seg') {
            $already = Test-ValidFile $item.Path $item.MinBytes
        } else {
            $already = Test-ValidFile $item.Path $item.MinBytes
        }
        if (-not $already) {
            $TotalBytes += $KnownSizes[$item.Label]
            $ToDownload += $item.Label
        }
    }

    if ($ToDownload.Count -eq 0) {
        Write-Info "すべてのモデルは既に揃っています。ダウンロードはスキップします。"
    } else {
        Write-Info ("未取得のモデル {0} 件・合計 約{1} をダウンロードします:" -f $ToDownload.Count, (Format-GB $TotalBytes))
        $i = 0
        foreach ($label in $ToDownload) {
            $i++
            Write-Info ("  {0}/{1}. {2} (約{3:N0}MB)" -f $i, $ToDownload.Count, $label, ($KnownSizes[$label] / 1MB))
        }
        Write-Info "（既に揃っているモデルは上の一覧に含めていません）"
        if (-not $Yes) {
            $answer = Read-Host "ダウンロードを開始しますか？ (Y/n)"
            if ($answer -match '^[Nn]') {
                Write-Host "ダウンロードを中止しました。もう一度実行すると、ここから再開できます。" -ForegroundColor Yellow
                exit 0
            }
        }

        Write-Step "AIモデルをダウンロードしています"
        $n = 0
        $total = $Plan.Count
        foreach ($item in $Plan) {
            $n++
            Write-Host ("[{0}/{1}] {2}" -f $n, $total, $item.Label) -ForegroundColor Cyan
            try {
                switch ($item.Kind) {
                    'curl' {
                        Invoke-Download -Url $item.Url -DestPath $item.Path -MinBytes $item.MinBytes -Label $item.Label
                    }
                    'ffmpeg' {
                        if (Test-ValidFile $item.Path $item.MinBytes) {
                            Write-Info "[済] $($item.Label) （既存ファイルを使用）"
                        } else {
                            $ZipUrl = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
                            $ZipPath = Join-Path $env:TEMP 'otoweave_ffmpeg_essentials.zip'
                            Invoke-Download -Url $ZipUrl -DestPath $ZipPath -MinBytes 50MB -Label 'ffmpeg (essentials zip)'
                            $ExtractDir = Join-Path $env:TEMP 'otoweave_ffmpeg_extract'
                            if (Test-Path $ExtractDir) { Remove-Item $ExtractDir -Recurse -Force }
                            New-Item -ItemType Directory -Force $ExtractDir | Out-Null
                            Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force
                            $found = Get-ChildItem $ExtractDir -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1
                            if (-not $found) { throw "ffmpeg.exe がダウンロードしたzipの中に見つかりませんでした。" }
                            New-Item -ItemType Directory -Force $EngineFfmpegDir | Out-Null
                            Copy-Item $found.FullName "$EngineFfmpegDir\ffmpeg.exe" -Force
                            Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue
                            Remove-Item $ExtractDir -Recurse -Force -ErrorAction SilentlyContinue
                            Write-Info ("完了: {0}" -f $item.Label)
                        }
                    }
                    'reazonspeech' {
                        $reazonSnapshot = Get-ChildItem "$RepoRoot\hf-cache\hub\models--reazon-research--reazonspeech-k2-v2\snapshots" `
                            -Filter 'encoder-epoch-99-avg-1.int8.onnx' -Recurse -ErrorAction SilentlyContinue
                        if ($reazonSnapshot) {
                            Write-Info "[済] $($item.Label) （既存ファイルを使用）"
                        } else {
                            Write-Info "取得中: 日本語ASR ReazonSpeech K2 v2（HuggingFace, int8のみ・約153MB）"
                            $env:HF_HOME = "$RepoRoot\hf-cache"
                            Remove-Item Env:\HF_HUB_OFFLINE -ErrorAction SilentlyContinue
                            $code = @'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    "reazon-research/reazonspeech-k2-v2",
    allow_patterns=["*.int8.onnx", "tokens.txt"],
)
print("OK")
'@
                            # Windows PowerShell 5.1 は `python -c "<文字列>"` の引数内の
                            # 二重引用符を落とすため、一時ファイル経由で実行する。
                            $TmpPy = Join-Path $env:TEMP ("otoweave_hf_dl_{0}.py" -f ([guid]::NewGuid().ToString('N')))
                            Set-Content -Path $TmpPy -Value $code -Encoding Ascii
                            $result = & $VenvPython $TmpPy
                            $dlExit = $LASTEXITCODE
                            Remove-Item $TmpPy -Force -ErrorAction SilentlyContinue
                            if ($dlExit -ne 0) {
                                throw (
                                    "ReazonSpeech K2 v2 のダウンロードに失敗しました。`n" +
                                    "  ネット接続を確認し、もう一度このスクリプトを実行してください。"
                                )
                            }
                            Write-Info "完了: 日本語ASR ReazonSpeech K2 v2"
                        }
                    }
                    'speechbrain' {
                        if ((Test-ValidFile "$SpeechBrainDir\lang-id-ecapa.onnx.data" 80MB) -and
                            (Test-ValidFile "$SpeechBrainDir\lang-id-ecapa.onnx" 500KB) -and
                            (Test-ValidFile "$SpeechBrainDir\labels.json" 100)) {
                            Write-Info "[済] $($item.Label) （既存ファイルを使用）"
                        } else {
                            $ZipUrl = 'https://github.com/badge-k2so/otoweave/releases/download/v0.1.0-beta/speechbrain-lang-id-voxlingua107-ecapa-onnx.zip'
                            $ZipPath = Join-Path $env:TEMP 'otoweave_speechbrain.zip'
                            Invoke-Download -Url $ZipUrl -DestPath $ZipPath -MinBytes 70MB -Label 'SpeechBrain (zip)'
                            $ExtractDir = Join-Path $env:TEMP 'otoweave_speechbrain_extract'
                            if (Test-Path $ExtractDir) { Remove-Item $ExtractDir -Recurse -Force }
                            New-Item -ItemType Directory -Force $ExtractDir | Out-Null
                            Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force
                            $srcDir = $ExtractDir
                            if (-not (Test-Path (Join-Path $srcDir 'lang-id-ecapa.onnx'))) {
                                $inner = Get-ChildItem $ExtractDir -Directory | Select-Object -First 1
                                if ($inner) { $srcDir = $inner.FullName }
                            }
                            New-Item -ItemType Directory -Force $SpeechBrainDir | Out-Null
                            foreach ($f in @('lang-id-ecapa.onnx', 'lang-id-ecapa.onnx.data', 'labels.json')) {
                                Copy-Item (Join-Path $srcDir $f) (Join-Path $SpeechBrainDir $f) -Force
                            }
                            Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue
                            Remove-Item $ExtractDir -Recurse -Force -ErrorAction SilentlyContinue
                            Write-Info ("完了: {0}" -f $item.Label)
                        }
                    }
                    'diarization_seg' {
                        if (Test-ValidFile $item.Path $item.MinBytes) {
                            Write-Info "[済] $($item.Label) （既存ファイルを使用）"
                        } else {
                            $TarUrl = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2'
                            $TarPath = Join-Path $env:TEMP 'sherpa-onnx-pyannote-segmentation-3-0.tar.bz2'
                            Invoke-Download -Url $TarUrl -DestPath $TarPath -MinBytes 5MB -Label '話者分離 pyannote segmentation (tar.bz2)'
                            $ExtractDir = Join-Path $env:TEMP 'otoweave_diarization_extract'
                            if (Test-Path $ExtractDir) { Remove-Item $ExtractDir -Recurse -Force }
                            New-Item -ItemType Directory -Force $ExtractDir | Out-Null
                            & tar.exe -xf $TarPath -C $ExtractDir
                            if ($LASTEXITCODE -ne 0) { throw "話者分離モデルの展開(tar)に失敗しました。" }
                            $found = Get-ChildItem $ExtractDir -Recurse -Filter 'model.onnx' | Select-Object -First 1
                            if (-not $found) { throw "model.onnx が展開後のフォルダに見つかりませんでした。" }
                            $DestDir = Join-Path $DiarizationDir 'sherpa-onnx-pyannote-segmentation-3-0'
                            New-Item -ItemType Directory -Force $DestDir | Out-Null
                            Copy-Item $found.FullName (Join-Path $DestDir 'model.onnx') -Force
                            Remove-Item $TarPath -Force -ErrorAction SilentlyContinue
                            Remove-Item $ExtractDir -Recurse -Force -ErrorAction SilentlyContinue
                            Write-Info ("完了: {0}" -f $item.Label)
                        }
                    }
                }
            } catch {
                if ($item.Required) {
                    throw
                } else {
                    Write-Host ("   [警告] {0} のダウンロードに失敗しました（任意のモデルのため続行します）: {1}" -f $item.Label, $_.Exception.Message) -ForegroundColor Yellow
                }
            }
        }
    }
} else {
    Write-Step "モデルのダウンロードをスキップしました（-SkipModels 指定）"
}

# ------------------------------------------------------------------
# 6. セットアップ検証（verify_setup.py）
#    verify_setup.py はリポジトリの配布パッケージ内で使う想定のため、
#    既定では自分のあるフォルダ（distribution）を基準に確認する。
#    git clone構成ではアプリ本体・models・engines がリポジトリ直下にあるため、
#    OTOWEAVE_VERIFY_ROOT 環境変数でその基準フォルダを上書きする
#    （verify_setup.py 側の対応は、この環境変数を読む1行のみ）。
# ------------------------------------------------------------------
$VerifyExit = 0
if (-not $SkipVerify) {
    Write-Step "セットアップを検証しています"
    $env:OTOWEAVE_VERIFY_ROOT = $RepoRoot
    $env:HF_HOME = "$RepoRoot\hf-cache"
    $env:PYTHONUTF8 = '1'
    & $VenvPython "$DistDir\verify_setup.py"
    $VerifyExit = $LASTEXITCODE
    Remove-Item Env:\OTOWEAVE_VERIFY_ROOT -ErrorAction SilentlyContinue
}

# ------------------------------------------------------------------
# 7. 起動方法の案内
# ------------------------------------------------------------------
Write-Step "セットアップが完了しました"
if ($VerifyExit -ne 0) {
    Write-Host "検証で NG がありました。上の表示（と $RepoRoot\setup_report.txt）を確認してください。" -ForegroundColor Yellow
    Write-Host "もう一度 .\distribution\setup_easy.ps1 を実行すると、済んだ部分はスキップして続きから再開します。"
} else {
    Write-Host "次のコマンドで起動できます:" -ForegroundColor Green
    Write-Host "  .\run_otoweave.ps1"
    Write-Host ""
    Write-Host "デモデータ入りで起動して画面を見るだけなら:"
    Write-Host "  .\run_otoweave.ps1 -Demo -DataRoot `".\runs\learning_access_demo`""
}
