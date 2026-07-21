# OtoWeave AI要約 上位棚モデル(9B)の追加ダウンロード（任意・メモリ16GBクラスの端末向け）
# 配布フォルダにこのファイルをコピーした状態で実行します（要インターネット接続）。
#
# 上位棚(9B)モデルはサイズが大きい（約5.3GB）ため、配布フォルダには
# 既定で同梱されていません。メモリ16GBクラスの端末では、これを追加する
# ことでAI要約に9Bモデルが使われるようになります（4Bより高品質・低速）。
# メモリ15GB以下の端末では9Bはそもそも使われず4Bにフォールバックするため、
# ダウンロードしても意味がありません（setup_test_pc.ps1 が自動的に削除します）。
#
# 使い方:
#   このファイルを右クリック→「PowerShellで実行」、
#   または配布フォルダで次を実行します:
#     powershell -ExecutionPolicy Bypass -File .\download_9b_model.ps1
#
# 途中で失敗した場合は、もう一度このスクリプトを実行してください
# （.part ファイルから続きをダウンロードします）。
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSCommandPath
Set-Location $Root

function Write-Step([string]$Message) {
    Write-Host "== $Message" -ForegroundColor Cyan
}

$ModelUrl = "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf"
$ModelDir = Join-Path $Root 'models'
$ModelName = 'Qwen3.5-9B-Q4_K_M.gguf'
$ModelPath = Join-Path $ModelDir $ModelName
$PartPath = "$ModelPath.part"
# 本体は約5.29GBのため、5GB未満は途中で切れたダウンロードとみなす
$MinSizeBytes = [long](5 * 1GB)
$MinFreeDiskGB = 6.0

Write-Host "OtoWeave AI要約 上位棚モデル(9B)の追加ダウンロード"
Write-Host "入手元: $ModelUrl"
Write-Host ""

if ((Test-Path $ModelPath) -and (-not $Force)) {
    $existingGB = (Get-Item $ModelPath).Length / 1GB
    Write-Host ("既に models\{0} があります（{1:N2}GB）。" -f $ModelName, $existingGB) -ForegroundColor Yellow
    Write-Host "作り直す場合は -Force を指定してもう一度実行してください。"
    exit 0
}

# 1. この端末の搭載メモリを確認し、9Bが実際に使われるかどうかを案内する
Write-Step "この端末の搭載メモリを確認しています"
$RamBytes = [long]0
try {
    $RamBytes = [long](Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
} catch { }
$RamGB = [Math]::Round($RamBytes / 1GB, 1)
# AI要約(9B・上位棚)を使うためのメモリ下限: 15GB超
# ※ この閾値を変えるときは otoweave_app/llm_chat.py の
#    _HIGH_MEMORY_THRESHOLD_BYTES = int(15 * 1024**3) も合わせること。
#    setup_test_pc.ps1 も同じ閾値で9Bの保持/削除を自動判定している。
$RamThresholdBytes = [long](15 * 1GB)

if ($RamBytes -gt 0) {
    Write-Host ("   搭載メモリ: {0:N1}GB（9Bを使うための目安: 15GB超、推奨16GBクラス）" -f $RamGB)
} else {
    Write-Host "   搭載メモリを取得できませんでした。" -ForegroundColor Yellow
}

if ($RamBytes -le $RamThresholdBytes) {
    Write-Host ""
    Write-Host "この端末では9Bは使用されません（4Bが使われます）。" -ForegroundColor Yellow
    if ($RamBytes -gt 0) {
        Write-Host ("   理由: 搭載メモリが{0:N1}GBで、9Bに必要な目安（15GB超）を満たしません。" -f $RamGB)
    } else {
        Write-Host "   理由: 搭載メモリを取得できませんでした。"
    }
    Write-Host "   （setup_test_pc.ps1 を実行すると、9Bモデルは自動的に削除されます）"
    $answer = Read-Host "それでも約5.3GBをダウンロードしますか？ (y/N)"
    if ($answer -notmatch '^[Yy]') {
        Write-Host "ダウンロードを中止しました。"
        exit 0
    }
}

# 2. ディスクの空き容量を確認
Write-Step "ディスクの空き容量を確認しています"
$freeGB = -1.0
try {
    $drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($Root))
    $freeGB = $drive.AvailableFreeSpace / 1GB
} catch { }
if ($freeGB -ge 0) {
    Write-Host ("   空き容量: {0:N1}GB（必要: {1:N1}GB以上）" -f $freeGB, $MinFreeDiskGB)
    if ($freeGB -lt $MinFreeDiskGB) {
        Write-Host ""
        Write-Host ("空き容量が足りないため、ダウンロードを中止しました。あと約{0:N1}GBの空きが必要です。" -f ($MinFreeDiskGB - $freeGB)) -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "   空き容量を確認できませんでした。そのまま続行します。" -ForegroundColor Yellow
}

# 3. ダウンロード（.part に保存し、検証OKで本来のファイル名にリネームする。
#    途中で失敗した場合は、もう一度このスクリプトを実行すると .part から再開する）
Write-Step "ダウンロードしています（約5.3GB・回線速度により数分〜数十分かかります）"
New-Item -ItemType Directory -Force $ModelDir | Out-Null
$curlCmd = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curlCmd) {
    throw "curl.exe が見つかりません（Windows 10 1803以降には標準搭載されています）。"
}
& curl.exe -L --fail --retry 5 --retry-delay 5 -C - -o "$PartPath" "$ModelUrl"
if ($LASTEXITCODE -ne 0) {
    throw (
        "ダウンロードに失敗しました（curl.exe 終了コード: $LASTEXITCODE）。`n" +
        "  ネット接続を確認し、もう一度このスクリプトを実行してください（$PartPath から再開します）。`n" +
        "  入手元: $ModelUrl"
    )
}

# 4. ダウンロードしたファイルのサイズを検証してから本来のファイル名にする
Write-Step "ダウンロードしたファイルを検証しています"
if (-not (Test-Path $PartPath)) {
    throw "ダウンロードされたファイルが見つかりません: $PartPath"
}
$downloadedBytes = (Get-Item $PartPath).Length
$downloadedGB = $downloadedBytes / 1GB
Write-Host ("   サイズ: {0:N2}GB" -f $downloadedGB)
if ($downloadedBytes -lt $MinSizeBytes) {
    throw (
        ("ダウンロードしたファイルが小さすぎます（{0:N2}GB、想定は約5.29GB）。`n" -f $downloadedGB) +
        "  ダウンロードが途中で切れた可能性があります。もう一度このスクリプトを実行すると続きから再開します。`n" +
        "  （$PartPath は削除せずそのまま残しています）"
    )
}

Move-Item $PartPath $ModelPath -Force
Write-Host ""
Write-Host ("完了: models\{0} （{1:N2}GB）" -f $ModelName, ((Get-Item $ModelPath).Length / 1GB)) -ForegroundColor Green
Write-Host "この後 setup_test_pc.ps1 を実行すると、この端末のメモリに応じて9Bを残す/削除するか自動判定されます。"
Write-Host "（セットアップ済みの場合は、そのまま起動すればメモリ15GB超の端末でAI要約が9Bを優先利用します）"
Read-Host "Enter キーで終了"
