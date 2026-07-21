# OtoWeave プロトタイプテスト機セットアップ
# このフォルダをテスト機へコピーした後、「setup.bat」をダブルクリック
# してください（インターネット接続は不要です）。
# ※ setup.bat は、スクリプト実行が制限された端末でも動くように
#    ExecutionPolicy Bypass でこのファイルを呼び出します。
#
# AI要約(4Bモデル)の自動判定:
#   セットアップ時にこの端末の搭載メモリと空き容量を調べて、
#   AI要約用の4Bモデル(models\Qwen3.5-4B-Q4_K_M.gguf)を
#   「残す（AI要約が使える）」か「削除して約2.5GBを解放する」かを
#   自動で決めます。強制したい場合は以下のオプションを使います。
#     -KeepSummaryModel   : 判定に関係なく4Bモデルを残す（テスト・検証向け）
#     -RemoveSummaryModel : 判定に関係なく4Bモデルを削除する（テスト・検証向け）
#
# 外部送信のブロック（任意・学校管理者向け）:
#     -BlockNetwork : Windowsファイアウォールに、このフォルダ配下の
#                     python.exe / pythonw.exe / ffmpeg.exe / llama-server.exe の
#                     「外部への送信をブロック」するルール（OtoWeave-NoNetwork-*）を
#                     作成します。「送信しない」を「送信できない」にする設定です。
#                     管理者権限が必要です（無い場合はこの処理だけスキップします）。
#                     端末内部の通信(127.0.0.1)には影響せず、アプリは全機能使えます。
param(
    [switch]$SkipVerify,
    [switch]$KeepSummaryModel,
    [switch]$RemoveSummaryModel,
    [switch]$BlockNetwork
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSCommandPath
Set-Location $Root

function Write-Step([string]$Message) {
    Write-Host "== $Message" -ForegroundColor Cyan
}

if ($KeepSummaryModel -and $RemoveSummaryModel) {
    throw "-KeepSummaryModel と -RemoveSummaryModel は同時に指定できません。"
}

# 0. インストール先ドライブの空き容量を確認（最初にチェックして途中で詰まらないようにする）
#    あわせて、この端末のスペック（搭載メモリ・空き容量）から
#    AI要約用の4Bモデルを残すかどうかを判定し、必要な空き容量に反映する。
Write-Step "ディスクの空き容量を確認しています"
# Lite版の判定: フォルダ名に _Lite が付く、または 4B モデルが同梱されていない
$Model4BPath = "$Root\models\Qwen3.5-4B-Q4_K_M.gguf"
$Model4BGB = 2.5   # 4Bモデルのおおよそのサイズ（削除すると解放される量）
$Has4B = Test-Path $Model4BPath
$IsLite = $false
$LiteKnown = $false
if ((Split-Path $Root -Leaf) -match '_Lite') {
    $IsLite = $true
    $LiteKnown = $true
} elseif (Test-Path "$Root\models") {
    $IsLite = -not $Has4B
    $LiteKnown = $true
}

# 物理メモリを取得（取得できない場合は 0 のまま = 安全側に倒して4Bを外す）
$RamBytes = [long]0
try {
    $RamBytes = [long](Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
} catch { }
$RamGB = [Math]::Round($RamBytes / 1GB, 1)
# AI要約(4B)を使うためのメモリ下限: 11.5GB 超
# ※ この閾値を変えるときは otoweave_app/llm_chat.py の
#    _LOW_MEMORY_THRESHOLD_BYTES = int(11.5 * 1024**3) も合わせること。
#    アプリ側も起動時に同じ閾値でAI要約の有効/無効を判定している。
$RamThresholdBytes = [long](11.5 * 1GB)
$RamEnough = ($RamBytes -gt $RamThresholdBytes)

# ドライブの空き容量と、コピー済みパッケージのサイズを取得
$freeGB = -1.0
$pkgGB = 0.0
$drive = $null
try {
    $drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($Root))
    $freeGB = $drive.AvailableFreeSpace / 1GB
    # このフォルダは既にコピー済みなので、その分は必要量から差し引く
    # （セットアップ自体の作成分＋動作用の余裕として最低2GBは要求する）
    try {
        $pkgGB = ((Get-ChildItem $Root -Recurse -File -ErrorAction SilentlyContinue |
            Measure-Object Length -Sum).Sum) / 1GB
    } catch { }
} catch { }

# AI要約(4B)モデルを残すかどうかの判定
#   優先順位: 4Bが無い（Lite版など） > 強制オプション > 自動判定（メモリ → 空き容量）
$Keep4B = $false
$SummaryReason = ''
if (-not $Has4B) {
    $SummaryReason = 'AI要約用の4Bモデルは同梱されていません（Lite版など）。AI要約は「じゅんび中」と表示されます。'
} elseif ($RemoveSummaryModel) {
    $SummaryReason = 'オプション -RemoveSummaryModel の指定により、AI要約用の4Bモデルを削除しました（じゅんび中と表示されます）。'
} elseif ($KeepSummaryModel) {
    $Keep4B = $true
    $SummaryReason = 'オプション -KeepSummaryModel の指定により、AI要約用の4Bモデルを残しました（メモリが足りない端末では、アプリ側の判定でAI要約が「じゅんび中」になることがあります）。'
} elseif ($RamBytes -le 0) {
    $SummaryReason = '搭載メモリを確認できなかったため、安全のためAI要約は使わない設定にしました（じゅんび中と表示されます）。'
} elseif (-not $RamEnough) {
    $SummaryReason = ("この端末はメモリが{0:N0}GBのため、AI要約は使わない設定にしました（じゅんび中と表示されます）。" -f $RamGB)
} elseif (($freeGB -ge 0) -and ($freeGB -lt [Math]::Max(2.0, 8.0 - $pkgGB))) {
    # メモリは足りているが、4Bを残したままではドライブの空きが足りない
    $SummaryReason = ("メモリは{0:N0}GBありますが、ドライブの空き容量に余裕がないため、AI要約は使わない設定にしました（じゅんび中と表示されます）。" -f $RamGB)
} else {
    $Keep4B = $true
    $SummaryReason = ("この端末はメモリが{0:N0}GBのため、AI要約が使えます。" -f $RamGB)
}

# 必要量（このフォルダ＋セットアップで作られる環境の合計。
#   4Bを残す標準構成 約8GB / 4Bを外す構成 約5.5GB / Lite版 約5GB）
if (-not $LiteKnown) {
    $TotalRequiredGB = 8.0   # 種類を判別できないときは安全側（標準版）で見積もる
} elseif ($IsLite) {
    $TotalRequiredGB = 5.0
} elseif ($Keep4B) {
    $TotalRequiredGB = 8.0
} else {
    $TotalRequiredGB = 8.0 - $Model4BGB   # 4Bはこの後削除して解放されるぶん必要量を下げる
}
if ($freeGB -ge 0) {
    $neededFreeGB = [Math]::Max(2.0, $TotalRequiredGB - $pkgGB)
    if (-not $LiteKnown) {
        Write-Host "   パッケージの種類（標準版/Lite版）を判別できなかったため、標準版（約8GB）として確認します。" -ForegroundColor Yellow
    }
    $edition = if ($IsLite) { "Lite版" } else { "標準版" }
    Write-Host ("   {0}: 全体で約{1:N1}GB使用（うちコピー済み {2:N1}GB）/ さらに約{3:N1}GBの空きが必要 / 現在の空き {4:N1}GB（ドライブ {5}）" -f `
        $edition, $TotalRequiredGB, $pkgGB, $neededFreeGB, $freeGB, $drive.Name)
    if ($freeGB -lt $neededFreeGB) {
        Write-Host ""
        Write-Host "空き容量が足りないため、セットアップを中止しました。" -ForegroundColor Red
        Write-Host ("   必要な空き容量: あと約{0:N1}GB（{1}は全体で約{2:N1}GB使います）" -f $neededFreeGB, $edition, $TotalRequiredGB) -ForegroundColor Red
        Write-Host ("   現在の空き容量: {0:N1}GB（ドライブ {1}）" -f $freeGB, $drive.Name) -ForegroundColor Red
        Write-Host ""
        Write-Host "不要なファイルを削除して空きを増やしてから、もう一度実行してください。"
        Write-Host "（この必要量は、端末に合わせてAI要約用モデルを外した場合の計算です）"
        # キー入力待ちは setup.bat 側の pause に任せる（二重の入力待ちを避ける）
        exit 1
    }
} else {
    # 空き容量を取得できない環境でもセットアップ自体は続行する
    Write-Host "   空き容量を確認できませんでした。そのまま続行します。" -ForegroundColor Yellow
}

# 1. 端末スペックの判定結果にもとづいて、AI要約(4B)モデルの構成を確定
Write-Step "この端末に合ったAI要約の設定を確認しています"
if ($RamBytes -gt 0) {
    Write-Host ("   搭載メモリ: {0:N1}GB（AI要約に必要なメモリ: 11.5GB超）" -f $RamGB)
} else {
    Write-Host "   搭載メモリを取得できませんでした。" -ForegroundColor Yellow
}
$SummaryAction = ''
if (-not $Has4B) {
    $SummaryAction = '同梱なし（変更なし）'
    Write-Host ("   " + $SummaryReason)
    Write-Host "   録音・文字起こし・読み上げ・AIへの質問はすべて使えます（故障ではありません）。"
} elseif ($Keep4B) {
    $SummaryAction = '残しました（AI要約 有効）'
    Write-Host ("   " + $SummaryReason) -ForegroundColor Green
} else {
    Remove-Item $Model4BPath -Force
    $SummaryAction = ("削除しました（約{0:N1}GBを解放・AI要約は「じゅんび中」）" -f $Model4BGB)
    Write-Host ("   " + $SummaryReason)
    Write-Host "   録音・文字起こし・読み上げ・AIへの質問はすべて使えます。"
    Write-Host ("   （AI要約用の4Bモデルを削除して、約{0:N1}GBの空きを増やしました）" -f $Model4BGB)
}

# 上位棚(9B)モデルが同梱されている場合の扱い（準備実装・最小限）。
#   build_distribution.ps1 は現時点で9Bを同梱しない方針だが、手動配置などで
#   models フォルダに置かれていた場合に備え、アプリ側と同じ判定基準
#   （高メモリ機のみ9Bを使う）に合わせて要否を決める。
#   ※ この閾値を変えるときは otoweave_app/llm_chat.py の
#      _HIGH_MEMORY_THRESHOLD_BYTES = int(15 * 1024**3) も合わせること。
$Model9BPath = "$Root\models\Qwen3.5-9B-Q4_K_M.gguf"
$Model9BGB = 5.5   # 9Bモデルのおおよそのサイズ（削除すると解放される量）
$Has9B = Test-Path $Model9BPath
$RamThreshold9BBytes = [long](15 * 1GB)
$Summary9BAction = '同梱なし（変更なし）'
if ($Has9B) {
    if ($RamBytes -gt $RamThreshold9BBytes) {
        $Summary9BAction = '残しました（AI要約は上位棚9Bを優先利用）'
        Write-Host ("   AI要約(上位棚9B): " + $Summary9BAction) -ForegroundColor Green
    } else {
        Remove-Item $Model9BPath -Force
        $Summary9BAction = ("削除しました（約{0:N1}GBを解放・この端末では9Bは使わず4Bにフォールバック）" -f $Model9BGB)
        Write-Host ("   AI要約(上位棚9B): " + $Summary9BAction)
    }
}
# setup_report.txt へ追記する判定結果（検証の後に追記する）
$SummaryReportLines = @(
    "",
    "---- 端末スペックの自動判定（セットアップ時） ----",
    ("搭載メモリ: {0:N1} GB（AI要約の目安: 11.5GB超）" -f $RamGB),
    ("AI要約用4Bモデル: " + $SummaryAction),
    ("理由: " + $SummaryReason),
    ("AI要約用9Bモデル（上位棚）: " + $Summary9BAction)
)

# 2. Python 3.12 を探す
Write-Step "Python 3.12 を確認しています"
$Python = $null
foreach ($candidate in @(
    @('py', @('-3.12', '-c', 'import sys;print(sys.executable)')),
    @('python', @('-c', 'import sys;print(sys.executable)'))
)) {
    try {
        $exe = & $candidate[0] @($candidate[1]) 2>$null | Select-Object -First 1
        if ($exe -and (Test-Path $exe)) {
            $version = & $exe -c "import sys;print('{0}.{1}'.format(*sys.version_info))" 2>$null
            if ($version -eq '3.12') { $Python = $exe; break }
        }
    } catch { }
}
if (-not $Python) {
    $installer = Get-ChildItem $Root -Filter 'python-3.12*-amd64.exe' | Select-Object -First 1
    if ($installer) {
        Write-Step "Python 3.12 をインストールします（同梱インストーラー）"
        Start-Process $installer.FullName -ArgumentList '/passive', 'InstallAllUsers=0', 'PrependPath=1', 'Include_launcher=1' -Wait
        $exe = & py -3.12 -c 'import sys;print(sys.executable)' 2>$null | Select-Object -First 1
        if ($exe -and (Test-Path $exe)) { $Python = $exe }
    }
}
if (-not $Python) {
    throw "Python 3.12 が見つかりません。python.org から 3.12 をインストールするか、配布担当者に連絡してください。"
}
Write-Host "   Python: $Python"

# 3. 仮想環境を作成
Write-Step "アプリ専用のPython環境を作成しています"
if (-not (Test-Path "$Root\.venv\Scripts\python.exe")) {
    & $Python -m venv "$Root\.venv"
}
$VenvPython = "$Root\.venv\Scripts\python.exe"

# 4. 同梱ホイールからオフラインインストール
Write-Step "ライブラリをインストールしています（オフライン）"
& $VenvPython -m pip install --no-index --find-links "$Root\wheels" -r "$Root\requirements.txt" --quiet
if ($LASTEXITCODE -ne 0) { throw "ライブラリのインストールに失敗しました。" }
$reazon = Get-ChildItem "$Root\wheels" -Filter 'reazonspeech*'
if ($reazon) {
    & $VenvPython -m pip install --no-index --no-deps $reazon.FullName --quiet
    if ($LASTEXITCODE -ne 0) { throw "ReazonSpeech パッケージのインストールに失敗しました。" }
}

# 5. セットアップ検証
$VerifyExit = 0
if (-not $SkipVerify) {
    Write-Step "セットアップを検証しています"
    $env:HF_HOME = "$Root\hf-cache"
    $env:PYTHONUTF8 = '1'
    & $VenvPython "$Root\verify_setup.py"
    $VerifyExit = $LASTEXITCODE
}

# 6. スペック自動判定の結果を setup_report.txt へ追記
#    （verify_setup.py が setup_report.txt を作り直した後に追記する。
#      テスト結果の回収時に、RAM実測値と4Bの扱いもあわせて確認できる）
try {
    Add-Content -Path "$Root\setup_report.txt" -Value $SummaryReportLines -Encoding UTF8
} catch {
    Write-Host "   判定結果を setup_report.txt に追記できませんでした（動作に影響はありません）。" -ForegroundColor Yellow
}

# 7. （任意）-BlockNetwork: 外部への送信をWindowsファイアウォールでブロック
#    「送信しない」を「送信できない」にする学校管理者向けオプション。
#    端末内部の通信(127.0.0.1)はWindowsファイアウォールの対象外のため、
#    アプリの動作（文字起こし・要約・チャット）には影響しない。
if ($BlockNetwork) {
    Write-Step "外部への送信をブロックしています（-BlockNetwork・学校管理者向け）"
    $BlockReportLines = @("", "---- 外部送信ブロック（-BlockNetwork） ----")
    $IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $IsAdmin) {
        Write-Host "   管理者権限が無いため、この設定だけスキップしました。" -ForegroundColor Yellow
        Write-Host "   （セットアップのほかの部分は、このまま最後まで実行されます）"
        Write-Host "   設定するには、PowerShell を「管理者として実行」で開き、次を実行してください:"
        Write-Host ("     powershell -ExecutionPolicy Bypass -File `"{0}`" -BlockNetwork -SkipVerify" -f $PSCommandPath)
        $BlockReportLines += "結果: 管理者権限が無いためスキップしました（未設定）。"
    } else {
        try {
            # 再実行時は既存の OtoWeave-NoNetwork-* ルールを置き換える
            $ExistingRules = @(Get-NetFirewallRule -DisplayName 'OtoWeave-NoNetwork-*' -ErrorAction SilentlyContinue)
            if ($ExistingRules.Count -gt 0) {
                $ExistingRules | Remove-NetFirewallRule
                Write-Host ("   既存のルール {0} 件を置き換えます。" -f $ExistingRules.Count)
            }
            $LlamaServerExe = $null
            if (Test-Path "$Root\engines") {
                $LlamaServerExe = Get-ChildItem "$Root\engines" -Recurse -Filter 'llama-server.exe' -ErrorAction SilentlyContinue |
                    Select-Object -First 1 -ExpandProperty FullName
            }
            $BlockTargets = @(
                @{ Suffix = 'python';       Path = "$Root\.venv\Scripts\python.exe" },
                @{ Suffix = 'pythonw';      Path = "$Root\.venv\Scripts\pythonw.exe" },
                @{ Suffix = 'ffmpeg';       Path = "$Root\engines\ffmpeg\ffmpeg.exe" },
                @{ Suffix = 'llama-server'; Path = $LlamaServerExe }
            )
            $BlockCount = 0
            foreach ($target in $BlockTargets) {
                if ($target.Path -and (Test-Path $target.Path)) {
                    New-NetFirewallRule -DisplayName ('OtoWeave-NoNetwork-' + $target.Suffix) `
                        -Description 'OtoWeave: 外部への送信をブロック（端末内部 127.0.0.1 の通信には影響しません）' `
                        -Direction Outbound -Action Block -Program $target.Path -Profile Any | Out-Null
                    Write-Host ("   ブロック: {0}" -f $target.Path)
                    $BlockReportLines += ("送信ブロック: " + $target.Path)
                    $BlockCount++
                }
            }
            Write-Host ("   外部への送信をブロックするルールを {0} 件作成しました（ルール名: OtoWeave-NoNetwork-*）。" -f $BlockCount) -ForegroundColor Green
            Write-Host "   端末内部の通信(127.0.0.1)には影響しないため、アプリは全機能使えます。"
            Write-Host "   解除するには（管理者のPowerShellで）:"
            Write-Host "     Remove-NetFirewallRule -DisplayName 'OtoWeave-NoNetwork-*'"
            $BlockReportLines += ("結果: 外部送信ブロックのルールを {0} 件作成しました（OtoWeave-NoNetwork-*）。" -f $BlockCount)
        } catch {
            Write-Host ("   ファイアウォール設定に失敗しました: {0}" -f $_.Exception.Message) -ForegroundColor Red
            Write-Host "   （アプリ自体はこの設定が無くても外部送信を行いません。設定は後から再実行できます）"
            $BlockReportLines += ("結果: 設定に失敗しました: " + $_.Exception.Message)
        }
    }
    try {
        Add-Content -Path "$Root\setup_report.txt" -Value $BlockReportLines -Encoding UTF8
    } catch { }
}

if ($VerifyExit -ne 0) {
    Write-Host "検証で NG があります。上の表示を確認してください。" -ForegroundColor Yellow
    # キー入力待ちは setup.bat 側の pause に任せる（二重の入力待ちを避ける）
    exit 1
}

Write-Host ""
Write-Host "セットアップ完了！「OtoWeaveを起動.bat」をダブルクリックで起動できます。" -ForegroundColor Green
Write-Host "ヒント: 空き容量が必要な場合、wheels フォルダ（約300MB）は削除して構いません。" -ForegroundColor Yellow
Read-Host "Enter キーで終了"
