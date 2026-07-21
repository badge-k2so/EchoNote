# OtoWeave 通信監視スクリプト（オフライン動作の確認・記録づくり）
#
# OtoWeave に関係するプログラム（このフォルダ配下の python.exe /
# pythonw.exe / ffmpeg.exe / llama-server.exe）の TCP 通信を 2 秒ごとに
# 確認し、端末の外（127.0.0.1 / ::1 以外）への接続が無いことを
# offline_report.txt に記録します。管理者権限は不要です。
#
# 使い方:
#   1) このファイルを右クリック →「PowerShell で実行」
#      （またはPowerShellで:
#        powershell -ExecutionPolicy Bypass -File .\verify_offline.ps1 ）
#   2) 監視が始まったら、OtoWeave をいつもどおり使う
#   3) 終了すると、同じフォルダの offline_report.txt に結果が追記される
#
# オプション:
#   -DurationMinutes 30 : 監視時間を分で指定（既定 10）
#   -DurationMinutes 0  : Ctrl+C で止めるまで監視し続ける
#                         （Ctrl+C で止めてもレポートは書き込まれます）
param(
    [double]$DurationMinutes = 10
)

$ErrorActionPreference = 'Stop'

if ($DurationMinutes -lt 0) {
    Write-Host "-DurationMinutes には 0 以上の数を指定してください（0 = Ctrl+C まで監視）。" -ForegroundColor Red
    exit 1
}

$Root = Split-Path -Parent $PSCommandPath
$ReportPath = Join-Path $Root 'offline_report.txt'
$IntervalSeconds = 2
# 監視対象のプロセス名（実体がこのフォルダ配下にあるものだけを対象にする）
$TargetNames = @('python', 'pythonw', 'ffmpeg', 'llama-server')

# ループバック（この端末自身を指す宛先。端末の外には出ない）かどうか
function Test-LoopbackAddress {
    param([string]$Address)
    if ([string]::IsNullOrWhiteSpace($Address)) { return $true }
    $addr = $Address.Split('%')[0].Trim()
    if ($addr -eq '::1' -or $addr -eq '::' -or $addr -eq '0.0.0.0') { return $true }
    if ($addr -like '127.*') { return $true }
    if ($addr -like '::ffff:127.*') { return $true }
    return $false
}

Write-Host "================================================================"
Write-Host " OtoWeave 通信監視（オフライン動作の確認）"
Write-Host "================================================================"
Write-Host ("監視対象フォルダ : {0}" -f $Root)
Write-Host ("監視対象         : このフォルダ配下の python / pythonw / ffmpeg / llama-server")
if ($DurationMinutes -gt 0) {
    Write-Host ("監視時間         : {0} 分（{1} 秒ごとに確認）" -f $DurationMinutes, $IntervalSeconds)
} else {
    Write-Host ("監視時間         : Ctrl+C で止めるまで（{0} 秒ごとに確認）" -f $IntervalSeconds)
}
Write-Host ("結果の保存先     : {0}" -f $ReportPath)
Write-Host ""
Write-Host "監視を始めました。この画面は開いたまま、OtoWeave をいつもどおり"
Write-Host "使ってください（録音・文字起こし・要約・チャットなど）。"
Write-Host ""

$StartTime = Get-Date
$SampleCount = 0      # 確認（サンプリング）できた回数
$ErrorCount = 0       # 確認に失敗した回数（あっても監視は続ける）
$MinProcs = -1        # 監視対象プロセス数の最小値
$MaxProcs = 0         # 監視対象プロセス数の最大値
$SeenProcs = @{}      # 監視中に確認できたプロセス（"名前 (PID n)" -> $true）
$Detections = @{}     # 外部宛て接続（"宛先:ポート|プロセス名" -> 詳細）

try {
    while ($true) {
        if (($DurationMinutes -gt 0) -and
            (((Get-Date) - $StartTime).TotalMinutes -ge $DurationMinutes)) {
            break
        }
        try {
            # 1) このフォルダ配下で動いている監視対象プロセスを探す
            $procs = @(Get-Process -Name $TargetNames -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.Path -and
                    $_.Path.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)
                })
            $procCount = $procs.Count
            if (($MinProcs -lt 0) -or ($procCount -lt $MinProcs)) { $MinProcs = $procCount }
            if ($procCount -gt $MaxProcs) { $MaxProcs = $procCount }
            $NameByPid = @{}
            foreach ($p in $procs) {
                $NameByPid[[uint32]$p.Id] = $p.ProcessName
                $SeenProcs[("{0} (PID {1})" -f $p.ProcessName, $p.Id)] = $true
            }

            # 2) それらのプロセスの TCP 接続を確認し、外部宛てだけを記録する
            if ($procCount -gt 0) {
                $conns = @(Get-NetTCPConnection -ErrorAction SilentlyContinue |
                    Where-Object { $NameByPid.ContainsKey([uint32]$_.OwningProcess) })
                foreach ($c in $conns) {
                    # 待ち受け（Listen/Bound）は「接続」ではないので除外
                    if (($c.State -eq 'Listen') -or ($c.State -eq 'Bound')) { continue }
                    # ループバック宛て（127.0.0.1 / ::1 など）は端末の外に出ないので除外
                    if (Test-LoopbackAddress $c.RemoteAddress) { continue }
                    $procName = $NameByPid[[uint32]$c.OwningProcess]
                    $key = ("{0}:{1}|{2}" -f $c.RemoteAddress, $c.RemotePort, $procName)
                    if ($Detections.ContainsKey($key)) {
                        $Detections[$key].Count = $Detections[$key].Count + 1
                    } else {
                        $Detections[$key] = @{
                            Remote    = $c.RemoteAddress
                            Port      = $c.RemotePort
                            Process   = $procName
                            FirstSeen = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
                            Count     = 1
                        }
                        Write-Host ("[検出] 外部宛ての接続: {0}:{1}（プロセス: {2}）" -f `
                            $c.RemoteAddress, $c.RemotePort, $procName) -ForegroundColor Red
                    }
                }
            }
            $SampleCount++
            # 30秒ごとに経過を表示
            if (($SampleCount % 15) -eq 0) {
                Write-Host ("  経過 {0:N1} 分 / 確認 {1} 回 / 監視対象プロセス {2} 個 / 外部宛て接続 {3} 件" -f `
                    ((Get-Date) - $StartTime).TotalMinutes, $SampleCount, $procCount, $Detections.Count)
            }
        } catch {
            $ErrorCount++
            if ($ErrorCount -le 3) {
                Write-Host ("  確認に失敗しました（監視は続けます）: {0}" -f $_.Exception.Message) -ForegroundColor Yellow
            }
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
} finally {
    # Ctrl+C で止めた場合もここは実行され、レポートが書き込まれる
    $EndTime = Get-Date
    $ElapsedMinutes = ($EndTime - $StartTime).TotalMinutes
    if ($MinProcs -lt 0) { $MinProcs = 0 }

    $lines = @()
    $lines += "================================================================"
    $lines += " OtoWeave オフライン動作確認レポート（verify_offline.ps1）"
    $lines += "================================================================"
    $lines += ("実施日時       : {0} ～ {1}" -f `
        $StartTime.ToString('yyyy-MM-dd HH:mm:ss'), $EndTime.ToString('yyyy-MM-dd HH:mm:ss'))
    $lines += ("端末名         : {0}（ユーザー: {1}）" -f $env:COMPUTERNAME, $env:USERNAME)
    $lines += ("監視対象       : {0} 配下の python / pythonw / ffmpeg / llama-server" -f $Root)
    $lines += ("監視方法       : {0} 秒ごとに TCP 接続を確認（Get-NetTCPConnection）" -f $IntervalSeconds)
    $lines += ("監視時間       : {0:N1} 分（確認 {1} 回、失敗 {2} 回）" -f $ElapsedMinutes, $SampleCount, $ErrorCount)
    $lines += ("監視対象プロセス数の推移: 最小 {0} 個 ～ 最大 {1} 個" -f $MinProcs, $MaxProcs)
    if ($SeenProcs.Count -gt 0) {
        $lines += ("確認できたプロセス      : " + (($SeenProcs.Keys | Sort-Object) -join ', '))
    } else {
        $lines += "確認できたプロセス      : なし（監視中に OtoWeave のプロセスは起動していませんでした。"
        $lines += "                          アプリを使いながらもう一度実行すると、記録として有効です）"
    }
    $lines += "----------------------------------------------------------------"
    $lines += ("外部宛て（127.0.0.1 / ::1 以外）への接続: {0} 件（期待値: 0 件）" -f $Detections.Count)
    if ($Detections.Count -eq 0) {
        $lines += "まとめ: 外部への通信は検出されませんでした。監視中、このアプリが"
        $lines += "        インターネットにデータを送っていないことが確認できました。"
    } else {
        $lines += "まとめ: 外部宛ての接続が検出されました。宛先の一覧は次のとおりです。"
        foreach ($key in ($Detections.Keys | Sort-Object)) {
            $d = $Detections[$key]
            $lines += ("  - 宛先 {0}:{1} / プロセス {2} / 初回検出 {3} / 検出回数 {4}" -f `
                $d.Remote, $d.Port, $d.Process, $d.FirstSeen, $d.Count)
        }
        $lines += "        （心当たりが無い場合は、配布担当者へこのレポートをお送りください）"
    }
    $lines += ""

    Write-Host ""
    try {
        Add-Content -Path $ReportPath -Value $lines -Encoding UTF8
        Write-Host ("結果を保存しました: {0}" -f $ReportPath) -ForegroundColor Green
    } catch {
        Write-Host ("結果を {0} に保存できませんでした: {1}" -f $ReportPath, $_.Exception.Message) -ForegroundColor Red
        Write-Host "以下に同じ内容を表示します。必要ならコピーして保存してください。"
    }
    foreach ($line in $lines) { Write-Host $line }
    if ($Detections.Count -eq 0) {
        Write-Host "外部への通信は検出されませんでした。" -ForegroundColor Green
    } else {
        Write-Host ("外部宛ての接続が {0} 件検出されました。レポートを確認してください。" -f $Detections.Count) -ForegroundColor Red
    }
}
