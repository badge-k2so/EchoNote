# Mac (Apple Silicon) 対応計画

状況: 2026-07-05 時点の決定事項。**着手は Windows プロトタイプテスト完了後**。

## 背景・対象

- N高の標準環境が Mac（MacBook Air M2 8GB 以上が推奨スペック）のため、
  高校での試用には Apple Silicon 版が必要
- 検証実機: 自宅の **MacBook M3 / RAM 8GB**（M2 8GB とほぼ同予算で代表機として十分）
- コード受け渡し: **GitHub 非公開リポジトリ**（Windowsでpush → M3でclone）。
  モデルファイルは .gitignore 済みなので別途コピーかMac上でダウンロード

## 工数見積り（設計調査済み）

コアの約85%（UI/ASR/LLM/ロジック/テスト196本）はそのまま動く。書き換え対象:

| 箇所 | 対応 | 規模 |
|---|---|---|
| 録音（PyAudioWPatch=Win専用） | sounddevice の InputStream へ差し替え | 中（最大の作業） |
| TTS（PowerShell System.Speech） | `say -v Kyoko` へ分岐 | 小 |
| 孤児プロセス対策（windows_job.py） | POSIXプロセスグループへ分岐 | 小 |
| ffmpeg.exe パス | プラットフォーム分岐 + arm64バイナリ | 小 |
| RAM検出（GlobalMemoryStatusEx） | sysctl 分岐 | 極小 |
| ファイル選択（PowerShellヘルパー） | tkinter.filedialog | 小 |
| デモ音声合成 | `say -o` + ffmpeg変換 | 極小 |
| .ps1 スクリプト群 | .sh 化 | 小〜中 |

フェーズ: M1=最小移植（マイクのみ）3〜5日 → M2=Apple Siliconプロファイル 3〜5日
→ M3=PC音声(ScreenCaptureKit)+公証 1〜2週

## Apple Silicon 用モデルプロファイル（M2フェーズ）

M2/M3 8GB は Metal GPU + ユニファイドメモリ（モデル実質予算 4〜5GB）。

| 役割 | GIGA-Windows 構成 | Apple Silicon 構成（案） |
|---|---|---|
| ライブ字幕 | ReazonSpeech K2 | ReazonSpeech K2 継続（低遅延・省メモリ） |
| あとから文字起こし | K2 / Parakeet / SpeechBrain振り分け | **Whisper large-v3-turbo**（約1.6GB）— 日英混在を単体処理、言語振り分けパイプライン不要に。代替: kotoba-whisper v2.2 |
| 要約 | Qwen3.5-4B（低RAMは2B） | **7-8B級 Q4（約4.5GB）** — サブプロセス実行+解放の現構造なら8GBで成立。llama.cpp に `--n_gpu_layers` を渡すこと（Metal 有効化。CLI引数の追加が必要） |
| チャット常駐 | 2B | 4B に格上げ |
| TTS | Haruka | `say`（Kyoko）→ 将来 AVSpeechSynthesis |

## N高特有の注意

1. **オンライン授業が中心** → 「PC音声録音」の優先度が上がる。
   macOS はループバック非搭載: ScreenCaptureKit(macOS13+) 実装 or BlackHole 案内
2. **映像授業（N予備校等）の録音許諾**を学校規約と要確認。
   合理的配慮としての申請が交渉材料になり得る
3. 配布は Gatekeeper 対策（Apple Developer $99/年 + 公証）が必要

## 着手時の手順メモ

1. GitHub 非公開リポジトリ作成 → Windows から push（`gh auth login` または PAT）
2. M3: Claude Code + Python 3.12 + Xcode Command Line Tools を導入、clone
3. M1フェーズを M3 上で実施（その場でテスト・修正のループが最速）
4. Windows 側は同一コードベースを維持（プラットフォーム分岐で共存）
