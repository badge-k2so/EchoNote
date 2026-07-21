# LearningAccess 音声AI パイプライン 開発記録

作成日: 2026-05-23  
目的: 小中学校の学習支援を想定したWindows CPU動作の音声AI環境構築

マイプロ提出用の証拠一覧はローカル管理の資料（Git管理外）を参照。

---

## 1. プロジェクト概要

### ゴール
- **教師向け**: 授業・面談の録音 → 文字起こし → 整形 → 要約
- **生徒向け**: 音声で話しかける → LLMがメタ認知を促す問いかけを返す → 読み上げ
- **動作環境**: Windows ノートPC（CPU のみ）、生徒・学校への配布を想定
- **配布方式**: Embeddable Python（Windowsアップデートの影響を受けない）
- **初期方針**: ネットを介さないローカル完結型を優先し、学校・学習支援の現場で安全に試せる形にする

### フェーズ1の最優先範囲

まずは **Windows ノートPC / CPU処理 / CUDAなし** で、安定して使える品質のSTTパイプラインを作る。ここを第1優先とする。

フェーズ1に含めるもの:

- 音声ファイル入力
- ffmpegによる16kHz mono WAV変換
- VAD / 無音区間分割
- ReazonSpeech-k2-v2 による日本語STT
- faster-whisper small INT8 による英語・多言語STT
- ローカルLLM（CUDAなし、GGUF / llama.cpp）による整形
- ローカルLLMによるサマリー生成
- ログ、処理時間、RAM使用量、エラー記録
- 出力ファイルの分離: raw transcript / formatted / summary / metadata
- 初回モデル取得後はネット接続なしで動く構成

フェーズ1では優先しないもの:

- GUIアプリ化
- リアルタイム音声Chat
- TTS読み上げ
- 話者分離
- GPU / CUDA前提のモデル
- VLM、図形・画像理解
- 外部API前提の機能

フェーズ1の採用方針:

| 領域 | 採用構成 | 理由 |
|---|---|---|
| 日本語STT | ReazonSpeech-k2-v2 + VAD | CPUで高速、学校・面談語彙に強い |
| 英語/多言語STT | faster-whisper small INT8 | CPUで現実的、言語自動検出あり |
| LLM整形/要約 | Qwen3.5-2B Q4_K_M | CUDAなしCPUで速度と品質のバランスが最良 |
| 実行環境 | `.venv` から Embeddable Python へ移行 | 学校PCで環境破壊を避ける |
| 音声変換 | ffmpeg同梱 | 入力形式差を吸収する |

### ローカル完結方針

学校・児童生徒の環境では、個人情報、録音、相談内容、学習履歴を外部APIへ送らない設計を基本にする。初期版はネット不通でも動くことを重視する。

外部APIを使う可能性がある機能、特にVLMによる画像理解、クラウドLLM、クラウドTTSはフェーズ1・2では扱わない。将来導入する場合も、学校側の規程、保護者同意、送信前確認、ログ削除ポリシーを別途設計する。

---

## 2. ASRエンジン比較と最終推奨

詳細: `runs/comparison_reazon_k2_vs_whisper.md`

### 比較対象

| エンジン | 特徴 |
|---|---|
| whisper.cpp (base/small/medium) | C++バイナリ、GGML量子化 |
| ReazonSpeech-k2-v2 | 日本語専用、sherpa-onnx、int8 |
| faster-whisper small INT8 | Python統一、Silero VAD内蔵 |
| WhisperX large-v3 (参考) | GPU、話者分離、最高精度だが要GPU |

### 10分テスト 速度比較（日本語音声）

| エンジン | end-to-end | Peak RAM |
|---|---|---|
| ReazonSpeech-k2-v2 VAD | **35 sec** | 876 MB |
| faster-whisper small INT8 | 159 sec | 1077 MB |
| whisper.cpp small | 306 sec | 919 MB |
| whisper.cpp medium | 954 sec | 2242 MB |

Moonshine Voice は速度・RAMでは最軽量だったが、今回の日本語面談では文字間スペースや誤認識が多く、開発候補から外す。詳細は実験記録 `runs/comparison_reazon_k2_vs_whisper.md` にのみ残す。

### 最終 ASR 推奨構成

| 用途 | エンジン | 理由 |
|---|---|---|
| 日本語授業・面談 | **ReazonSpeech-k2-v2 + VAD** | 最速・日本語ドメイン語に強い |
| 英語授業 | **faster-whisper small INT8** | 2倍速・言語自動検出・純Python |
| 混在・不明 | faster-whisper small INT8 (auto) | 言語検出p=0.93 |

### VADチャンキング

ffmpeg `silencedetect` で自然な無音点を検出し最大28秒チャンクに分割。固定30秒より精度向上（`テキストスピーチ`誤認識が解消）。

---

## 3. LLM後処理パイプライン

### モデル比較

| モデル | サイズ | 整形時間 | 要約時間 | 品質 |
|---|---|---|---|---|
| Qwen2.5-1.5B Q4_K_M | ~1.0GB | 44 sec | 18 sec | ○ バランス型 |
| Qwen3-0.6B Q4_K_M | ~0.5GB | 36 sec | 8 sec | △ 要約が不安定 |
| Qwen3-1.7B Q4_K_M | ~1.1GB | 89 sec | 29 sec | 除外済み（FINAL失敗・偽タイムスタンプが出やすい） |
| **Qwen3.5-2B Q4_K_M** | ~1.3GB | 検証済み | 検証済み | ◎ 標準採用 |
| Qwen3.5-4B Q4_K_M | ~2.7GB | 2Bより約2.5倍遅い | 検証済み | ○ 再処理・品質確認用 |

### 最終 LLM 推奨

**Qwen3.5-2B Q4_K_M**

- CPU処理で速度と品質のバランスが最良
- Thinking OFF運用で応答生成にCPUを集中
- 英語・日本語バイリンガル
- 4Bは2Bが崩れた場合の再処理候補として残す
- Qwen3-1.7Bは候補から外し、モデルファイルも削除済み

### RAM別実行プリセット

2026-05-24に `run_school_pipeline_preset.ps1` を追加。GUIから呼び出す前提の統合ランナーとして、ASRとLLM整形の組み合わせを固定する。

| プリセット | 想定RAM | 用途 | 処理方針 |
|---|---:|---|---|
| `Lite` | 4GB | 生徒用・低スペック端末 | 文字起こし優先。標準ではLLM後処理を省略し、落ちにくさを優先 |
| `Standard` | 8GB | メイン標準 | ReazonSpeech K2 int8 + Qwen3.5-2B。日本語中心の学校記録向け |
| `HighQuality` | 16GB | 教員PC・再処理 | faster-whisper patch数やcontextを増やし、品質確認に使う |

ContentType:

- `Japanese`: 日本語中心。StandardではReazon K2先行のhybrid処理。
- `EnglishMixed`: 英語授業・日英交互。英語らしい区間だけfaster-whisper smallで補助。
- `English`: 英語主体。Reazon先行処理を使わずfaster-whisper autoで処理。

メイン運用は `-Mode Standard -ContentType Japanese` とする。

### 出力ファイル（切り離し設計）

```
output/
  raw_transcript.txt        # ASR出力そのまま
  safe_transcript.md        # Stage 1標準。ASR結果をチャンク順・時刻付きで確認する原文確認用出力
  clean_transcript.md       # 旧互換名。新規設計ではsafe_transcript.mdを優先
  ai_readable_transcript.md # 任意。Qwenによる読みやすさ優先の参考出力
  chunks_raw/               # chunk単位の原文
  chunks_safe/              # chunk単位のsafe版
  chunks_clean/             # 旧互換。safeまたはAI整形版
  chunks_ai_readable/       # 任意。chunk単位のAI整形版
  formatted_transcript.md  # チャンク順の整形文字起こし
  formatted_transcript.txt # 整形文字起こしのテキスト版
  school_record.md         # 学校向け記録
  meeting_record.md        # school_record互換の面談記録名
  review_flags.md          # 要確認リスト
  summary.md               # 3行サマリーのみ
  summary.txt              # 3行サマリーのテキスト版
  school_postprocess_summary.json
```

---

## 4. 学習支援モード設計

### メタ認知を促すプロンプト設計方針

- 答えを直接教えない
- 「なぜそう思う？」「どこでつまずいた？」等のソクラテス式問いかけ
- 1回の返答は2文以内
- 作文の代わりに書かない → ヒントと問いかけのみ

### 用途別モード

| モード | 用途 | 設計方針 |
|---|---|---|
| `study` | 算数・国語・学習支援 | 答えを教えない、考えを引き出す |
| `english` | 英会話練習 | 1〜2文、最後に質問 |
| `chat` | 汎用対話 | 短く返す |

### 教科別の注意点

| 教科 | LLM向き不向き | 注意 |
|---|---|---|
| 国語・作文 | ◎ | 代わりに書かせない設計が必要 |
| 英語 | ◎ | 文法添削・英作文に有効 |
| 算数・数学 | ○（説明のみ） | 計算の正確性は保証できない |
| 理科・社会 | △（記述支援のみ） | 事実をLLMに言わせない設計が必要 |
| 図形・グラフ | ✗（テキストのみ） | VLMが必要（現状CPUでは非実用） |

---

## 5. フェーズ3: 音声Chatパイプライン

第3フェーズでは、音声Chatによる学習サポートを想定する。児童生徒が音声で話しかけ、AIが短い問いかけやヒントを返すことで、読み書きが苦手な児童生徒でも使いやすい支援体験を作る。

ただし、フェーズ1・2で採用するSTT、TTS、LLMモデルをそのまま音声Chatに使うと、ターンごとの遅延が大きくなる可能性がある。音声Chatではバッチ文字起こしと違い、初声までの待ち時間が体験品質に直結する。

### アーキテクチャ

```
[マイク] → VAD（エネルギーベース）
         → faster-whisper ASR
         → Qwen3.5-2B（ストリーミング生成）
         → 文末ごとに VOICEVOX API へ dispatch
         → sounddevice で再生
```

### 設計のポイント

- **エコー防止**: TTS再生中は録音を停止（`is_speaking`フラグ）
- **Qwen3 thinking除去**: `<think>...</think>`をトークン単位ステートマシンで除去
- **文末ストリーミング**: 句点・疑問符・改行で即TTS → 最初の音声まで3〜5秒
- **コンテキスト保持**: 直近10ターンを会話履歴として維持

### 音声Chatでの遅延課題

| 要素 | フェーズ1・2の候補 | 音声Chatでの懸念 | 対策候補 |
|---|---|---|---|
| STT | ReazonSpeech-k2-v2 + VAD | 録音終了後にチャンク処理する設計だと会話応答が遅れる | 短い発話単位でVAD確定、1〜5秒発話を即処理 |
| STT | faster-whisper small INT8 | 英語・多言語には便利だが、CPUではReazonSpeechより遅い | 英語モード専用、短発話限定、必要ならtiny/baseも検証 |
| LLM | Qwen3.5-2B Q4_K_M | 品質は良いが初声まで数秒かかる可能性 | Thinking OFF、ストリーミング生成、短文プロンプト、必要なら0.6Bへ切替 |
| TTS | VOICEVOX / ローカルTTS | 音声合成完了待ちで体感遅延が増える | 文単位で逐次TTS、短い返答に制限 |

音声Chatでは、フェーズ1の「録音全体を高品質に処理する」設計とは別に、**短い発話を低遅延で処理する専用モード** が必要になる。

### 音声Chat向けの暫定方針

- 1ターンのAI返答は1〜2文に制限する
- LLMには長い議事録整形プロンプトを使わない
- STTは発話終了検知後すぐ処理する
- TTSは全文生成を待たず、文末ごとに読み上げる
- 速度優先モードでは Qwen3-0.6B も選択可能にする
- 標準モードでは Qwen3.5-2B を使う
- 音声Chatはフェーズ1・2の安定後に、別途レイテンシーテストを行う

### CPU実機での期待レイテンシー（初声まで）

| CPU | モデル | 期待値 |
|---|---|---|
| Core i7 12世代 | Qwen3.5-2B | 要実測 |
| Core i5 10世代 | Qwen3.5-2B | 要実測 |
| Core i5 + Qwen3-0.6B | Qwen3-0.6B | ~1〜2秒（品質低下） |

### TTS: VOICEVOX

- インストール: https://voicevox.hiroshiba.jp/
- ローカルHTTP API（`http://localhost:50021`）
- 主なspeaker_id: 1=ずんだもん, 3=ずんだもん囁き, 8=春日部つむぎ
- Microsoft Haruka（pyttsx3）より大幅に音質向上

---

## 6. スクリプト一覧

### PowerShell ランナー

| スクリプト | 用途 |
|---|---|
| `run_reazon_k2_vad_test.ps1` | ReazonSpeech VAD 文字起こしテスト |
| `run_faster_whisper_test.ps1` | faster-whisper 文字起こしテスト |
| `run_pipeline_cpu.ps1` | フェーズ1統合ランナー（STT→整形→要約） |
| `run_whisper_test.ps1` | whisper.cpp 文字起こしテスト |
| `run_llm_postprocess.ps1` | LLM整形・要約（バッチ処理） |
| `run_voice_chat.ps1` | リアルタイム音声Chat |

### Pythonスクリプト

| スクリプト | 用途 |
|---|---|
| `scripts/faster_whisper_transcribe.py` | faster-whisper メイン |
| `scripts/reazon_k2_transcribe.py` | ReazonSpeech メイン |
| `scripts/vad_chunk.py` | VADチャンキング |
| `scripts/llm_postprocess.py` | LLM整形・要約 |
| `scripts/voice_chat.py` | 音声Chat（VAD+ASR+LLM+TTS） |

### モデルファイル（`models/`）

| ファイル | 用途 |
|---|---|
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | LLM（速度優先） |
| `Qwen3-0.6B-Q4_K_M.gguf` | LLM（最軽量） |
| `Qwen3.5-2B-Q4_K_M.gguf` | LLM（標準・推奨） |
| `Qwen3.5-4B-Q4_K_M.gguf` | LLM（再処理・品質確認用） |

---

## 7. 実行例

### フェーズ1統合処理（日本語メイン）

```powershell
.\run_pipeline_cpu.ps1 -InputFile "C:\録音.m4a" -AsrMode ja -MaxDurationSeconds 0
```

出力:

```text
runs\<timestamp>_cpu_pipeline_ja_full\output\raw_transcript.txt
runs\<timestamp>_cpu_pipeline_ja_full\output\formatted.txt
runs\<timestamp>_cpu_pipeline_ja_full\output\summary.txt
runs\<timestamp>_cpu_pipeline_ja_full\output\metadata.json
```

### フェーズ1統合処理（英語授業・英語混じり）

英語授業や英語が多い録音では `auto` または `en` を使う。

```powershell
.\run_pipeline_cpu.ps1 -InputFile "C:\録音.m4a" -AsrMode auto -MaxDurationSeconds 0
.\run_pipeline_cpu.ps1 -InputFile "C:\録音.m4a" -AsrMode en -MaxDurationSeconds 0
```

運用方針:

- 通常の日本語授業・面談は `-AsrMode ja`
- 英語授業、英語音声が多い録音は `-AsrMode auto` または `-AsrMode en`
- 日本語中心だが `iPad`、`SAT`、`MacBook` など英語固有名詞が混じる程度なら `ja` を基本にし、必要なら後処理辞書で補正する

### 文字起こし（日本語）

```powershell
.\run_reazon_k2_vad_test.ps1 -InputFile "C:\録音.m4a" -MaxDurationSeconds 0
```

### 文字起こし（英語・自動検出）

```powershell
.\run_faster_whisper_test.ps1 -InputFile "C:\録音.m4a" -Language None -MaxDurationSeconds 0
```

### LLM整形・要約

```powershell
.\run_llm_postprocess.ps1 -TranscriptFile "runs\XXXXX\output\full_transcript.txt"
```

### 音声Chat（VOICEVOXを先に起動）

```powershell
.\run_voice_chat.ps1 -Mode study -Language ja
.\run_voice_chat.ps1 -Mode english -Language en -SpeakerId 8
```

---

## 8. 今後の拡張候補

| 項目 | 概要 | 優先度 |
|---|---|---|
| CPU STT+LLM統合ランナー | 音声入力から文字起こし、整形、要約までを1コマンド化 | ★★★ |
| エラー復旧・再実行設計 | chunk単位失敗時の再実行、途中結果保持、ログ標準化 | ★★★ |
| Embeddable Python配布パッケージ | 学校向けインストーラ、外部Python非依存 | ★★★ |
| GUIアプリ化 | フェーズ1安定後に tkinter / Electron 等で実装、ローカル完結を維持 | ★★☆ |
| ローカルTTS | VOICEVOXまたはIrodori-TTS-Lite-int4等、外部APIなしの読み上げ | ★★☆ |
| 音声Chatレイテンシーテスト | STT→LLM→TTSの初声時間、ターン時間を実測 | ★★☆ |
| RAG（教科書参照） | 理科・社会の事実精度向上 | ★★☆ |
| VLM（図形・グラフ対応） | Qwen2.5-VL 3B等、要高スペックPC | ★☆☆ |
| 話者分離（ReazonSpeech拡張） | 面談記録の話者別出力 | ★★☆ |

### 直近の実装優先順位

1. **CPU STT+LLM統合ランナー**
   - 音声ファイルを渡すだけで、VAD文字起こし、整形、要約、メタデータ出力まで一括実行する。
   - フェーズ1の核。GUIより先にCLIで安定させる。
   - 実装済み: `run_pipeline_cpu.ps1`
   - 日本語中心: `-AsrMode ja`
   - 英語授業・英語多め: `-AsrMode auto` または `-AsrMode en`
   - 2026-05-23 スモークテスト:
     - 入力: 既存OGGの先頭60秒
     - STT: ReazonSpeech-k2-v2 + VAD、3チャンク、成功3、ASR合計約4.5秒
     - LLM: Qwen3-1.7B Q4_K_M、整形約2.4秒、要約約2.3秒
     - 統合処理: 約15.8秒
     - 出力: `runs\20260523_194338_cpu_pipeline_ja_60sec\output\`
   - 注意: Qwen3は既定だと `<think>` を長く出す場合があるため、`scripts\llm_postprocess.py` で `/no_think` とthink除去を入れて運用する。
   - 2026-05-23 英日混在らしい医療会話データの先頭10分テスト:
     - 入力: `2025-12-21 テスト音声B.ogg`（医療系対話）
     - `-AsrMode auto`: faster-whisper small INT8、言語検出ja p=0.9518、ASR約401秒、統合約502秒
     - `-AsrMode ja`: ReazonSpeech-k2-v2 + VAD、27チャンク、ASR約29秒、統合約53秒
     - `auto` は英語断片を拾える一方、日本語会話で反復・翻訳調・別言語混入が目立った。
     - `ja` は高速で要点を拾うが、英語断片や小声/雑音部は落ちやすく、出力文字数が少ない。
     - 現時点の運用方針: 日本語主体の学校/面談/医療会話は `ja` を基本にし、英語授業や英語が主役の録音だけ `auto` / `en` を使う。英語が少し混じるだけなら、全文を `auto` にするより、後処理辞書または必要部分だけ再処理する設計を検討する。
   - 2026-05-23 ハイブリッド試作:
     - 実装: `run_pipeline_hybrid.ps1` + `scripts\hybrid_english_patch.py`
     - 方式: 先にReazonSpeech-k2-v2で全文処理し、Reazon出力が短すぎるVADチャンクだけfaster-whisper small INT8で補助処理する。
     - 入力: `2025-12-21 テスト音声B.ogg` 先頭10分
     - Reazon全体: 27チャンク、ASR約27.5秒
     - 補助候補: 20チャンク、実処理は上位8チャンク
     - faster-whisper補助: 約94.2秒
     - 統合処理: 約131.5秒
     - 全文`auto`約502秒より大幅に速く、英語断片も一部拾えた。
     - 出力: `hybrid_review_transcript.txt` に `ja_base` と `fw_patch` を併記し、誤置換を避ける。
     - 暫定判断: 日本の英語授業向けには、この「日本語優先 + 英語疑い箇所だけ補助」の方式を第1候補にする。
   - 2026-05-23 ハイブリッド フルレングステスト:
     - 入力: `2025-12-21 テスト音声B.ogg` 全体 2:16:45
     - 比較基準: `Testdata\2025-12-21 テスト音声B.txt`（WhisperX large-v3）
     - Reazon VADチャンク: 342
     - Reazon全文ASR: 約347.1秒
     - 補助候補: 294チャンク
     - faster-whisper補助: 上位80チャンク、約774.7秒
     - 統合処理: 約1279.0秒（約21.3分）
     - 文字量: WhisperX基準 約32,881字、hybrid best-effort 約12,901字、hybrid review 約14,183字
     - 英単語量: WhisperX基準 5,478語、hybrid best-effort 2,062語、hybrid review 2,916語
     - キーワード: 基準に存在した21語のうちbest-effortが15語、reviewが16語を検出
     - レポート: `runs\comparison_hybrid_full_vs_whisperx_2025-12-21_testaudio_b.md`
     - 判断: 速度は全文faster-whisperより現実的だが、WhisperX large-v3相当の完全な文字起こしには届かない。授業支援の「要点把握・英語候補提示」には使えるが、医療記録のような高精度用途では人の確認が必須。
   - 2026-05-24 学校向けLLM整形 v10テスト:
     - 対象: `runs\20260524_005434_hybrid_ja_fw_full\output\hybrid_review_transcript.txt`（テスト音声A: 教育相談の面談）
     - 改修: `scripts\school_hybrid_postprocess.py` に2段階圧縮を追加。
       - part整形後、`part_summaries.md` を作成してから最終統合する。
       - Qwen3-1.7Bが空出力・相づち中心・メタ出力に寄った場合は、元チャンクから抽出的にfallbackする。
       - 入力にない「生徒会」「集会」等の場面ラベルを作らないよう、プロンプト例示を弱め、後処理でも除去する。
     - 実行: `runs\20260524_015851_school_postprocess_hybrid_review_transcript\`
     - バッチ数: 5
     - LLMロード: 約1.38秒
     - part summary生成: 約0.008秒（LLMではなく決定的抽出）
     - final生成: 約50.9秒
     - final fallback: 使用
     - 出力: `output\school_record.md`, `output\part_summaries.md`, `output\partial_records.md`
     - 確認: `school_record.md` に `<think>`、推論文、`[00:00]`、不要な「生徒会」「集会」混入なし。
     - 判断: Qwen3-1.7B単体の自然な最終要約はまだ不安定。現時点では「自然な議事録」より、`part_summaries.md` を含む根拠付き抽出記録として使う方が安全。学校テスト向けには、AIが補完しすぎない設計として有効。
   - 2026-05-24 Qwen3.5-2B学校向け整形テスト:
     - モデル: `models\Qwen3.5-2B-Q4_K_M.gguf`
     - 実行: `runs\20260524_062529_school_postprocess_hybrid_review_transcript\`
     - 対象: `runs\20260524_005434_hybrid_ja_fw_full\output\hybrid_review_transcript.txt`
     - バッチ数: 5
     - LLMロード: 約1.35秒
     - final生成: 約40.9秒
     - final fallback: 不使用
     - 確認済み:
       - `## 整形記録` から `01:05.19 -` 形式のタイムスタンプ除去に成功。
       - `fix_section_heading_bullets()` により、箇条書き化されたセクション見出しを補正。
       - `ensure_3line_summary()` により、3行サマリー欠落・件数不整合を補正。
       - Qwen3.5 `<think>` token ID `248068/248069` を `logit_bias` に追加。
       - `LD（低機能障害）`、`ADHD（注意欠如症候群）` のような危険な既知語展開を `normalize_known_terms()` で補正。
       - `<think>`、推論文、`[00:00]`、不要な「生徒会」「集会」混入なし。
     - 出力: `runs\20260524_062529_school_postprocess_hybrid_review_transcript\output\school_record.md`
     - 判断: Qwen3.5-2BはQwen3-1.7Bより最終統合が安定し、学校向け整形の第1候補にできる可能性が高い。ただし専門語の誤展開は後処理辞書で抑える前提。
   - 2026-05-24 PART生発話欠落対策:
     - 背景: 英語主体のテスト音声C（面接記録音声）では、STT後の `filtered_input.txt` に候補者本人の日本語発話（応募理由や体験、感想など）が残っていたが、PART LLMが形式的な面接手順を優先し、候補者発話を `school_record.md` から落としていた。
     - 改修: `scripts\school_hybrid_postprocess.py`
       - PART promptに「形式的な説明・日程・手順だけを優先しない。本人の理由、体験、感想、意見、希望、困りごと、学びたいこと、伝えたいことを残す」を追加。
       - `extract_personal_statement_items()` と `ensure_personal_statements_in_record()` を追加し、LLMが落とした本人発話を元chunkから決定的に補う。
       - 番号付きリスト `1.` `2.` も箇条書きとして正規化し、重複ループ除去と件数制限の対象にした。
       - `## 英語・専門語・固有名詞` / `## 要確認箇所` の素の行も箇条書きに正規化。
       - 3行サマリーでは、手順説明より本人の理由・体験・希望を優先する。
     - 再テスト: `runs\20260524_091438_school_postprocess_hybrid_review_transcript\`
     - 結果:
       - `school_record.md` に、応募理由や体験、今後への抱負など、候補者本人の具体的な発話内容が反映されるようになった。
       - 1バッチのためFINALバイパス、`final_seconds=0.001`、`final_used_fallback=false`。
   - 残課題:
       - 英語主体音声ではReazon日本語優先STTが大きく崩れるため、このパイプラインは英語本編の完全記録には不適合。
       - 面接手順の一部に「スキューティング視点」などSTT/LLM由来の誤認識が残る。英語主体音声は `faster-whisper auto/en` 別ルートが必要。
   - 2026-05-24 日英交互モード追加:
     - 背景: 英語授業では、先生が英語文を読み、その直後に日本語で意味や背景を説明する流れが多い。日本語主体パイプラインだけでは英語文が「はい」「あれ」などに崩れるため、英語らしいchunkのみ `faster-whisper` で補助するモードを追加。
     - 追加ファイル:
       - `run_pipeline_ja_en_alternating.ps1`
       - `scripts\ja_en_alternating_patch.py`
     - 方式:
       - まず既存の ReazonSpeech K2 VAD で全体をchunk化・日本語認識。
       - Reazon結果が短い、またはフィラー化しているchunkを英語候補として抽出。
       - 候補chunkだけ `faster-whisper small/int8` で英語認識。
       - chunkごとに `[lang_hint] ja/en/mixed/unknown` を付与。
       - 英語chunkと近接する日本語説明chunkを `english_japanese_pairs.md` に抽出。
     - テスト音声: `<開発機のダウンロードフォルダ>\新しいフォルダー (2)\2025-11-02 10_58_44.ogg`
     - 実行結果:
       - 入力長: 約54分37秒。
       - Reazon VAD chunks: 128。
       - 英語patch候補: 28chunk。
       - 言語推定: `ja=103`, `mixed=7`, `en=18`。
       - 英日ペア候補: 25件。
       - 全体処理時間: 約309秒。
       - 出力: `runs\20260524_094150_ja_en_alternating_full\output\alternating_review_transcript.txt`
       - 英日ペア: `runs\20260524_094150_ja_en_alternating_full\output\english_japanese_pairs.md`
     - 評価:
       - 英語文の直後に日本語説明が続く授業形式を検出する用途として有望。
       - `Thank you very much.` と「遠くから来て頂きありがとうございます」、Director of Operations と24時間勤務説明、広告・家事役割の英日対応などを抽出できた。
       - 一方で、短い日本語chunkを英語候補にしすぎると `crime scene`、`baby is pregnant` などの英語patchノイズが混入する。
       - 既存の `run_school_hybrid_postprocess.ps1` にそのまま渡すと、英語patchノイズが `school_record.md` に残る場合がある。
     - 次の調整候補:
       - `ShortTextChars` を16から8程度に下げ、短い日本語を英語patchしすぎないようにする。
       - `faster-whisper` の検出言語・信頼度・Latin単語率でpatch採用を絞る。
       - 日英交互授業用の専用整形出力を作る。例: `## 英文と日本語説明`, `## 授業の主な内容`, `## 重要表現・単語`, `## 要確認箇所`。
   - 2026-05-24 日英交互モード再テスト (`ShortTextChars=8`):
     - 実行: `run_pipeline_ja_en_alternating.ps1 -InputFile "<開発機のダウンロードフォルダ>\新しいフォルダー (2)\2025-11-02 10_58_44.ogg" -MaxDurationSeconds 0 -MaxPatchChunks 240 -ShortTextChars 8 -MinCandidateSeconds 2.0 -PatchLanguage en`
     - 出力: `runs\20260524_100215_ja_en_alternating_full\`
     - 結果:
       - Reazon VAD chunks: 128。
       - 英語patch候補: 18件。前回の28件から減少。
       - 言語推定: `ja=113`, `en=15`。前回の `mixed=7` は消え、より保守的になった。
       - 英日ペア候補: 15件。前回の25件から減少。
       - 全体処理時間: 約416秒。
     - 評価:
       - 前回school_recordに悪影響を与えた `crime scene` と `baby is pregnant` は `alternating_review_transcript.txt` 上から消えた。
       - `Director of Operations`、休憩・ランチ、24時間勤務、広告と家事役割など、授業の核になる英語箇所は維持できた。
       - 一方で `See you in the next video`、単独 `You`、空patchなどの無効patchはまだ残る。
       - `ShortTextChars=8` は現時点のデフォルト候補。ただし最終採用には、patch結果の最小文字数・禁止フレーズ・日本語近接説明の有無でさらに採用フィルタを入れる必要がある。
   - 2026-05-24 Qwen3.5-4B学校向け整形テスト:
     - 参照: Unsloth Qwen3.5 model docs。4Bの4bit目安メモリは約5.5GB。
     - 追加モデル: `models\Qwen3.5-4B-Q4_K_M.gguf` (約2.74GB)。
     - 追加ランナー: `run_qwen35_4b_postprocess_test.ps1`。
     - 注意: 現行プロンプトでは `n_ctx=8192` だと `Requested tokens (9437) exceed context window of 8192` で失敗。4Bテストでは `NCtx=12288` が必要。
     - テスト入力: `runs\20260522_テスト音声A_largev3_transcript.txt`（教育相談の面談、71分）。
     - 4B実行:
       - コマンド: `run_school_hybrid_postprocess.ps1 -TranscriptFile ".\runs\20260522_テスト音声A_largev3_transcript.txt" -ModelFile ".\models\Qwen3.5-4B-Q4_K_M.gguf" -NCtx 12288 -MaxCharsPerBatch 5000 -MaxTokensPart 1200 -MaxTokensFinal 1800`
       - 出力: `runs\20260524_101644_school_postprocess_20260522_テスト音声A_largev3_transcript\`
       - model_load_seconds: 2.661。
       - batch_001_seconds: 368.314。
       - final_used_fallback: false。
       - `<think>`、偽タイムスタンプ、`chunk/base/patch/lang/hint` の混入なし。
       - 内容は簡潔で、専門用語や本人・支援者の発言を精度よく抽出できていた。
       - ただし `## 英語・専門語・固有名詞` と `## 要確認箇所` が欠落し、後処理で3行サマリーのみ補われた。
     - 同条件2B比較:
       - 出力: `runs\20260524_102355_school_postprocess_20260522_テスト音声A_largev3_transcript\`
       - model_load_seconds: 1.359。
       - batch_001_seconds: 149.855。
       - batch_001_fallback_used: true。
       - 2Bは速いが、この条件ではPART fallbackになり、生の `[SPEAKER_00]...` が整形記録に残った。
     - 判断:
       - Qwen3.5-4Bは2Bより約2.5倍遅いが、長め入力でLLM整形に成功しやすい。
       - 学校PC標準は引き続き2Bが現実的。4Bは「品質確認用」「余裕のあるPC用」「2Bがfallbackしたときの再処理候補」として残す。
       - 実運用前には、4Bでもセクション欠落を補う後処理またはプロンプト調整が必要。
   - 2026-05-24 CPU運用時のThinking OFF標準化:
     - 方針: CPU処理ではThinkingモードをOFFにして、推論過程の生成にCPU時間を使わせず、回答生成速度を優先する。
     - 対象:
       - `scripts\school_hybrid_postprocess.py`
       - `scripts\llm_postprocess.py`
       - `scripts\voice_chat.py`
     - 実装:
       - `/no_think` とsystem promptで「思考過程、推論メモ、<think>タグは出力しない」を明示。
       - Qwen3/Qwen3.5の `<think>` / `</think>` token IDを `logit_bias=-100` で抑制。
       - モデル語彙サイズ外のtoken IDは `llm.n_vocab()` で除外し、Qwen3-1.7BとQwen3.5系の両方で安全に動かす。
       - 万一出力された `<think>...</think>` は後処理でも除去。
       - ログとsummary JSONに `thinking_mode=off` を記録する。
     - 判断:
       - 学校PC向けCPU運用ではThinking OFFを標準とする。
       - 品質確認や研究目的でThinkingを使う設定は、通常運用とは別スクリプトまたは明示オプションに分ける。
   - 2026-05-24 Qwen3-1.7B除外:
     - 判断: 今回用途では Qwen3.5-2B が標準、Qwen3.5-4B が再処理候補となったため、Qwen3-1.7B は候補から外す。
     - 削除: `models\Qwen3-1.7B-Q4_K_M.gguf` を削除し、約1.1GBを解放。
     - 現行スクリプトのデフォルトを更新:
       - `run_pipeline_cpu.ps1`: `models\Qwen3.5-2B-Q4_K_M.gguf`
       - `run_voice_chat.ps1`: `models\Qwen3.5-2B-Q4_K_M.gguf`
     - 過去の `runs\` 内ログや実験記録には、当時の再現性のため Qwen3-1.7B の記録を残す。

2. **出力仕様の固定**
   - `raw_transcript.txt`
   - `formatted_transcript.md`
   - `formatted_transcript.txt`
   - `school_record.md`
   - `summary.md`
   - `summary.txt`
   - `metadata.json`
   - `logs/log.txt`
   - 将来GUIから読みやすいファイル構造にする。
   - 2026-05-24 更新:
     - `scripts\school_hybrid_postprocess.py` が `formatted_transcript.md/.txt` を正式出力する。
     - `formatted_transcript` はサマリーではなく、チャンク順の整形文字起こしとして扱う。
     - `summary.md/.txt` は `school_record.md` の `## 3行サマリー` だけを切り出した軽量表示用ファイル。
     - 追加LLM呼び出しは行わず、4GB/8GB端末でも負荷を増やさない。
   - 2026-05-24 Standardプリセット60秒比較:
     - レポート: `runs\preset_comparison_20260524_60sec.md`
     - Japanese / EnglishMixed / English の3系統がASRからLLM後処理まで完走。
     - 60秒サンプルは挨拶・相づち中心で品質比較には不向きだったが、プリセット疎通は確認できた。
     - `empty` が専門語欄・3行サマリーに混入する問題を修正。
     - 次は内容が濃い区間を5分程度で切り出して比較する。
   - 2026-05-24 クリーン文字起こしレイヤー追加:
     - `raw_transcript.txt` を保存して原文を保持。
     - Qwen3.5-2Bで `clean_transcript.md` と `chunks_clean\chunk_XXX_clean.md` を生成。
     - 原文chunkは `chunks_raw\chunk_XXX_raw.txt` に保存。
     - `clean_transcript.md` から学校向け記録を作る流れに変更。
     - `meeting_record.md` と `review_flags.md` を正式出力に追加。
     - 注意書きとして、クリーン文字起こし・記録ともに元音声/原文照合が必要であることを明記。
     - 60秒既存transcriptで出力確認済み: `runs\20260524_220944_school_postprocess_hybrid_review_transcript\`
   - 2026-05-24 Stage 2テンプレート要約を分離:
     - 追加: `scripts\template_summarize.py`
     - 追加: `run_template_summarize.ps1`
     - 入力: `safe_transcript.md`（旧 `clean_transcript.md` は互換扱い）
     - 出力: `summaries\<template>.md`, `summaries\<template>_review_flags.md`, `summaries\<template>_metadata.json`
     - 対応テンプレート:
       - `meeting_record`
       - `support_record`
       - `lesson_record`
       - `self_reflection`
       - `meeting_memo`
       - `interview_record`
     - これにより、Stage 1「文字起こし・クリーン整形」とStage 2「用途別要約・記録化」を分離。
     - 短すぎる/相づちのみのclean transcriptではLLMを起動せず、該当なしテンプレートを生成するガードを追加。
   - 2026-05-24 Stage 1品質チェック:
     - レポート: `runs\stage1_quality_check_20260524_ja_5min.md`
     - 日本語面談の3分地点から5分を `Standard/Japanese` で実行。
     - `RunStage2` を付けない場合は `raw_transcript.txt`, `clean_transcript.md`, `review_flags.md` までで停止する方針を確認。
     - `clean_mode=llm` ではQwen3.5-2Bが「インタビュアー」「面接担当」などを推測して混入したため、Stage 1標準としては不採用。
     - `clean_mode=safe` を標準に変更。Qwenをロードせず、chunk順・時刻付きの安全なクリーン文字起こしを作る。
     - Stage 2は `run_template_summarize.ps1` を明示実行した場合のみ進める。
     - 日英交互5分テスト (`2025-11-02 10_58_44.ogg`, 20:00-25:00) では12 chunks中7 chunksをfaster-whisper smallで補助。英語本文と日本語説明を時系列で保持できた。
     - `review_flags.md` は `fw_patch` 使用チャンクを要確認として列挙するよう修正。英語補助認識は正式利用前に元音声照合する設計にした。
     - レポート: `runs\stage1_quality_check_20260524_englishmixed_5min.md`
     - 同じ日英交互5分で `clean_mode=safe` と `clean_mode=llm` を比較。Qwen整形版はプロンプト混入、英語本文の翻訳、意味変換が見られたため、Stage 1標準はsafeを維持。
     - AI整形版は将来GUIで明示操作した場合のみ `ai_readable_transcript.md` として参考出力にする方針。
     - 比較レポート: `runs\stage1_clean_vs_ai_format_compare_20260524_englishmixed_5min.md`
   - 2026-05-24 Stage 1/2ファイル名整理:
     - Stage 1標準出力を `safe_transcript.md` に変更。
     - 旧互換として `clean_transcript.md` は残すが、新規設計・Stage 2入力では `safe_transcript.md` を使う。
     - `clean_mode=llm` 実行時は `ai_readable_transcript.md` を別出力にする。safe版も必ず同時に保存する。
     - `run_template_summarize.ps1` / `scripts\template_summarize.py` は `safe_transcript.md` 入力を正式対応。旧 `--clean_transcript` は互換扱い。
     - Stage 2授業記録で、Qwenが本文中ラベルにした見出しを正規 `##` 見出しへ補正し、空セクションは `- 該当なし` で埋める後処理を追加。

3. **エラー復旧・再実行**
   - chunk単位の失敗を記録する。
   - 成功済みchunkを再処理しない。
   - 途中停止後に再開できるようにする。

4. **運用・安全ポリシー**
   - 録音ファイルを残すかどうか。
   - 出力の保存先。
   - 削除方法。
   - AI結果は下書きであり人が確認する注意文。

5. **Embeddable Python配布試作**
   - `.venv` で固めた構成を、学校PC向けに持ち運べる形へ移行する。
   - 管理者権限なしで起動できるか確認する。

6. **第2フェーズ: GUI + ローカルTTS**
   - フェーズ1のCLIが安定してから着手する。
   - GUIは「録音選択 → 文字起こし → 整形 → 要約 → 保存/削除」を迷わず実行できる最小構成から始める。

7. **第3フェーズ: 音声Chat**
   - STT、LLM、TTSの低遅延専用モードを別途検証する。
   - 学習サポート用の短い問いかけ応答に限定して実測する。

---

## 9. 環境構成

```
c:\whisper\
├── .venv\                          # Embeddable Python仮想環境
│   └── Lib\site-packages\
│       ├── faster_whisper          # ASR（英語・多言語）
│       ├── reazonspeech_k2_asr     # ASR（日本語専用）
│       ├── llama_cpp               # LLM推論（GGUF）
│       ├── ctranslate2             # CTranslate2バックエンド
│       └── pyttsx3                 # TTS（フォールバック用）
├── engines\
│   ├── ffmpeg\ffmpeg.exe           # 音声変換
│   └── whisper\                    # whisper.cpp バイナリ
├── models\                         # LLM GGUFモデル
├── scripts\                        # Pythonスクリプト
├── runs\                           # テスト結果（自動生成）
└── run_*.ps1                       # PowerShellランナー
```

外部依存:
- **VOICEVOX**: 音声Chat使用時のみ（https://voicevox.hiroshiba.jp/）
- **HuggingFace**: 初回モデルダウンロード時のみ（その後オフライン動作）

---

## 10. 2026-06-23 LearningAccess Windows MVP

- 読み書き困難のある学習者向けに、授業中のノート負担を減らすGUI試作を追加。
- 起動: `run_otoweave.ps1`
- 実装: `otoweave_app\`
- 日本語はReazonSpeech K2、英語はParakeet TDT int8を手動選択する。
- 日英混在時や低速端末向けに「録音」を用意し、リアルタイム自動言語判定や2モデル同時実行は行わない。
- マイクとPyAudioWPatchによるWindows WASAPIループバックを選択可能。
- 録音中は16 kHzモノPCMへ連続保存し、VADは発話区間の時刻だけを作る。無音削除による時刻ずれは発生させない。
- 停止後はOpus 40 kbpsへ変換し、`transcript.json`, `transcript.md`, `marks.json`, `metadata.json` とともにローカル保存する。
- 文字起こしは選択・コピー可能な通常テキストとして表示し、各区間から元音声を再生できる。
- `★ 重要` と `? 要確認` は作成時刻付きで保存する。
- 年、月、ISO週、授業日の順に一覧化し、今週・重要・要確認フィルターを持つ。
- OneDrive連携は授業後のユーザー操作によるフォルダーコピーのみ。自動同期や外部APIは使わない。
- LLM要約、話者分離、リアルタイム自動言語ルーティング、リアルタイム要約はMVP対象外。
- 実マイク録音からOpus変換までのスモークテスト、Reazon/Parakeetのモデルロード、単体テストを確認済み。
- 既存録音の取り込みを追加。OGG/WAV/MP3/M4A/Opus/FLAC/AAC/WMAを先に音声保存し、完了後に言語処理を選択する。
- 取り込み元の録音は変更・削除せず、録音日と元ファイル名をメタデータへ記録する。
- 選択授業の削除を追加。保存ルート配下で必須メタデータを持つ授業フォルダーだけを、確認後に完全削除する。
- 「あとから文字起こし」を追加。音声保存した授業を選び、日本語、逐次2パスの日英混在、英語で後処理できる。
- 既存文字起こしがある場合はテキストの置換確認を表示し、★/?は近い時刻へ引き継ぐ。認識失敗時は既存文字起こしを保持する。
- 初期操作を「リアルタイム字幕」「録音」「録音を取り込む」の3つへ変更。録音と取り込みは音声完成後に言語を選ぶ。
- GIGA端末8GB向けの標準日英混在は、Reazonを全区間に実行して解放後、Parakeetを実行する逐次2パスを維持する。
- Qwen3-ASR-1.7B Q8を16GB以上推奨の実験オプションとして追加。録音後処理時だけ`llama-server`を起動し、処理完了後に解放する。実測ピーク約4.5GB。
- ファイル管理はWindows標準を採用。録音取り込みの標準ファイルダイアログは別プロセスで開き、Explorer/OneDriveが遅い場合もアプリ本体を停止させない。
- 「保存場所を開く」から選択授業をWindows Explorerで管理できる。独自ファイル管理画面は持たない。
- カードの文字起こし訂正を追加し、`corrections.jsonl` に修正日時・修正前・修正後を保存する。
- 2026-06-27 UIを`Live Notes`、`Review & Edit`、`Library`の3画面へ再構成。録音中は文書表示と最小操作だけを見せ、編集・検索・コピー・TXT/Markdown/SRT出力はReviewへ分離した。
- `! 質問`マーカー、話者名編集、区間分割・結合、5秒送り、再生速度、表示サイズ・行間・読み幅・Live Follow設定を後方互換で追加した。
