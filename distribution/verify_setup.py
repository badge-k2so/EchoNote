"""OtoWeave 配布パッケージのセットアップ検証。

テスト機で setup_test_pc.ps1 の最後に実行され、
「起動する前に何が欠けているか」を日本語で一覧表示します。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# 日本語Windowsのコンソール(cp932)でも文字化け・例外なく表示する
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 既定ではこのファイル自身のフォルダ（配布パッケージではmodels/engines等と同じ階層）
# を基準にする。git clone構成のように、models/engines がこのファイルより
# 一つ上の階層（リポジトリ直下）にある場合は、setup_easy.ps1 が
# OTOWEAVE_VERIFY_ROOT 環境変数でこの基準フォルダを上書きする。
ROOT = Path(os.environ.get("OTOWEAVE_VERIFY_ROOT") or Path(__file__).resolve().parent)
RESULTS: list[tuple[bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((ok, label + (f" : {detail}" if detail else "")))


def check_python() -> None:
    version = sys.version_info
    check(
        f"Python {version.major}.{version.minor}.{version.micro}",
        (version.major, version.minor) == (3, 12),
        "" if (version.major, version.minor) == (3, 12) else "3.12 が必要です",
    )


def check_imports() -> None:
    modules = [
        ("customtkinter", "UI"),
        ("tkinterdnd2", "ドラッグ＆ドロップ"),
        ("PIL", "アイコン描画"),
        ("numpy", "音声処理"),
        ("scipy", "リサンプリング"),
        ("sounddevice", "音声再生"),
        ("pyaudiowpatch", "録音（マイク/PC音声）"),
        ("sherpa_onnx", "文字起こしエンジン"),
        ("onnxruntime", "言語判定エンジン"),
        ("huggingface_hub", "モデル読み込み"),
        ("llama_cpp", "AI要約・チャット"),
        ("reazonspeech.k2.asr", "日本語文字起こし"),
    ]
    for name, purpose in modules:
        try:
            __import__(name)
            check(f"ライブラリ {name}（{purpose}）", True)
        except Exception as exc:
            check(f"ライブラリ {name}（{purpose}）", False, str(exc)[:80])


def check_model_files() -> None:
    targets = [
        (
            "英語ASR (Parakeet)",
            ROOT / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8" / "encoder.int8.onnx",
        ),
        (
            "言語判定 (SpeechBrain)",
            ROOT / "models" / "speechbrain-lang-id-voxlingua107-ecapa-onnx" / "lang-id-ecapa.onnx",
        ),
        ("AIチャット (Qwen3.5-2B)", ROOT / "models" / "Qwen3.5-2B-Q4_K_M.gguf"),
        ("ffmpeg", ROOT / "engines" / "ffmpeg" / "ffmpeg.exe"),
    ]
    for label, path in targets:
        check(f"ファイル {label}", path.is_file(), "" if path.is_file() else str(path))
    # 4B は「任意」扱い: Lite 版では同梱されず、標準版でも
    # setup_test_pc.ps1 の自動判定（メモリ11.5GB以下など）で削除される。
    # 4B が無い構成では AI要約が「準備中」表示になるのが正常
    # （2Bで代替生成はしない。チャットは2Bで動作する）。
    qwen_4b = ROOT / "models" / "Qwen3.5-4B-Q4_K_M.gguf"
    qwen_2b = ROOT / "models" / "Qwen3.5-2B-Q4_K_M.gguf"
    if qwen_4b.is_file():
        check("ファイル AI要約 (Qwen3.5-4B) ※任意", True)
    else:
        check(
            "ファイル AI要約 (Qwen3.5-4B) ※任意",
            qwen_2b.is_file(),
            "なし（Lite版またはセットアップの自動判定で削除済み。"
            "AI要約は「準備中」表示になりますが正常です・チャットは利用可）",
        )


def check_disk_space() -> None:
    import shutil as shutil_module

    try:
        free_bytes = shutil_module.disk_usage(ROOT).free
        free_gb = free_bytes / 1024**3
        check(
            f"ディスク空き容量（{free_gb:.1f} GB）",
            free_gb >= 2.0,
            "" if free_gb >= 2.0 else "2GB以上の空きを確保してください",
        )
    except Exception as exc:
        check("ディスク空き容量", False, str(exc)[:80])


def check_reazonspeech() -> None:
    os.environ["HF_HOME"] = str(ROOT / "hf-cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from huggingface_hub import snapshot_download

        model_dir = Path(
            snapshot_download(
                "reazon-research/reazonspeech-k2-v2",
                local_files_only=True,
            )
        )
        encoder = model_dir / "encoder-epoch-99-avg-1.int8.onnx"
        check("日本語ASR (ReazonSpeech K2)", encoder.is_file(), str(model_dir))
    except Exception as exc:
        check("日本語ASR (ReazonSpeech K2)", False, str(exc)[:100])


def check_tts_voice() -> None:
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Add-Type -AssemblyName System.Speech;"
                "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                ".GetInstalledVoices() | ForEach-Object "
                "{ $_.VoiceInfo.Culture.Name }",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        ok = "ja-JP" in (result.stdout or "")
        check("読み上げ用 日本語音声 (Haruka)", ok, "" if ok else "日本語音声が見つかりません")
    except Exception as exc:
        check("読み上げ用 日本語音声 (Haruka)", False, str(exc)[:80])


def check_microphone() -> None:
    try:
        sys.path.insert(0, str(ROOT))
        from otoweave_app.audio import available_audio_sources

        sources = available_audio_sources()
        microphones = [s for s in sources if s.kind == "microphone"]
        check(
            f"マイク入力（{len(microphones)}台検出）",
            bool(microphones),
            "" if microphones else "マイクが見つかりません",
        )
    except Exception as exc:
        check("マイク入力", False, str(exc)[:80])


def write_report(lines: list[str]) -> None:
    """検証結果を setup_report.txt にも保存する（テスト結果の回収用）。"""
    report_path = ROOT / "setup_report.txt"
    try:
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"この結果を {report_path.name} に保存しました。")
    except Exception as exc:
        print(f"結果ファイルの保存に失敗しました: {str(exc)[:80]}")


def main() -> int:
    import datetime

    header = [
        "=" * 56,
        "OtoWeave セットアップ検証",
        "実行日時: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "=" * 56,
    ]
    for line in header:
        print(line)
    check_python()
    check_imports()
    check_model_files()
    check_reazonspeech()
    check_tts_voice()
    check_microphone()
    check_disk_space()

    lines = list(header)
    failures = 0
    for ok, message in RESULTS:
        line = ("[OK] " if ok else "[NG] ") + message
        print(line)
        lines.append(line)
        if not ok:
            failures += 1
    print("-" * 56)
    lines.append("-" * 56)
    if failures:
        summary = f"NG {failures}件: 上の [NG] を解決してから起動してください。"
    else:
        summary = "すべてOKです。「OtoWeaveを起動.bat」で起動できます。"
    print(summary)
    lines.append(summary)
    write_report(lines)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
