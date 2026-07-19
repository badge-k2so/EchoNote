#!/usr/bin/env bash
# OtoWeave — Apple Silicon (macOS) 初回セットアップ
#
# Homebrew で python@3.12 / ffmpeg を用意し、.venv を作成して依存関係を
# インストールします。llama-cpp-python は Metal (GPU) を有効にしてソース
# からビルドし、AI要約・チャットの動作を高速化します。
#
# 実行方法（ターミナルで、このファイルがあるフォルダに移動してから）:
#   chmod +x setup_mac.sh   ← 初回だけ（「実行権限が無い」と言われた場合）
#   ./setup_mac.sh
#
# 完了すると、最後に verify_setup_mac.sh が自動で実行され、
# 「起動する前に何が足りないか」を日本語で表示します。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

fail() {
  echo "" >&2
  echo "==================================================" >&2
  echo "[エラー] $1" >&2
  echo "==================================================" >&2
  echo "" >&2
  echo "解決しない場合は、上に表示された内容ごと開発者に連絡してください。" >&2
  exit 1
}

echo "==> 環境を確認しています"
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
if [ "$OS_NAME" != "Darwin" ]; then
  fail "このスクリプトは macOS 専用です（検出したOS: $OS_NAME）。"
fi

IS_APPLE_SILICON=0
if [ "$ARCH_NAME" = "arm64" ]; then
  IS_APPLE_SILICON=1
  echo "   Apple Silicon (arm64) を検出しました。llama-cpp-python は Metal を有効にしてビルドします。"
else
  echo "   Intel Mac ($ARCH_NAME) を検出しました。Metal による高速化は使われません（動作はしますが要約・チャットが低速になります）。"
fi

echo "==> Homebrew を確認しています"
if ! command -v brew >/dev/null 2>&1; then
  fail "Homebrew（Macのパッケージ管理ツール）が見つかりません。
まずターミナルで下記を実行してインストールしてください。

  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"

インストールの最後に表示される「Next steps」の案内（PATHの設定）に従ってから、
ターミナルを開き直して ./setup_mac.sh を再実行してください。"
fi

echo "==> python@3.12 を確認・導入しています（brew）"
if ! brew list python@3.12 >/dev/null 2>&1; then
  brew install python@3.12 || fail "python@3.12 の導入に失敗しました。
'brew install python@3.12' を手動で実行し、表示されたエラー内容を開発者に共有してください。"
fi

echo "==> ffmpeg を確認・導入しています（brew）"
if ! brew list ffmpeg >/dev/null 2>&1; then
  brew install ffmpeg || fail "ffmpeg の導入に失敗しました。
'brew install ffmpeg' を手動で実行し、表示されたエラー内容を開発者に共有してください。"
fi

PYTHON_BIN="$(brew --prefix python@3.12 2>/dev/null || true)/bin/python3.12"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3.12"
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "python3.12 が見つかりませんでした（探した場所: $PYTHON_BIN）。
'brew doctor' を実行し、結果を開発者に共有してください。"

echo "==> 仮想環境 (.venv) を作成しています"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv || fail ".venv の作成に失敗しました。"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> pip を更新しています"
python -m pip install --upgrade pip wheel >/dev/null || fail "pip の更新に失敗しました。ネット接続を確認してください。"

echo "==> 依存関係をインストールしています（requirements_mac.txt）"
pip install -r requirements_mac.txt || fail "依存関係のインストールに失敗しました。
ネット接続を確認するか、上に表示されたパッケージ名とエラー内容を開発者に共有してください。"

echo "==> llama-cpp-python をインストールしています（AI要約・チャット用。数分かかることがあります）"
if [ "$IS_APPLE_SILICON" = "1" ]; then
  if ! CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-binary llama-cpp-python; then
    fail "llama-cpp-python の Metal ビルドに失敗しました。
よくある原因: Xcode Command Line Tools が未導入です。次を実行してから再度 ./setup_mac.sh を試してください。

  xcode-select --install

それでも失敗する場合は、上に表示されたエラーの最後の20〜30行を開発者に共有してください。"
  fi
else
  pip install llama-cpp-python || fail "llama-cpp-python のインストールに失敗しました。"
fi

echo "==> ReazonSpeech K2（日本語文字起こしエンジン）をインストールしています"
if pip install "git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/k2-asr"; then
  echo "   OK: reazonspeech-k2-asr をインストールしました。"
else
  echo ""
  echo "--------------------------------------------------------" >&2
  echo "[警告] reazonspeech-k2-asr のインストールに失敗しました。" >&2
  echo "  日本語のリアルタイム文字起こしが使えない可能性があります" >&2
  echo "  （英語(Parakeet)・要約・チャット・読み上げ等はこの警告と無関係に動作します）。" >&2
  echo "  対処方法:" >&2
  echo "    1) いったんネット接続を確認し、次のコマンドを再実行する:" >&2
  echo "       pip install \"git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/k2-asr\"" >&2
  echo "    2) それでも失敗する場合は開発者に連絡し、代わりの配布ファイル（wheel）を" >&2
  echo "       受け取って 'pip install <受け取ったファイルのパス>' でインストールする" >&2
  echo "  （このセットアップ自体はこのまま続行します）" >&2
  echo "--------------------------------------------------------" >&2
  echo ""
fi

echo ""
echo "==> セットアップの検証を実行しています (verify_setup_mac.sh)"
if [ -f "./verify_setup_mac.sh" ]; then
  bash "./verify_setup_mac.sh" || true
else
  echo "   verify_setup_mac.sh が見つかりませんでした（スキップします）。" >&2
fi

echo ""
echo "=================================================="
echo "セットアップスクリプトが完了しました。"
echo "起動するには:  ./run_otoweave.sh"
echo ""
echo "注意: モデルファイル（AIの本体データ）は Git に含まれていません。"
echo "      開発者から受け取った models/ と hf-cache/ をこのフォルダに"
echo "      配置してから起動してください（詳しくは配布時の手順書を参照）。"
echo "=================================================="
