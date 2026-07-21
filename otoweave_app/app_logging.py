"""OtoWeave 共通ログ設定とクラッシュ捕捉。

pythonw.exe で起動すると標準エラー出力が見えず、起動失敗や未捕捉例外が
無音で消えるため、ローテーションするログファイルに記録する。

個人情報保護のため、文字起こし本文や生徒の発話・入力テキストは
ログに出力しない（例外メッセージとスタックトレースのみを記録する）。
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

LOGGER_NAME = "otoweave"
LOG_FILE_NAME = "otoweave.log"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3

_active_handler: logging.Handler | None = None
_hooks_installed = False


def get_logger() -> logging.Logger:
    """アプリ共通のロガーを返す。"""
    return logging.getLogger(LOGGER_NAME)


def default_log_dir() -> Path:
    """データルート確定前でも書き込める一時的なログ置き場。"""
    base = os.environ.get("LOCALAPPDATA", "").strip() or tempfile.gettempdir()
    return Path(base) / "OtoWeave" / "logs"


def setup_logging(log_dir: Path | str) -> logging.Logger:
    """<log_dir>/otoweave.log へのローテーションログを設定する。

    再度呼ぶと出力先を切り替える（起動直後は一時置き場、データルート
    確定後に <データルート>/logs へ切替える使い方を想定）。あわせて
    sys.excepthook / threading.excepthook を設定し、未捕捉例外を残す。
    """
    global _active_handler
    logger = get_logger()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        directory / LOG_FILE_NAME,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    )
    if _active_handler is not None:
        logger.removeHandler(_active_handler)
        _active_handler.close()
    logger.addHandler(handler)
    _active_handler = handler
    install_crash_hooks()
    return logger


def log_exception(message: str, exc_info: Any) -> None:
    """例外をスタックトレース付きでログに残す（本文テキストは渡さないこと）。"""
    try:
        get_logger().error(message, exc_info=exc_info)
    except Exception:
        pass


def install_crash_hooks() -> None:
    """未捕捉例外を sys / threading のフックでログへ流す（多重設定しない）。"""
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True

    previous_sys_hook = sys.excepthook

    def _sys_hook(exc_type, exc_value, exc_traceback):
        if not issubclass(exc_type, KeyboardInterrupt):
            log_exception(
                "未捕捉の例外が発生しました",
                (exc_type, exc_value, exc_traceback),
            )
        previous_sys_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _sys_hook

    previous_thread_hook = threading.excepthook

    def _thread_hook(args):
        if args.exc_type is not SystemExit:
            log_exception(
                "スレッド内で未捕捉の例外が発生しました",
                (args.exc_type, args.exc_value, args.exc_traceback),
            )
        previous_thread_hook(args)

    threading.excepthook = _thread_hook


def shutdown_logging() -> None:
    """現在のログハンドラを閉じて外す（主にテスト用）。"""
    global _active_handler
    if _active_handler is not None:
        get_logger().removeHandler(_active_handler)
        _active_handler.close()
        _active_handler = None
