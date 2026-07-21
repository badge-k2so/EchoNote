"""Tests for the app shell fixes.

Covers (UI なしで検証できる範囲):
  - initial_window_geometry(): GIGA 端末 (1366x768) での画面内クランプ
  - is_cloud_synced_path(): クラウド同期フォルダ検知の純関数
  - cloud_sync_notice_*(): 警告済みフラグの保存と読み込み
  - _maybe_warn_cloud_sync(): 一度だけ警告を表示すること
  - _confirm_save_location(): 保存先選択時のクラウド同期警告の分岐
  - _handle_controller_event(): llm_chat_chunk のストリーミング配線
  - app_logging: RotatingFileHandler 設定と sys/threading excepthook
  - report_callback_exception(): ログ記録とダイアログ抑制
"""
from __future__ import annotations

import logging.handlers
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from otoweave_app import app_logging
from otoweave_app.otoweave_app import (
    OtoWeaveApp,
    cloud_sync_notice_path,
    initial_window_geometry,
    is_cloud_synced_path,
    load_cloud_sync_notice_shown,
    save_cloud_sync_notice_shown,
)


# ---------------------------------------------------------------------------
# 1. initial_window_geometry()
# ---------------------------------------------------------------------------

class InitialWindowGeometryTests(unittest.TestCase):
    def test_large_screen_keeps_default_size(self) -> None:
        self.assertEqual(
            initial_window_geometry(1920, 1080),
            (1440, 860, 1120, 680),
        )

    def test_giga_screen_1366x768_fits_inside(self) -> None:
        width, height, min_w, min_h = initial_window_geometry(1366, 768)
        self.assertEqual((width, height), (1326, 668))
        self.assertLessEqual(width, 1366 - 40)
        self.assertLessEqual(height, 768 - 100)
        # minsize must not force the window back over the screen edge.
        self.assertLessEqual(min_w, width)
        self.assertLessEqual(min_h, height)

    def test_small_screen_clamps_minsize_too(self) -> None:
        width, height, min_w, min_h = initial_window_geometry(1024, 768)
        self.assertEqual(width, 1024 - 40)
        self.assertLessEqual(min_w, width)
        self.assertLessEqual(min_h, height)

    def test_tiny_screen_keeps_usable_floor(self) -> None:
        width, height, min_w, min_h = initial_window_geometry(500, 400)
        self.assertGreaterEqual(width, 640)
        self.assertGreaterEqual(height, 480)
        self.assertLessEqual(min_w, width)
        self.assertLessEqual(min_h, height)


# ---------------------------------------------------------------------------
# 2. is_cloud_synced_path()
# ---------------------------------------------------------------------------

class CloudSyncedPathTests(unittest.TestCase):
    def test_plain_documents_path_is_not_synced(self) -> None:
        self.assertFalse(
            is_cloud_synced_path(
                r"C:\Users\student\Documents\LearningAccess",
                environ={},
            )
        )

    def test_path_under_onedrive_env_var(self) -> None:
        # 環境変数の指すルート配下なら、フォルダ名に依存せず検知する。
        self.assertTrue(
            is_cloud_synced_path(
                r"C:\Users\student\SyncRoot\Documents\LearningAccess",
                environ={"OneDrive": r"C:\Users\student\SyncRoot"},
            )
        )

    def test_env_var_match_is_case_insensitive(self) -> None:
        self.assertTrue(
            is_cloud_synced_path(
                r"c:\users\student\syncroot\LearningAccess",
                environ={"OneDriveCommercial": r"C:\Users\Student\SyncRoot"},
            )
        )

    def test_sibling_folder_of_env_root_is_not_synced(self) -> None:
        self.assertFalse(
            is_cloud_synced_path(
                r"C:\Users\student\SyncRoot2\LearningAccess",
                environ={"OneDriveConsumer": r"C:\Users\student\SyncRoot"},
            )
        )

    def test_env_root_itself_is_synced(self) -> None:
        self.assertTrue(
            is_cloud_synced_path(
                r"C:\Users\student\SyncRoot",
                environ={"OneDrive": r"C:\Users\student\SyncRoot"},
            )
        )

    def test_empty_env_value_is_ignored(self) -> None:
        self.assertFalse(
            is_cloud_synced_path(
                r"C:\Users\student\Documents\LearningAccess",
                environ={"OneDrive": ""},
            )
        )

    def test_onedrive_keyword_in_path(self) -> None:
        self.assertTrue(
            is_cloud_synced_path(
                r"C:\Users\student\OneDrive - 市立学校\Documents\LearningAccess",
                environ={},
            )
        )

    def test_google_drive_japanese_keyword(self) -> None:
        self.assertTrue(
            is_cloud_synced_path(
                r"G:\マイドライブ\Google ドライブ\LearningAccess",
                environ={},
            )
        )

    def test_googledrive_keyword(self) -> None:
        self.assertTrue(
            is_cloud_synced_path(
                r"D:\GoogleDrive\LearningAccess",
                environ={},
            )
        )

    def test_dropbox_keyword_case_insensitive(self) -> None:
        self.assertTrue(
            is_cloud_synced_path(
                r"C:\Users\student\DROPBOX\LearningAccess",
                environ={},
            )
        )

    def test_google_drive_for_desktop_user_folder(self) -> None:
        # Google Drive for Desktop の標準マウント（スペース入り）を検知する。
        self.assertTrue(
            is_cloud_synced_path(
                r"C:\Users\student\Google Drive\LearningAccess",
                environ={},
            )
        )

    def test_google_drive_for_desktop_my_drive_letter(self) -> None:
        self.assertTrue(
            is_cloud_synced_path(
                r"G:\My Drive\LearningAccess",
                environ={},
            )
        )

    def test_google_drive_japanese_my_drive(self) -> None:
        self.assertTrue(
            is_cloud_synced_path(
                r"G:\マイドライブ\LearningAccess",
                environ={},
            )
        )


# ---------------------------------------------------------------------------
# 3. Cloud-sync notice flag persistence
# ---------------------------------------------------------------------------

class CloudSyncNoticeFlagTests(unittest.TestCase):
    def test_flag_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = cloud_sync_notice_path(Path(temporary))
            self.assertFalse(load_cloud_sync_notice_shown(path))
            save_cloud_sync_notice_shown(path)
            self.assertTrue(load_cloud_sync_notice_shown(path))

    def test_broken_flag_file_treated_as_not_shown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = cloud_sync_notice_path(Path(temporary))
            path.write_text("not json", encoding="utf-8")
            self.assertFalse(load_cloud_sync_notice_shown(path))


class MaybeWarnCloudSyncTests(unittest.TestCase):
    @staticmethod
    def _app(root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            controller=SimpleNamespace(store=SimpleNamespace(root=root)),
        )

    def test_warns_once_and_saves_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = self._app(root)
            with patch(
                "otoweave_app.otoweave_app.is_cloud_synced_path",
                return_value=True,
            ), patch(
                "otoweave_app.otoweave_app.messagebox"
            ) as fake_box:
                OtoWeaveApp._maybe_warn_cloud_sync(app)
                OtoWeaveApp._maybe_warn_cloud_sync(app)
            self.assertEqual(fake_box.showwarning.call_count, 1)
            self.assertTrue(
                load_cloud_sync_notice_shown(cloud_sync_notice_path(root))
            )
            # 生徒向け文言に内部用語（例外名など）を混ぜていないこと。
            message = fake_box.showwarning.call_args.args[1]
            self.assertIn("保存先", message)
            self.assertIn("OneDrive", message)

    def test_no_warning_for_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            app = self._app(Path(temporary))
            with patch(
                "otoweave_app.otoweave_app.is_cloud_synced_path",
                return_value=False,
            ), patch(
                "otoweave_app.otoweave_app.messagebox"
            ) as fake_box:
                OtoWeaveApp._maybe_warn_cloud_sync(app)
            fake_box.showwarning.assert_not_called()

    def test_no_warning_without_controller(self) -> None:
        app = SimpleNamespace(controller=None)
        with patch(
            "otoweave_app.otoweave_app.messagebox"
        ) as fake_box:
            OtoWeaveApp._maybe_warn_cloud_sync(app)
        fake_box.showwarning.assert_not_called()

    def test_flag_saved_only_after_dialog_was_shown(self) -> None:
        # ダイアログを出す前に落ちた場合、次回もう一度警告できること。
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = self._app(root)
            with patch(
                "otoweave_app.otoweave_app.is_cloud_synced_path",
                return_value=True,
            ), patch(
                "otoweave_app.otoweave_app.messagebox"
            ) as fake_box:
                fake_box.showwarning.side_effect = RuntimeError("表示前に異常終了")
                with self.assertRaises(RuntimeError):
                    OtoWeaveApp._maybe_warn_cloud_sync(app)
            self.assertFalse(
                load_cloud_sync_notice_shown(cloud_sync_notice_path(root))
            )


# ---------------------------------------------------------------------------
# 3b. 保存先フォルダ選択後のクラウド同期確認
# ---------------------------------------------------------------------------

class ConfirmSaveLocationTests(unittest.TestCase):
    def test_local_folder_is_accepted_without_dialog(self) -> None:
        app = SimpleNamespace()
        with patch(
            "otoweave_app.otoweave_app.is_cloud_synced_path",
            return_value=False,
        ), patch(
            "otoweave_app.otoweave_app.messagebox"
        ) as fake_box:
            result = OtoWeaveApp._confirm_save_location(
                app, Path(r"C:\Users\student\Documents\LearningAccess")
            )
        self.assertTrue(result)
        fake_box.askyesno.assert_not_called()

    def test_cloud_folder_warns_and_can_continue(self) -> None:
        app = SimpleNamespace()
        with patch(
            "otoweave_app.otoweave_app.is_cloud_synced_path",
            return_value=True,
        ), patch(
            "otoweave_app.otoweave_app.messagebox"
        ) as fake_box:
            fake_box.askyesno.return_value = True
            result = OtoWeaveApp._confirm_save_location(
                app, Path(r"G:\My Drive\LearningAccess")
            )
        self.assertTrue(result)
        fake_box.askyesno.assert_called_once()
        # 生徒向けの平易な文言で、選び直せることが分かること。
        message = fake_box.askyesno.call_args.args[1]
        self.assertIn("インターネット", message)
        self.assertIn("先生", message)
        self.assertIn("選び直せ", message)

    def test_cloud_folder_warns_and_can_reselect(self) -> None:
        app = SimpleNamespace()
        with patch(
            "otoweave_app.otoweave_app.is_cloud_synced_path",
            return_value=True,
        ), patch(
            "otoweave_app.otoweave_app.messagebox"
        ) as fake_box:
            fake_box.askyesno.return_value = False
            result = OtoWeaveApp._confirm_save_location(
                app, Path(r"G:\My Drive\LearningAccess")
            )
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# 4. llm_chat_chunk event wiring
# ---------------------------------------------------------------------------

class ChatStreamingEventTests(unittest.TestCase):
    @staticmethod
    def _app(folder: Path) -> SimpleNamespace:
        return SimpleNamespace(
            right_pane=SimpleNamespace(
                append_answer=Mock(),
                append_answer_chunk=Mock(),
                set_thinking=Mock(),
            ),
            main_pane=SimpleNamespace(update_status=Mock()),
            _active_folder=folder,
        )

    def test_chunk_event_appends_to_streaming_bubble(self) -> None:
        folder = Path("/tmp/lesson")
        app = self._app(folder)
        OtoWeaveApp._handle_controller_event(
            app, "llm_chat_chunk", ("光合成とは", folder)
        )
        app.right_pane.append_answer_chunk.assert_called_once_with("光合成とは")
        app.right_pane.append_answer.assert_not_called()

    def test_chunk_for_other_note_is_ignored(self) -> None:
        # ノートを切り替えた後に届く旧ノート宛ての差分は表示しない。
        app = self._app(Path("/tmp/lesson_b"))
        OtoWeaveApp._handle_controller_event(
            app, "llm_chat_chunk", ("旧ノートの答え", Path("/tmp/lesson_a"))
        )
        app.right_pane.append_answer_chunk.assert_not_called()

    def test_chunks_then_done_finishes_with_full_answer(self) -> None:
        folder = Path("/tmp/lesson")
        app = self._app(folder)
        OtoWeaveApp._handle_controller_event(
            app, "llm_chat_chunk", ("光合", folder)
        )
        OtoWeaveApp._handle_controller_event(
            app, "llm_chat_chunk", ("成です", folder)
        )
        OtoWeaveApp._handle_controller_event(
            app, "llm_chat_done", ("光合成です", folder)
        )
        self.assertEqual(app.right_pane.append_answer_chunk.call_count, 2)
        app.right_pane.append_answer.assert_called_once_with("光合成です")

    def test_chunk_payload_is_stringified(self) -> None:
        folder = Path("/tmp/lesson")
        app = self._app(folder)
        OtoWeaveApp._handle_controller_event(
            app, "llm_chat_chunk", (123, folder)
        )
        app.right_pane.append_answer_chunk.assert_called_once_with("123")


# ---------------------------------------------------------------------------
# 5. app_logging
# ---------------------------------------------------------------------------

class AppLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._sys_hook = sys.excepthook
        self._thread_hook = threading.excepthook
        self._hooks_installed = app_logging._hooks_installed
        # 各テストがフック設置から検証できるよう毎回リセットする。
        app_logging._hooks_installed = False
        sys.excepthook = Mock()
        threading.excepthook = Mock()

    def tearDown(self) -> None:
        app_logging.shutdown_logging()
        app_logging._hooks_installed = self._hooks_installed
        sys.excepthook = self._sys_hook
        threading.excepthook = self._thread_hook

    @staticmethod
    def _exc_info() -> tuple:
        try:
            raise ValueError("boom for test")
        except ValueError:
            return sys.exc_info()

    def test_setup_creates_rotating_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_dir = Path(temporary) / "logs"
            logger = app_logging.setup_logging(log_dir)
            handler = app_logging._active_handler
            self.assertIsInstance(
                handler, logging.handlers.RotatingFileHandler
            )
            self.assertEqual(handler.maxBytes, app_logging.MAX_BYTES)
            self.assertEqual(handler.backupCount, app_logging.BACKUP_COUNT)
            logger.info("起動テスト")
            log_file = log_dir / app_logging.LOG_FILE_NAME
            self.assertTrue(log_file.exists())
            self.assertIn(
                "起動テスト", log_file.read_text(encoding="utf-8")
            )
            app_logging.shutdown_logging()

    def test_sys_excepthook_logs_and_chains(self) -> None:
        previous = sys.excepthook
        with tempfile.TemporaryDirectory() as temporary:
            log_dir = Path(temporary) / "logs"
            app_logging.setup_logging(log_dir)
            self.assertIsNot(sys.excepthook, previous)
            info = self._exc_info()
            sys.excepthook(*info)
            content = (log_dir / app_logging.LOG_FILE_NAME).read_text(
                encoding="utf-8"
            )
            self.assertIn("未捕捉の例外", content)
            self.assertIn("ValueError", content)
            self.assertIn("boom for test", content)
            previous.assert_called_once_with(*info)
            app_logging.shutdown_logging()

    def test_threading_excepthook_logs_and_chains(self) -> None:
        previous = threading.excepthook
        with tempfile.TemporaryDirectory() as temporary:
            log_dir = Path(temporary) / "logs"
            app_logging.setup_logging(log_dir)
            self.assertIsNot(threading.excepthook, previous)
            exc_type, exc_value, exc_traceback = self._exc_info()
            args = SimpleNamespace(
                exc_type=exc_type,
                exc_value=exc_value,
                exc_traceback=exc_traceback,
                thread=None,
            )
            threading.excepthook(args)
            content = (log_dir / app_logging.LOG_FILE_NAME).read_text(
                encoding="utf-8"
            )
            self.assertIn("スレッド内で未捕捉の例外", content)
            self.assertIn("boom for test", content)
            previous.assert_called_once_with(args)
            app_logging.shutdown_logging()

    def test_second_setup_switches_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            app_logging.setup_logging(first)
            app_logging.get_logger().info("最初のログ")
            app_logging.setup_logging(second)
            app_logging.get_logger().info("切替後のログ")
            first_text = (first / app_logging.LOG_FILE_NAME).read_text(
                encoding="utf-8"
            )
            second_text = (second / app_logging.LOG_FILE_NAME).read_text(
                encoding="utf-8"
            )
            self.assertIn("最初のログ", first_text)
            self.assertNotIn("切替後のログ", first_text)
            self.assertIn("切替後のログ", second_text)
            app_logging.shutdown_logging()

    def test_hooks_installed_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            app_logging.setup_logging(Path(temporary) / "a")
            hook_after_first = sys.excepthook
            app_logging.setup_logging(Path(temporary) / "b")
            self.assertIs(sys.excepthook, hook_after_first)
            app_logging.shutdown_logging()


# ---------------------------------------------------------------------------
# 6. report_callback_exception()
# ---------------------------------------------------------------------------

class ReportCallbackExceptionTests(unittest.TestCase):
    def test_logs_and_shows_plain_dialog_once_per_window(self) -> None:
        app = SimpleNamespace(_last_error_dialog_at=0.0)
        info = AppLoggingTests._exc_info()
        with patch(
            "otoweave_app.otoweave_app.log_exception"
        ) as fake_log, patch(
            "otoweave_app.otoweave_app.messagebox"
        ) as fake_box:
            OtoWeaveApp.report_callback_exception(app, *info)
            OtoWeaveApp.report_callback_exception(app, *info)
        self.assertEqual(fake_log.call_count, 2)
        # 10 秒以内の連続エラーではダイアログを 1 回に抑える。
        self.assertEqual(fake_box.showerror.call_count, 1)
        # 生徒向けダイアログに例外メッセージ（内部用語）を出さない。
        dialog_text = fake_box.showerror.call_args.args[1]
        self.assertNotIn("ValueError", dialog_text)
        self.assertNotIn("boom", dialog_text)


if __name__ == "__main__":
    unittest.main()
