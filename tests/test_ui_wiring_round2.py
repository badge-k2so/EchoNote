"""Round 2 の UI 配線テスト（UI を起動せずに検証できる範囲）。

Covers:
  - summary_progress_text(): summary_progress イベントの progress dict →
    生徒向け表示文字列への変換（未知 stage のフォールバック含む）
  - _handle_controller_event(): summary_progress / llm_error /
    llm_chat_error の配線
  - summary_error_status(): 技術的なエラー文を画面に出さない
  - build_friendly_error() / _show_friendly_error(): 例外はログへ、
    ダイアログには平易な日本語だけ
  - processing_indicator_state(): 取り込み・再文字起こしの中止ボタンの
    状態遷移
  - _on_cancel_processing(): controller.cancel_transcription() との接続
  - close_wait_status(): アプリ終了待ちの文言切替
  - audio_level_text(): マイクテスト結果に RMS/Peak の生値を出さない
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from otoweave_app import app_logging
from otoweave_app.customtkinter_views import (
    audio_level_text,
    processing_indicator_state,
)
from otoweave_app.otoweave_app import (
    CLOSE_WAIT_NOTICE_SECONDS,
    FRIENDLY_ERROR_LOG_NOTE,
    OtoWeaveApp,
    build_friendly_error,
    close_wait_status,
    summary_error_status,
    summary_progress_text,
)


# ---------------------------------------------------------------------------
# 1. summary_progress_text()
# ---------------------------------------------------------------------------

class SummaryProgressTextTests(unittest.TestCase):
    def test_load_stage(self) -> None:
        self.assertEqual(
            summary_progress_text({"stage": "load"}),
            "AIを準備しています…",
        )

    def test_part_stage_with_counts(self) -> None:
        self.assertEqual(
            summary_progress_text({"stage": "part", "current": 2, "total": 5}),
            "要約を作っています… (2/5)",
        )

    def test_all_known_stages_have_japanese_labels(self) -> None:
        for stage in ("load", "clean", "part", "compact", "merge"):
            text = summary_progress_text({"stage": stage})
            self.assertNotIn(stage, text, msg=stage)
            self.assertTrue(text.endswith("…"), msg=stage)

    def test_unknown_stage_falls_back(self) -> None:
        self.assertEqual(
            summary_progress_text({"stage": "quantize"}),
            "要約を処理中…",
        )

    def test_unknown_stage_keeps_counts(self) -> None:
        self.assertEqual(
            summary_progress_text(
                {"stage": "quantize", "current": 1, "total": 3}
            ),
            "要約を処理中… (1/3)",
        )

    def test_non_mapping_payload_falls_back(self) -> None:
        self.assertEqual(summary_progress_text(None), "要約を処理中…")
        self.assertEqual(summary_progress_text("part"), "要約を処理中…")

    def test_missing_counts_show_label_only(self) -> None:
        self.assertEqual(
            summary_progress_text({"stage": "merge"}),
            "要約を仕上げています…",
        )

    def test_numeric_strings_are_accepted(self) -> None:
        self.assertEqual(
            summary_progress_text(
                {"stage": "clean", "current": "1", "total": "4"}
            ),
            "文字起こしを整えています… (1/4)",
        )

    def test_broken_counts_do_not_crash(self) -> None:
        self.assertEqual(
            summary_progress_text(
                {"stage": "part", "current": "x", "total": 5}
            ),
            "要約を作っています…",
        )

    def test_zero_total_shows_label_only(self) -> None:
        self.assertEqual(
            summary_progress_text({"stage": "part", "current": 0, "total": 0}),
            "要約を作っています…",
        )


# ---------------------------------------------------------------------------
# 2. _handle_controller_event(): summary_progress の配線
# ---------------------------------------------------------------------------

def _event_app(folder: Path) -> SimpleNamespace:
    return SimpleNamespace(
        right_pane=SimpleNamespace(
            set_summary_progress=Mock(),
            set_summarizing=Mock(),
            set_thinking=Mock(),
            append_answer=Mock(),
        ),
        main_pane=SimpleNamespace(
            update_status=Mock(),
            set_transcribing_blocked=Mock(),
        ),
        _active_folder=folder,
    )


class SummaryProgressEventTests(unittest.TestCase):
    def test_progress_event_updates_right_pane_and_status(self) -> None:
        folder = Path("C:/lessons/a")
        app = _event_app(folder)
        OtoWeaveApp._handle_controller_event(
            app,
            "summary_progress",
            (folder, {"stage": "part", "current": 2, "total": 5}),
        )
        app.right_pane.set_summary_progress.assert_called_once_with(
            "要約を作っています… (2/5)"
        )
        app.main_pane.update_status.assert_called_once_with(
            "要約を作っています… (2/5)"
        )

    def test_unknown_stage_does_not_crash(self) -> None:
        folder = Path("C:/lessons/a")
        app = _event_app(folder)
        OtoWeaveApp._handle_controller_event(
            app, "summary_progress", (folder, {"stage": "future_stage"})
        )
        app.right_pane.set_summary_progress.assert_called_once_with(
            "要約を処理中…"
        )


# ---------------------------------------------------------------------------
# 3. llm_error / llm_chat_error: 技術的なエラー文を画面に出さない
# ---------------------------------------------------------------------------

class LlmErrorEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        app_logging.setup_logging(Path(self._tmp.name))

    def tearDown(self) -> None:
        app_logging.shutdown_logging()
        self._tmp.cleanup()

    def test_technical_error_is_hidden_and_logged(self) -> None:
        folder = Path("C:/lessons/a")
        app = _event_app(folder)
        payload = (
            "要約スクリプトが失敗しました\n"
            "Traceback (most recent call last):\n"
            "ValueError: llama_decode returned -3"
        )
        OtoWeaveApp._handle_controller_event(app, "llm_error", payload)
        app.main_pane.set_transcribing_blocked.assert_called_once_with(False)
        app.right_pane.set_summarizing.assert_called_once_with(False)
        shown = app.main_pane.update_status.call_args[0][0]
        self.assertNotIn("Traceback", shown)
        self.assertNotIn("ValueError", shown)
        self.assertNotIn("llama", shown)
        self.assertIn("要約", shown)
        log_text = (
            Path(self._tmp.name) / app_logging.LOG_FILE_NAME
        ).read_text(encoding="utf-8")
        self.assertIn("llama_decode returned -3", log_text)

    def test_friendly_timeout_message_is_shown(self) -> None:
        folder = Path("C:/lessons/a")
        app = _event_app(folder)
        payload = (
            "要約がタイムアウトしました（4時間超過）。"
            "文字起こしが長すぎる可能性があります。"
        )
        OtoWeaveApp._handle_controller_event(app, "llm_error", payload)
        shown = app.main_pane.update_status.call_args[0][0]
        self.assertIn("要約がタイムアウトしました", shown)

    def test_chat_error_shows_friendly_bubble(self) -> None:
        folder = Path("C:/lessons/a")
        app = _event_app(folder)
        OtoWeaveApp._handle_controller_event(
            app,
            "llm_chat_error",
            ("RuntimeError: model file corrupt", folder),
        )
        app.right_pane.set_thinking.assert_called_once_with(False)
        shown = app.right_pane.append_answer.call_args[0][0]
        self.assertNotIn("RuntimeError", shown)
        self.assertNotIn("corrupt", shown)
        self.assertIn("もう一度", shown)

    def test_chat_error_for_other_note_shows_no_bubble(self) -> None:
        app = _event_app(Path("C:/lessons/b"))
        OtoWeaveApp._handle_controller_event(
            app,
            "llm_chat_error",
            ("boom", Path("C:/lessons/a")),
        )
        app.right_pane.set_thinking.assert_called_once_with(False)
        app.right_pane.append_answer.assert_not_called()


class SummaryErrorStatusTests(unittest.TestCase):
    def test_timeout_first_line_passes_through(self) -> None:
        message = (
            "要約がタイムアウトしました（45分以上応答がありません）。"
            "パソコンが混み合っているか、処理が止まっている可能性があります。"
        )
        self.assertEqual(summary_error_status(message), f"⚠ {message}")

    def test_technical_stderr_is_replaced(self) -> None:
        text = summary_error_status(
            "要約スクリプトが失敗しました\nOSError: [WinError 8]"
        )
        self.assertNotIn("OSError", text)
        self.assertNotIn("WinError", text)
        self.assertIn("要約", text)

    def test_empty_message_is_replaced(self) -> None:
        text = summary_error_status("")
        self.assertIn("要約", text)


# ---------------------------------------------------------------------------
# 4. build_friendly_error() / _show_friendly_error()
# ---------------------------------------------------------------------------

class FriendlyErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        app_logging.setup_logging(Path(self._tmp.name))

    def tearDown(self) -> None:
        app_logging.shutdown_logging()
        self._tmp.cleanup()

    def _log_text(self) -> str:
        return (
            Path(self._tmp.name) / app_logging.LOG_FILE_NAME
        ).read_text(encoding="utf-8")

    def test_exception_goes_to_log_not_dialog(self) -> None:
        try:
            raise ZeroDivisionError("secret technical detail")
        except ZeroDivisionError as exc:
            body = build_friendly_error(
                "録音",
                "録音を始められませんでした。",
                exc,
            )
        self.assertIn("録音を始められませんでした。", body)
        self.assertIn(FRIENDLY_ERROR_LOG_NOTE, body)
        self.assertNotIn("ZeroDivisionError", body)
        self.assertNotIn("secret technical detail", body)
        log_text = self._log_text()
        self.assertIn("ZeroDivisionError", log_text)
        self.assertIn("secret technical detail", log_text)
        self.assertIn("Traceback", log_text)

    def test_detail_string_goes_to_log_not_dialog(self) -> None:
        body = build_friendly_error(
            "取り込み",
            "ファイル選択の画面を開けませんでした。",
            detail="powershell.exe exited with code 1",
        )
        self.assertNotIn("powershell", body)
        self.assertIn(FRIENDLY_ERROR_LOG_NOTE, body)
        self.assertIn("powershell.exe exited with code 1", self._log_text())

    def test_without_technical_info_no_log_note(self) -> None:
        body = build_friendly_error("要約", "文字起こしがありません。")
        self.assertEqual(body, "文字起こしがありません。")

    def test_show_friendly_error_uses_messagebox_with_clean_body(self) -> None:
        app = SimpleNamespace()
        exc = RuntimeError("CUDA out of memory")
        with patch(
            "otoweave_app.otoweave_app.messagebox.showerror"
        ) as showerror:
            OtoWeaveApp._show_friendly_error(
                app,
                "要約",
                "要約を作れませんでした。",
                exc,
            )
        self.assertEqual(showerror.call_count, 1)
        args, kwargs = showerror.call_args
        self.assertEqual(args[0], "要約")
        self.assertIn("要約を作れませんでした。", args[1])
        self.assertNotIn("CUDA", args[1])
        self.assertIs(kwargs.get("parent"), app)


# ---------------------------------------------------------------------------
# 5. processing_indicator_state(): 中止ボタンの状態遷移
# ---------------------------------------------------------------------------

class ProcessingIndicatorStateTests(unittest.TestCase):
    def test_idle_is_hidden(self) -> None:
        state = processing_indicator_state(False, False, False)
        self.assertFalse(state["visible"])

    def test_transcribing_shows_cancel_button(self) -> None:
        state = processing_indicator_state(True, False, False)
        self.assertTrue(state["visible"])
        self.assertEqual(state["button_text"], "中止")
        self.assertEqual(state["button_state"], "normal")

    def test_importing_shows_cancel_button(self) -> None:
        state = processing_indicator_state(False, True, False)
        self.assertTrue(state["visible"])
        self.assertEqual(state["button_text"], "中止")

    def test_cancel_pending_disables_button(self) -> None:
        state = processing_indicator_state(True, False, True)
        self.assertTrue(state["visible"])
        self.assertEqual(state["button_text"], "中止中…")
        self.assertEqual(state["button_state"], "disabled")

    def test_finished_hides_again(self) -> None:
        # 中止直後（cancel_pending のまま）でも処理が終われば非表示。
        state = processing_indicator_state(False, False, True)
        self.assertFalse(state["visible"])


# ---------------------------------------------------------------------------
# 6. _on_cancel_processing(): controller への配線
# ---------------------------------------------------------------------------

class CancelProcessingWiringTests(unittest.TestCase):
    @staticmethod
    def _app(cancel_result: bool) -> SimpleNamespace:
        return SimpleNamespace(
            controller=SimpleNamespace(
                cancel_transcription=Mock(return_value=cancel_result)
            ),
            main_pane=SimpleNamespace(
                set_cancel_processing_pending=Mock(),
                update_status=Mock(),
            ),
        )

    def test_cancel_accepted_marks_button_pending(self) -> None:
        app = self._app(cancel_result=True)
        OtoWeaveApp._on_cancel_processing(app)
        app.controller.cancel_transcription.assert_called_once_with()
        app.main_pane.set_cancel_processing_pending.assert_called_once_with()
        shown = app.main_pane.update_status.call_args[0][0]
        self.assertIn("中止", shown)

    def test_cancel_rejected_keeps_button_normal(self) -> None:
        # IMPORTING / TRANSCRIBING 以外では False が返り、表示は変えない。
        app = self._app(cancel_result=False)
        OtoWeaveApp._on_cancel_processing(app)
        app.main_pane.set_cancel_processing_pending.assert_not_called()
        app.main_pane.update_status.assert_not_called()

    def test_without_controller_is_noop(self) -> None:
        app = SimpleNamespace(controller=None)
        OtoWeaveApp._on_cancel_processing(app)  # 例外を出さない


# ---------------------------------------------------------------------------
# 7. close_wait_status(): 終了待ちの文言
# ---------------------------------------------------------------------------

class CloseWaitStatusTests(unittest.TestCase):
    def test_initial_wait_message(self) -> None:
        self.assertEqual(
            close_wait_status(0.0),
            "保存中です。しばらくお待ちください…",
        )

    def test_just_below_threshold_keeps_initial_message(self) -> None:
        self.assertEqual(
            close_wait_status(CLOSE_WAIT_NOTICE_SECONDS - 0.1),
            "保存中です。しばらくお待ちください…",
        )

    def test_long_wait_switches_message(self) -> None:
        self.assertEqual(
            close_wait_status(CLOSE_WAIT_NOTICE_SECONDS),
            "保存を続けています。もう少しお待ちください…",
        )
        self.assertEqual(
            close_wait_status(120.0),
            "保存を続けています。もう少しお待ちください…",
        )


# ---------------------------------------------------------------------------
# 8. audio_level_text(): 生の測定値を見せない
# ---------------------------------------------------------------------------

class AudioLevelTextTests(unittest.TestCase):
    def test_good_result_is_plain_japanese(self) -> None:
        text = audio_level_text({"state": "Good", "rms": 0.05, "peak": 0.4})
        self.assertIn("良好", text)
        self.assertNotIn("RMS", text)
        self.assertNotIn("Peak", text)
        self.assertNotIn("0.05", text)

    def test_caution_result_gives_next_action(self) -> None:
        text = audio_level_text(
            {"state": "Caution", "rms": 0.008, "peak": 0.2}
        )
        self.assertIn("マイク", text)
        self.assertNotIn("0.008", text)

    def test_poor_result_gives_next_action(self) -> None:
        text = audio_level_text({"state": "Poor", "rms": 0.001, "peak": 1.0})
        self.assertIn("マイク", text)
        self.assertNotIn("RMS", text)

    def test_unknown_state_has_fallback(self) -> None:
        text = audio_level_text({"state": "Odd", "rms": 0.5})
        self.assertNotIn("Odd", text)
        self.assertTrue(text)

    def test_none_result_has_fallback(self) -> None:
        self.assertTrue(audio_level_text(None))


if __name__ == "__main__":
    unittest.main()
