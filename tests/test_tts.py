import subprocess
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from otoweave_app.customtkinter_views import MainPane
from otoweave_app.otoweave_app import OtoWeaveApp
from otoweave_app.tts import WindowsTts, readable_text


class ReadableTextTests(unittest.TestCase):
    def test_markdown_headings_and_bullets_are_stripped(self) -> None:
        value = "## 今日のテーマ\n- 分数の計算\n* 約分\n・記号付き"
        result = readable_text(value)
        self.assertEqual(result, "今日のテーマ\n分数の計算\n約分\n記号付き")

    def test_leading_timestamps_are_stripped(self) -> None:
        value = "00:05  こんにちは\n75:30  長い授業の後半\n[01:00-01:10] 抜粋"
        result = readable_text(value)
        self.assertNotIn("00:05", result)
        self.assertNotIn("75:30", result)
        self.assertIn("こんにちは", result)
        self.assertIn("長い授業の後半", result)

    def test_warning_mark_is_spoken_as_word(self) -> None:
        self.assertIn("注意。", readable_text("⚠ 一部が欠けている可能性"))

    def test_empty_result_for_noise_only(self) -> None:
        self.assertEqual(readable_text("## \n- \n00:00  "), "")


class _FakeTtsProcess:
    def __init__(self) -> None:
        self._done = threading.Event()
        self.killed = False
        self.returncode: int | None = None

    def communicate(self):
        self._done.wait(timeout=10)
        if self.returncode is None:
            self.returncode = 0
        return b"", b""

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = 1
        self._done.set()

    def finish(self) -> None:
        self.returncode = 0
        self._done.set()


class WindowsTtsTests(unittest.TestCase):
    def test_empty_text_is_rejected_without_process(self) -> None:
        with patch("otoweave_app.tts.subprocess.Popen") as popen:
            tts = WindowsTts()
            self.assertFalse(tts.speak("## \n- "))
            popen.assert_not_called()

    def test_finish_fires_on_finished_once(self) -> None:
        finished = threading.Event()
        process = _FakeTtsProcess()
        with patch(
            "otoweave_app.tts.subprocess.Popen",
            return_value=process,
        ):
            tts = WindowsTts(on_finished=finished.set)
            self.assertTrue(tts.speak("読み上げテスト"))
            self.assertTrue(tts.speaking)
            process.finish()
            self.assertTrue(finished.wait(timeout=5))
        deadline = time.time() + 5
        while tts.speaking and time.time() < deadline:
            time.sleep(0.02)
        self.assertFalse(tts.speaking)

    def test_stop_kills_the_speaking_process(self) -> None:
        finished = threading.Event()
        process = _FakeTtsProcess()
        with patch(
            "otoweave_app.tts.subprocess.Popen",
            return_value=process,
        ):
            tts = WindowsTts(on_finished=finished.set)
            tts.speak("停止テスト")
            tts.stop()
        self.assertTrue(process.killed)
        self.assertTrue(finished.wait(timeout=5))

    def test_japanese_voice_is_installed_on_this_machine(self) -> None:
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
        self.assertIn("ja-JP", result.stdout)


class SpeakToggleTests(unittest.TestCase):
    @staticmethod
    def _app(speaking: bool, target: str) -> SimpleNamespace:
        app = SimpleNamespace(
            _tts=SimpleNamespace(
                speak=Mock(return_value=True),
                stop=Mock(),
                speaking=speaking,
            ),
            _tts_target=target,
            controller=None,
            main_pane=SimpleNamespace(
                set_speaking_transcript=Mock(),
                update_status=Mock(),
            ),
            right_pane=SimpleNamespace(set_speaking_summary=Mock()),
        )
        return app

    def test_pressing_again_stops_current_speech(self) -> None:
        app = self._app(speaking=True, target="summary")
        OtoWeaveApp._speak(app, "summary", "要約テキスト")
        app._tts.stop.assert_called_once()
        app._tts.speak.assert_not_called()

    def test_new_target_starts_and_updates_buttons(self) -> None:
        app = self._app(speaking=False, target="")
        OtoWeaveApp._speak(app, "transcript", "本文テキスト")
        app._tts.speak.assert_called_once_with("本文テキスト")
        self.assertEqual(app._tts_target, "transcript")
        app.main_pane.set_speaking_transcript.assert_called_once_with(True)
        app.right_pane.set_speaking_summary.assert_called_once_with(False)


class TimestampClickTests(unittest.TestCase):
    def test_leading_timestamp_is_parsed(self) -> None:
        self.assertEqual(MainPane._parse_leading_timestamp("00:05  文章"), 5.0)
        self.assertEqual(MainPane._parse_leading_timestamp("75:30  後半"), 4530.0)
        self.assertIsNone(MainPane._parse_leading_timestamp("文章のみ"))
        self.assertIsNone(MainPane._parse_leading_timestamp(""))


if __name__ == "__main__":
    unittest.main()
