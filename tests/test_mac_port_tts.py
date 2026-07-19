"""Mac (Apple Silicon) port: MacTts / NullTts / create_tts() factory.

The dev machine is Windows, so MacTts is exercised directly (it has no
platform guard of its own -- the guard lives in create_tts()) with
subprocess.Popen and the `say -v ?` voice probe mocked out.
"""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from otoweave_app.tts import (
    NullTts,
    TTS_TEMP_PREFIX,
    WindowsTts,
    create_tts,
)
from otoweave_app import tts as tts_module


class _FakeSayProcess:
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
        self.returncode = -9
        self._done.set()

    def finish(self) -> None:
        self.returncode = 0
        self._done.set()


def _voice_probe_result(voices: list[str]):
    stdout = "\n".join(f"{name}    ja_JP    # comment" for name in voices)
    return type("Result", (), {"returncode": 0, "stdout": stdout})()


class MacTtsCommandConstructionTests(unittest.TestCase):
    """MacTts must shell out to `say` the same way WindowsTts shells out to
    PowerShell: one utterance at a time, rate mapped to the CLI, Kyoko
    selected when installed."""

    def test_kyoko_voice_is_selected_when_installed(self) -> None:
        with patch(
            "otoweave_app.tts.subprocess.run",
            return_value=_voice_probe_result(["Kyoko", "Alex"]),
        ):
            tts = tts_module.MacTts()
        self.assertEqual(tts._voice, "Kyoko")

    def test_falls_back_to_default_voice_when_kyoko_is_missing(self) -> None:
        with patch(
            "otoweave_app.tts.subprocess.run",
            return_value=_voice_probe_result(["Alex", "Samantha"]),
        ):
            tts = tts_module.MacTts()
        self.assertIsNone(tts._voice)

    def test_voice_probe_failure_is_treated_as_unavailable(self) -> None:
        with patch(
            "otoweave_app.tts.subprocess.run",
            side_effect=OSError("say not found"),
        ):
            tts = tts_module.MacTts()
        self.assertIsNone(tts._voice)

    def test_speak_builds_say_command_with_voice_rate_and_file(self) -> None:
        with patch(
            "otoweave_app.tts.subprocess.run",
            return_value=_voice_probe_result(["Kyoko"]),
        ):
            tts = tts_module.MacTts(rate=0)
        process = _FakeSayProcess()
        with patch(
            "otoweave_app.tts.subprocess.Popen", return_value=process
        ) as popen:
            self.assertTrue(tts.speak("こんにちは"))
            process.finish()
            command = popen.call_args.args[0]
        self.assertEqual(command[0], "say")
        self.assertIn("-r", command)
        self.assertIn("-v", command)
        self.assertEqual(command[command.index("-v") + 1], "Kyoko")
        self.assertIn("-f", command)
        text_file = Path(command[command.index("-f") + 1])
        self.assertTrue(text_file.name.startswith(TTS_TEMP_PREFIX))

    def test_speak_omits_voice_flag_when_kyoko_unavailable(self) -> None:
        with patch(
            "otoweave_app.tts.subprocess.run",
            return_value=_voice_probe_result(["Alex"]),
        ):
            tts = tts_module.MacTts()
        process = _FakeSayProcess()
        with patch(
            "otoweave_app.tts.subprocess.Popen", return_value=process
        ) as popen:
            self.assertTrue(tts.speak("hello"))
            process.finish()
            command = popen.call_args.args[0]
        self.assertNotIn("-v", command)

    def test_rate_maps_to_words_per_minute_within_say_bounds(self) -> None:
        with patch(
            "otoweave_app.tts.subprocess.run",
            return_value=_voice_probe_result([]),
        ):
            neutral = tts_module.MacTts(rate=0)
            fast = tts_module.MacTts(rate=10)
            slow = tts_module.MacTts(rate=-10)
        self.assertEqual(neutral._words_per_minute(), 180)
        self.assertEqual(fast._words_per_minute(), 300)
        self.assertEqual(slow._words_per_minute(), 90)  # clamped to the minimum

    def test_empty_text_is_rejected_without_process(self) -> None:
        with patch(
            "otoweave_app.tts.subprocess.run",
            return_value=_voice_probe_result([]),
        ):
            tts = tts_module.MacTts()
        with patch("otoweave_app.tts.subprocess.Popen") as popen:
            self.assertFalse(tts.speak("## \n- "))
            popen.assert_not_called()

    def test_finish_fires_on_finished_once(self) -> None:
        finished = threading.Event()
        process = _FakeSayProcess()
        with patch(
            "otoweave_app.tts.subprocess.run",
            return_value=_voice_probe_result([]),
        ):
            tts = tts_module.MacTts(on_finished=finished.set)
        with patch(
            "otoweave_app.tts.subprocess.Popen", return_value=process
        ):
            self.assertTrue(tts.speak("読み上げテスト"))
            self.assertTrue(tts.speaking)
            process.finish()
            self.assertTrue(finished.wait(timeout=5))

    def test_stop_kills_the_speaking_process(self) -> None:
        finished = threading.Event()
        process = _FakeSayProcess()
        with patch(
            "otoweave_app.tts.subprocess.run",
            return_value=_voice_probe_result([]),
        ):
            tts = tts_module.MacTts(on_finished=finished.set)
        with patch(
            "otoweave_app.tts.subprocess.Popen", return_value=process
        ):
            tts.speak("停止テスト")
            tts.stop()
        self.assertTrue(process.killed)
        self.assertTrue(finished.wait(timeout=5))

    def test_killed_process_does_not_report_error(self) -> None:
        # say returns non-zero (-9/-15) when killed for a newer utterance or
        # by stop(); that must not surface as a spurious error message.
        errors: list[str] = []
        process = _FakeSayProcess()
        with patch(
            "otoweave_app.tts.subprocess.run",
            return_value=_voice_probe_result([]),
        ):
            tts = tts_module.MacTts(on_error=errors.append)
        with patch(
            "otoweave_app.tts.subprocess.Popen", return_value=process
        ):
            tts.speak("停止テスト")
            tts.stop()
        deadline = threading.Event()
        deadline.wait(timeout=0.2)
        self.assertEqual(errors, [])

    def test_temp_file_is_created_inside_given_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "data" / ".tmp"
            process = _FakeSayProcess()
            with patch(
                "otoweave_app.tts.subprocess.run",
                return_value=_voice_probe_result([]),
            ):
                tts = tts_module.MacTts(temp_dir=target)
            with patch(
                "otoweave_app.tts.subprocess.Popen", return_value=process
            ) as popen:
                self.assertTrue(tts.speak("プライバシーテスト"))
                command = popen.call_args.args[0]
                text_file = Path(command[command.index("-f") + 1])
                self.assertEqual(text_file.parent, target)
                process.finish()


class NullTtsTests(unittest.TestCase):
    """Linux (and any platform without a bundled synthesizer) gets a
    harmless no-op backend instead of a crash."""

    def test_speak_reports_error_and_returns_false(self) -> None:
        errors: list[str] = []
        tts = NullTts(on_error=errors.append)
        self.assertFalse(tts.speak("何か"))
        self.assertEqual(len(errors), 1)

    def test_speaking_is_always_false(self) -> None:
        tts = NullTts()
        self.assertFalse(tts.speaking)

    def test_stop_and_close_are_harmless(self) -> None:
        tts = NullTts()
        tts.stop()
        tts.close()


class CreateTtsFactoryTests(unittest.TestCase):
    """create_tts() is the single seam the rest of the app should use
    instead of importing WindowsTts/MacTts/NullTts directly."""

    def test_windows_gets_windows_tts(self) -> None:
        with patch("otoweave_app.platform_support.IS_WINDOWS", True), patch(
            "otoweave_app.platform_support.IS_MACOS", False
        ):
            tts = create_tts()
        self.assertIsInstance(tts, WindowsTts)

    def test_macos_gets_mac_tts(self) -> None:
        with patch("otoweave_app.platform_support.IS_WINDOWS", False), patch(
            "otoweave_app.platform_support.IS_MACOS", True
        ), patch(
            "otoweave_app.tts.subprocess.run",
            return_value=_voice_probe_result([]),
        ):
            tts = create_tts()
        self.assertIsInstance(tts, tts_module.MacTts)

    def test_linux_gets_null_tts(self) -> None:
        with patch("otoweave_app.platform_support.IS_WINDOWS", False), patch(
            "otoweave_app.platform_support.IS_MACOS", False
        ):
            tts = create_tts()
        self.assertIsInstance(tts, NullTts)

    def test_temp_dir_is_forwarded_to_the_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / ".tmp"
            with patch("otoweave_app.platform_support.IS_WINDOWS", True), patch(
                "otoweave_app.platform_support.IS_MACOS", False
            ):
                tts = create_tts(temp_dir=target)
            self.assertEqual(tts.temp_dir, target)


if __name__ == "__main__":
    unittest.main()
