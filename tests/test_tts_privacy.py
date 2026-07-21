"""読み上げ（TTS）一時ファイルのプライバシー対策のテスト。

読み上げテキストには文字起こし本文（生徒の発話を含む、最大2万字）が
入るため、共有 %TEMP% ではなくデータルート配下に置き、前回の強制終了で
残ったファイルは起動時に削除する。

Covers:
  - tts_temp_dir(): データルート配下の一時フォルダの解決
  - WindowsTts(temp_dir=...): 指定フォルダへの一時ファイル作成と後始末
  - temp_dir 未指定・作成不能時の %TEMP% フォールバック（従来互換）
  - cleanup_stale_tts_files(): 残存ファイルの削除
"""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from otoweave_app.tts import (
    TTS_TEMP_PREFIX,
    WindowsTts,
    cleanup_stale_tts_files,
    tts_temp_dir,
)


class _FakeTtsProcess:
    def __init__(self) -> None:
        self._done = threading.Event()
        self.returncode: int | None = None

    def communicate(self):
        self._done.wait(timeout=10)
        if self.returncode is None:
            self.returncode = 0
        return b"", b""

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.returncode = 1
        self._done.set()

    def finish(self) -> None:
        self.returncode = 0
        self._done.set()


class TtsTempDirTests(unittest.TestCase):
    def test_temp_dir_is_under_data_root(self) -> None:
        root = Path(r"C:\Users\student\Documents\LearningAccess")
        self.assertEqual(tts_temp_dir(root), root / ".tmp")

    def test_accepts_string_root(self) -> None:
        self.assertEqual(
            tts_temp_dir(r"D:\OtoWeaveData"),
            Path(r"D:\OtoWeaveData") / ".tmp",
        )


class WindowsTtsTempDirTests(unittest.TestCase):
    @staticmethod
    def _speak_and_capture(tts: WindowsTts, popen) -> tuple[_FakeTtsProcess, Path]:
        """speak() を実行し、PowerShell へ渡された一時ファイルのパスを返す。"""
        process = popen.return_value
        assert tts.speak("プライバシーテスト本文")
        env = popen.call_args.kwargs["env"]
        return process, Path(env["OTOWEAVE_TTS_FILE"])

    def test_temp_file_is_created_inside_given_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "data" / ".tmp"
            finished = threading.Event()
            with patch(
                "otoweave_app.tts.subprocess.Popen",
                return_value=_FakeTtsProcess(),
            ) as popen:
                tts = WindowsTts(on_finished=finished.set, temp_dir=target)
                process, text_file = self._speak_and_capture(tts, popen)
                # 指定フォルダ（無ければ作成される）の中に平文を置くこと。
                self.assertEqual(text_file.parent, target)
                self.assertTrue(text_file.name.startswith(TTS_TEMP_PREFIX))
                self.assertIn(
                    "プライバシーテスト本文",
                    text_file.read_text(encoding="utf-8"),
                )
                # 読み上げ完了後は削除されること（既存の後始末の維持）。
                process.finish()
                self.assertTrue(finished.wait(timeout=5))
                self.assertFalse(text_file.exists())

    def test_default_falls_back_to_system_temp(self) -> None:
        finished = threading.Event()
        with patch(
            "otoweave_app.tts.subprocess.Popen",
            return_value=_FakeTtsProcess(),
        ) as popen:
            tts = WindowsTts(on_finished=finished.set)
            process, text_file = self._speak_and_capture(tts, popen)
            self.assertEqual(
                text_file.parent,
                Path(tempfile.gettempdir()),
            )
            process.finish()
            self.assertTrue(finished.wait(timeout=5))
            self.assertFalse(text_file.exists())

    def test_unusable_temp_dir_falls_back_to_system_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            # フォルダを作れない場所（既存ファイルと同名）を指定する。
            blocker = Path(temporary) / "blocker"
            blocker.write_text("x", encoding="utf-8")
            finished = threading.Event()
            with patch(
                "otoweave_app.tts.subprocess.Popen",
                return_value=_FakeTtsProcess(),
            ) as popen:
                tts = WindowsTts(
                    on_finished=finished.set,
                    temp_dir=blocker / ".tmp",
                )
                process, text_file = self._speak_and_capture(tts, popen)
                self.assertEqual(
                    text_file.parent,
                    Path(tempfile.gettempdir()),
                )
                process.finish()
                self.assertTrue(finished.wait(timeout=5))
                text_file.unlink(missing_ok=True)

    def test_temp_dir_can_be_switched_later(self) -> None:
        # 保存先変更に追従して一時フォルダを差し替えられること。
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first" / ".tmp"
            second = Path(temporary) / "second" / ".tmp"
            finished = threading.Event()
            with patch(
                "otoweave_app.tts.subprocess.Popen",
                return_value=_FakeTtsProcess(),
            ) as popen:
                tts = WindowsTts(on_finished=finished.set, temp_dir=first)
                tts.temp_dir = second
                process, text_file = self._speak_and_capture(tts, popen)
                self.assertEqual(text_file.parent, second)
                process.finish()
                self.assertTrue(finished.wait(timeout=5))


class CleanupStaleTtsFilesTests(unittest.TestCase):
    def test_removes_only_tts_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            stale_a = folder / f"{TTS_TEMP_PREFIX}abc.txt"
            stale_b = folder / f"{TTS_TEMP_PREFIX}def.txt"
            other = folder / "keep_me.txt"
            for path in (stale_a, stale_b, other):
                path.write_text("残存データ", encoding="utf-8")
            removed = cleanup_stale_tts_files(folder)
            self.assertEqual(removed, 2)
            self.assertFalse(stale_a.exists())
            self.assertFalse(stale_b.exists())
            self.assertTrue(other.exists())

    def test_missing_directory_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "no_such_dir"
            self.assertEqual(cleanup_stale_tts_files(missing), 0)

    def test_subdirectories_are_not_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            subdir = folder / f"{TTS_TEMP_PREFIX}dir"
            subdir.mkdir()
            self.assertEqual(cleanup_stale_tts_files(folder), 0)
            self.assertTrue(subdir.exists())


if __name__ == "__main__":
    unittest.main()
