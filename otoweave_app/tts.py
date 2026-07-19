"""Offline text-to-speech via the Windows built-in synthesizer.

Reading difficulty is the whole reason OtoWeave exists, so summaries and
transcripts can be listened to instead of read. System.Speech (the stock
Windows Japanese voice, e.g. Haruka) is used in a short-lived PowerShell
subprocess:

- zero model download, zero extra dependency — works on stock GIGA machines
- fully local, nothing is sent anywhere
- killing the subprocess stops the speech immediately
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from .app_logging import log_exception


_SPEAK_SCRIPT = (
    "$ErrorActionPreference='Stop';"
    "Add-Type -AssemblyName System.Speech;"
    "$voice=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "try {"
    "$voice.SelectVoiceByHints("
    "[System.Speech.Synthesis.VoiceGender]::NotSet,"
    "[System.Speech.Synthesis.VoiceAge]::NotSet,"
    "0,"
    "[System.Globalization.CultureInfo]::GetCultureInfo('ja-JP'))"
    "} catch { };"
    "$voice.Rate=[int]$env:OTOWEAVE_TTS_RATE;"
    "$text=[System.IO.File]::ReadAllText($env:OTOWEAVE_TTS_FILE,"
    "[System.Text.Encoding]::UTF8);"
    "$voice.Speak($text);"
    "$voice.Dispose();"
)

TTS_TEMP_PREFIX = "otoweave_tts_"


def tts_temp_dir(data_root: Path | str) -> Path:
    """読み上げ用一時ファイルの置き場所（データルート配下）。

    共有 %TEMP% に文字起こし本文（生徒の発話を含む）を置くと、強制終了時に
    他の利用者から読める平文が残るため、データルート配下に隔離する。"""
    return Path(data_root) / ".tmp"


def cleanup_stale_tts_files(temp_dir: Path | str) -> int:
    """前回の強制終了などで残った読み上げ用一時ファイルを削除する。

    削除できた件数を返す。フォルダが無い・読めない場合は 0。"""
    removed = 0
    try:
        entries = list(Path(temp_dir).glob(TTS_TEMP_PREFIX + "*"))
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file():
                entry.unlink()
                removed += 1
        except OSError:
            pass
    return removed


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
_BULLET_RE = re.compile(r"^\s*[-*・●○]\s*")
_TIMESTAMP_RE = re.compile(r"^\s*\d{1,3}:\d{2}(?:\s*-\s*\d{1,3}:\d{2})?\s*")
_BRACKET_TAG_RE = re.compile(r"\[[^\]]{1,40}\]")


def readable_text(value: str) -> str:
    """Strip markdown/timestamp noise so the voice reads only the content.

    「## 今日のテーマ」 should be spoken as 「今日のテーマ」, and transcript
    lines should not start with 「ぜろごじゅうご」 for 00:55."""
    lines: list[str] = []
    for line in str(value).splitlines():
        line = _HEADING_RE.sub("", line)
        line = _BULLET_RE.sub("", line)
        line = _TIMESTAMP_RE.sub("", line)
        line = _BRACKET_TAG_RE.sub("", line)
        line = line.replace("⚠", "注意。").strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


class WindowsTts:
    """Speak one text at a time; a new speak() cancels the previous one."""

    MAX_CHARS = 20000

    def __init__(
        self,
        on_finished: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        rate: int = 0,
        temp_dir: Path | str | None = None,
    ) -> None:
        self._on_finished = on_finished
        self._on_error = on_error
        self.rate = rate
        # 読み上げテキストの一時ファイル置き場。None なら従来どおり %TEMP%。
        # 保存先変更に追従できるよう、後から差し替え可能な公開属性にする。
        self.temp_dir: Path | None = Path(temp_dir) if temp_dir else None
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._generation = 0

    def _resolve_temp_dir(self) -> str | None:
        """一時ファイルの作成先を返す。作れない場合は %TEMP% へフォールバック。"""
        directory = self.temp_dir
        if directory is None:
            return None
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return str(directory)

    @property
    def speaking(self) -> bool:
        with self._lock:
            process = self._process
        return process is not None and process.poll() is None

    def speak(self, text: str) -> bool:
        cleaned = readable_text(text).strip()
        if not cleaned:
            return False
        if len(cleaned) > self.MAX_CHARS:
            cleaned = cleaned[: self.MAX_CHARS]
        self.stop()

        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=TTS_TEMP_PREFIX,
            suffix=".txt",
            delete=False,
            dir=self._resolve_temp_dir(),
        )
        try:
            with handle:
                handle.write(cleaned)
            text_file = Path(handle.name)
            environment = dict(os.environ)
            environment["OTOWEAVE_TTS_FILE"] = str(text_file)
            environment["OTOWEAVE_TTS_RATE"] = str(int(self.rate))
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    _SPEAK_SCRIPT,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=environment,
                creationflags=creationflags,
            )
        except Exception as exc:
            log_exception("読み上げの開始に失敗", exc)
            Path(handle.name).unlink(missing_ok=True)
            if self._on_error is not None:
                self._on_error("読み上げを開始できませんでした。もう一度お試しください。")
            return False
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._process = process
        threading.Thread(
            target=self._watch,
            args=(process, text_file, generation),
            name="tts-watcher",
            daemon=True,
        ).start()
        return True

    def _watch(
        self,
        process: subprocess.Popen,
        text_file: Path,
        generation: int,
    ) -> None:
        try:
            _stdout, stderr = process.communicate()
        except Exception:
            stderr = b""
        text_file.unlink(missing_ok=True)
        with self._lock:
            is_current = generation == self._generation
            if is_current:
                self._process = None
        if not is_current:
            # A newer speak() superseded this one; its watcher will report.
            return
        if process.returncode not in (0, None) and stderr and self._on_error is not None:
            message = stderr.decode("utf-8", errors="replace").strip()
            if message and "OperationStopped" not in message:
                self._on_error("読み上げに失敗しました（音声合成を利用できない可能性があります）。")
        if self._on_finished is not None:
            self._on_finished()

    def stop(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def close(self) -> None:
        self.stop()


class MacTts:
    """Offline text-to-speech via the macOS built-in `say` command.

    Mirrors WindowsTts: one utterance at a time, a new speak() cancels the
    previous one, killing the subprocess stops speech immediately, and the
    text is written to a temp file under the data root (not the shared
    system temp dir) for the same privacy reason as WindowsTts. `say` ships
    with every Mac, needs no model download, and stays fully local. The
    Japanese voice (Kyoko) is used when installed, matching the ja-JP voice
    OtoWeave uses on Windows; otherwise the system default is used.
    """

    MAX_CHARS = 20000
    DEFAULT_VOICE = "Kyoko"

    def __init__(
        self,
        on_finished: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        rate: int = 0,
        temp_dir: Path | str | None = None,
    ) -> None:
        self._on_finished = on_finished
        self._on_error = on_error
        self.rate = rate
        self.temp_dir: Path | None = Path(temp_dir) if temp_dir else None
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._generation = 0
        self._voice = self.DEFAULT_VOICE if self._voice_available(self.DEFAULT_VOICE) else None

    def _resolve_temp_dir(self) -> str | None:
        """一時ファイルの作成先を返す。作れない場合は %TEMP% へフォールバック。"""
        directory = self.temp_dir
        if directory is None:
            return None
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return str(directory)

    @staticmethod
    def _voice_available(voice: str) -> bool:
        try:
            result = subprocess.run(
                ["say", "-v", "?"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return False
        if result.returncode != 0:
            return False
        return any(line.split(maxsplit=1)[:1] == [voice] for line in result.stdout.splitlines())

    @property
    def speaking(self) -> bool:
        with self._lock:
            process = self._process
        return process is not None and process.poll() is None

    def _words_per_minute(self) -> int:
        # Windows SAPI rate is roughly -10..10 around a neutral 0; map that
        # onto `say`'s words-per-minute so the speed slider behaves the same.
        return max(90, min(360, 180 + int(self.rate) * 12))

    def speak(self, text: str) -> bool:
        cleaned = readable_text(text).strip()
        if not cleaned:
            return False
        if len(cleaned) > self.MAX_CHARS:
            cleaned = cleaned[: self.MAX_CHARS]
        self.stop()

        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=TTS_TEMP_PREFIX,
            suffix=".txt",
            delete=False,
            dir=self._resolve_temp_dir(),
        )
        try:
            with handle:
                handle.write(cleaned)
            text_file = Path(handle.name)
            command = ["say", "-r", str(self._words_per_minute())]
            if self._voice:
                command += ["-v", self._voice]
            command += ["-f", str(text_file)]
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            log_exception("読み上げの開始に失敗", exc)
            Path(handle.name).unlink(missing_ok=True)
            if self._on_error is not None:
                self._on_error("読み上げを開始できませんでした。もう一度お試しください。")
            return False
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._process = process
        threading.Thread(
            target=self._watch,
            args=(process, text_file, generation),
            name="tts-watcher",
            daemon=True,
        ).start()
        return True

    def _watch(
        self,
        process: subprocess.Popen,
        text_file: Path,
        generation: int,
    ) -> None:
        try:
            _stdout, stderr = process.communicate()
        except Exception:
            stderr = b""
        text_file.unlink(missing_ok=True)
        with self._lock:
            is_current = generation == self._generation
            if is_current:
                self._process = None
        if not is_current:
            # A newer speak() superseded this one; its watcher will report.
            return
        # `say` returns non-zero when killed for a newer utterance; a real
        # failure (voice missing etc.) also lands here, so only report when
        # this utterance was the current one and left an error message.
        if process.returncode not in (0, None, -9, -15) and self._on_error is not None:
            message = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
            self._on_error(
                message or "読み上げに失敗しました（音声合成を利用できない可能性があります）。"
            )
        if self._on_finished is not None:
            self._on_finished()

    def stop(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def close(self) -> None:
        self.stop()


class NullTts:
    """No-op TTS for platforms without a bundled synthesizer (e.g. Linux).

    Also the harmless fallback when no offline synthesizer is available at
    all, so the rest of the app never has to special-case a missing TTS."""

    def __init__(
        self,
        on_finished: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        rate: int = 0,
        temp_dir: Path | str | None = None,
    ) -> None:
        self._on_error = on_error
        self.rate = rate
        self.temp_dir: Path | None = Path(temp_dir) if temp_dir else None

    speaking = False

    def speak(self, text: str) -> bool:
        if self._on_error is not None:
            self._on_error("この環境では読み上げを利用できません。")
        return False

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def create_tts(
    on_finished: Callable[[], None] | None = None,
    on_error: Callable[[str], None] | None = None,
    rate: int = 0,
    temp_dir: Path | str | None = None,
):
    """Return the platform's offline TTS backend (WindowsTts / MacTts /
    NullTts), so callers do not need their own IS_WINDOWS/IS_MACOS branch."""
    from .platform_support import IS_MACOS, IS_WINDOWS

    if IS_WINDOWS:
        return WindowsTts(on_finished=on_finished, on_error=on_error, rate=rate, temp_dir=temp_dir)
    if IS_MACOS:
        return MacTts(on_finished=on_finished, on_error=on_error, rate=rate, temp_dir=temp_dir)
    return NullTts(on_finished=on_finished, on_error=on_error, rate=rate, temp_dir=temp_dir)
