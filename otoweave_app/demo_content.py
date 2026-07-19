"""Demo-lesson creation and repair, separated from the controller."""
from __future__ import annotations

import os
import subprocess
import wave
from datetime import datetime
from pathlib import Path

from .models import LessonRecord, TranscriptSegment
from .storage import LessonStore


class DemoContent:
    def __init__(self, store: LessonStore) -> None:
        self.store = store

    def create_demo_lesson(self) -> None:
        if self.store._lesson_folders():
            self.repair_demo_audio()
            return
        now = datetime.now().astimezone()
        lesson = LessonRecord.create("english", "microphone", now=now)
        lesson.title = "English S&E"
        lesson.is_demo = True
        lesson.status = "complete"
        lesson.segments = [
            TranscriptSegment(
                id="seg_0001",
                start=0.0,
                end=5.0,
                text="Brain drain means skilled workers leave their home country.",
                important=True,
            ),
            TranscriptSegment(
                id="seg_0002",
                start=5.0,
                end=10.0,
                text="This can cause shortages of doctors, nurses, and teachers.",
                unclear=True,
            ),
            TranscriptSegment(
                id="seg_0003",
                start=10.0,
                end=15.0,
                text="Think about one possible solution with your partner.",
                question=True,
            ),
        ]
        folder = self.store.create_lesson(lesson)
        self._attach_demo_audio(folder, lesson)

    def repair_demo_audio(
        self,
        lessons: list[tuple[Path, LessonRecord]] | None = None,
    ) -> None:
        if lessons is not None:
            for folder, lesson in lessons:
                self._repair_one(folder, lesson)
            return
        # Metadata-only pre-filter: only demo candidates are fully loaded,
        # so a large lesson store no longer slows down every app start.
        for folder, metadata in self.store.list_lesson_metadata():
            if not metadata.get("is_demo") and str(metadata.get("title", "")) != "English S&E":
                continue
            try:
                lesson = self.store.load(folder)
            except Exception:
                continue
            self._repair_one(folder, lesson)

    def _repair_one(self, folder: Path, lesson: LessonRecord) -> None:
        is_legacy_demo = (
            lesson.title == "English S&E"
            and bool(lesson.segments)
            and lesson.segments[0].text.startswith("Brain drain means skilled")
        )
        if is_legacy_demo and not lesson.is_demo:
            lesson.is_demo = True
            self.store.save(folder, lesson)
        if lesson.is_demo and self._lesson_audio_path(folder, lesson) is None:
            self._attach_demo_audio(folder, lesson)

    @staticmethod
    def _lesson_audio_path(folder: Path, lesson: LessonRecord) -> Path | None:
        if not lesson.audio_file:
            return None
        path = folder / lesson.audio_file
        return path if path.is_file() else None

    def _attach_demo_audio(self, folder: Path, lesson: LessonRecord) -> None:
        audio_path = folder / "demo_english.wav"
        try:
            if not audio_path.exists():
                self._synthesize_demo_audio(audio_path)
            with wave.open(str(audio_path), "rb") as source:
                duration = source.getnframes() / max(1, source.getframerate())
            segment_duration = duration / max(1, len(lesson.segments))
            for index, segment in enumerate(lesson.segments):
                segment.start = index * segment_duration
                segment.end = (index + 1) * segment_duration
            lesson.audio_file = audio_path.name
            lesson.source_audio_name = audio_path.name
            self.store.save(folder, lesson)
        except Exception as exc:
            (folder / "demo_audio_error.txt").write_text(str(exc), encoding="utf-8")

    @staticmethod
    def _synthesize_demo_audio(audio_path: Path) -> None:
        text = (
            "Brain drain means skilled workers leave their home country. "
            "This can cause shortages of doctors, nurses, and teachers. "
            "Think about one possible solution with your partner."
        )
        from .platform_support import IS_WINDOWS

        if not IS_WINDOWS:
            DemoContent._synthesize_demo_audio_say(audio_path, text)
            return
        script = (
            "$ErrorActionPreference='Stop';"
            "Add-Type -AssemblyName System.Speech;"
            "$voice=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$voice.Rate=-1;"
            "$voice.SetOutputToWaveFile($env:OTOWEAVE_DEMO_WAV);"
            "$voice.Speak($env:OTOWEAVE_DEMO_TEXT);"
            "$voice.Dispose();"
        )
        environment = dict(os.environ)
        environment["OTOWEAVE_DEMO_WAV"] = str(audio_path)
        environment["OTOWEAVE_DEMO_TEXT"] = text
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-STA",
                "-Command",
                script,
            ],
            capture_output=True,
            timeout=45,
            env=environment,
        )
        if result.returncode != 0 or not audio_path.is_file():
            message = (
                result.stderr.decode("utf-8", errors="replace").strip()
                or "Windows音声合成でデモ音声を作成できませんでした。"
            )
            raise RuntimeError(message)

    @staticmethod
    def _synthesize_demo_audio_say(audio_path: Path, text: str) -> None:
        """macOS/Linux demo audio via the built-in `say` command (16 kHz
        mono little-endian PCM WAVE, so the stdlib wave module can read it)."""
        result = subprocess.run(
            [
                "say",
                "-o", str(audio_path),
                "--file-format=WAVE",
                "--data-format=LEI16@16000",
                text,
            ],
            capture_output=True,
            timeout=45,
        )
        if result.returncode != 0 or not audio_path.is_file():
            message = (
                result.stderr.decode("utf-8", errors="replace").strip()
                or "音声合成（say）でデモ音声を作成できませんでした。"
            )
            raise RuntimeError(message)
