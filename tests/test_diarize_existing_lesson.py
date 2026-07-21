"""既存レッスンへの事後話者分離（ASRを再実行せず、保存済み音声から話者分離だけを
後がけする機能）のテスト。

- TranscriptionService.diarize_existing_lesson: 音声ロード→話者分離→
  transcript保存の一連、カスタム話者名の保護、音声/文字起こし欠如時の安全な失敗。
- LearningAccessController.diarize_lesson_async: 既存の非同期実行パターン
  （busy判定・SessionState・イベント）への配線。
"""
import queue as queue_module
import tempfile
import time
import unittest
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from otoweave_app.controller import LearningAccessController, SessionState
from otoweave_app.diarization import DiarizationResult, DiarizedSpan
from otoweave_app.models import LessonRecord, TranscriptSegment
from otoweave_app.storage import LessonStore
from otoweave_app.transcription_service import TranscriptionService


def _write_wave(path: Path, seconds: float = 10.0, sample_rate: int = 16000) -> None:
    samples = np.zeros(int(seconds * sample_rate), dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


class FakeDiarizer:
    """2話者、前半/後半で綺麗に分かれる決め打ちの分離結果を返す。"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def diarize(self, samples, num_speakers):
        assert num_speakers == 2
        return DiarizationResult(
            spans=[
                DiarizedSpan(0.0, 5.0, 0),
                DiarizedSpan(5.0, 10.0, 1),
            ]
        )


class DiarizeExistingLessonServiceTests(unittest.TestCase):
    """TranscriptionService.diarize_existing_lesson の契約。"""

    def _make_lesson_with_audio(self, store: LessonStore) -> tuple[Path, LessonRecord]:
        lesson = LessonRecord.create(
            "japanese",
            "microphone",
            now=datetime.fromisoformat("2026-07-01T09:00:00+09:00"),
        )
        lesson.status = "complete"
        lesson.audio_file = "audio.wav"
        folder = store.create_lesson(lesson)
        _write_wave(folder / "audio.wav")
        return folder, lesson

    def _service(self, store: LessonStore):
        project_root = Path(__file__).resolve().parent.parent
        events: "queue_module.Queue" = queue_module.Queue()
        ready_calls: list[tuple[Path, LessonRecord]] = []
        service = TranscriptionService(
            project_root,
            store,
            events,
            ffmpeg=Path("ffmpeg"),
            correct_text=lambda text: text,
            on_lesson_ready=lambda f, l: ready_calls.append((f, l)),
        )
        return service, events, ready_calls

    @staticmethod
    def _drain(events: "queue_module.Queue") -> list[str]:
        kinds = []
        while not events.empty():
            kind, _ = events.get_nowait()
            kinds.append(kind)
        return kinds

    def test_success_assigns_speakers_and_persists_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            folder, lesson = self._make_lesson_with_audio(store)
            lesson.segments = [
                TranscriptSegment("seg_0001", 0.0, 5.0, "先生の発話です。"),
                TranscriptSegment("seg_0002", 5.0, 10.0, "生徒の発話です。"),
            ]
            store.save(folder, lesson)

            service, events, ready_calls = self._service(store)
            with patch(
                "otoweave_app.transcription_service.SpeakerDiarizer", FakeDiarizer
            ):
                result = service.diarize_existing_lesson(folder, lesson, 2)

            self.assertTrue(result)
            self.assertEqual(lesson.segments[0].speaker, "話者1")
            self.assertEqual(lesson.segments[1].speaker, "話者2")
            self.assertEqual(len(ready_calls), 1)

            stored = store.load(folder)
            self.assertEqual(stored.segments[0].speaker, "話者1")
            self.assertEqual(stored.segments[1].speaker, "話者2")

            kinds = self._drain(events)
            self.assertIn("status", kinds)
            self.assertNotIn("error", kinds)

    def test_custom_speaker_name_is_protected_from_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            folder, lesson = self._make_lesson_with_audio(store)
            lesson.segments = [
                TranscriptSegment(
                    "seg_0001", 0.0, 5.0, "先生の発話です。", speaker="山田先生"
                ),
                TranscriptSegment("seg_0002", 5.0, 10.0, "生徒の発話です。"),
            ]
            store.save(folder, lesson)

            service, events, _ = self._service(store)
            with patch(
                "otoweave_app.transcription_service.SpeakerDiarizer", FakeDiarizer
            ):
                result = service.diarize_existing_lesson(folder, lesson, 2)

            self.assertTrue(result)
            # 手動でリネームした話者名は上書きされない。
            self.assertEqual(lesson.segments[0].speaker, "山田先生")
            # 空だったセグメントには新しい分離結果が入る。
            self.assertEqual(lesson.segments[1].speaker, "話者2")

            stored = store.load(folder)
            self.assertEqual(stored.segments[0].speaker, "山田先生")

    def test_previous_auto_label_is_eligible_for_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            folder, lesson = self._make_lesson_with_audio(store)
            # 以前の話者分離で付いた自動ラベル(話者N)は、素の名前扱いとして
            # 新しい分離結果で上書きしてよい。
            lesson.segments = [
                TranscriptSegment("seg_0001", 0.0, 5.0, "先生の発話です。", speaker="話者9"),
                TranscriptSegment("seg_0002", 5.0, 10.0, "生徒の発話です。", speaker="話者9"),
            ]
            store.save(folder, lesson)

            service, _, _ = self._service(store)
            with patch(
                "otoweave_app.transcription_service.SpeakerDiarizer", FakeDiarizer
            ):
                result = service.diarize_existing_lesson(folder, lesson, 2)

            self.assertTrue(result)
            self.assertEqual(lesson.segments[0].speaker, "話者1")
            self.assertEqual(lesson.segments[1].speaker, "話者2")

    def test_missing_audio_file_fails_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            lesson = LessonRecord.create(
                "japanese",
                "microphone",
                now=datetime.fromisoformat("2026-07-01T09:00:00+09:00"),
            )
            lesson.status = "complete"
            lesson.audio_file = "audio.wav"  # ファイルは実際には存在しない
            lesson.segments = [TranscriptSegment("seg_0001", 0.0, 5.0, "テストです。")]
            folder = store.create_lesson(lesson)
            store.save(folder, lesson)

            service, events, ready_calls = self._service(store)
            result = service.diarize_existing_lesson(folder, lesson, 2)

            self.assertFalse(result)
            self.assertEqual(lesson.segments[0].speaker, "")
            self.assertEqual(ready_calls, [])
            stored = store.load(folder)
            self.assertEqual(stored.segments[0].text, "テストです。")
            kinds = self._drain(events)
            self.assertIn("error", kinds)

    def test_no_transcript_fails_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            folder, lesson = self._make_lesson_with_audio(store)
            lesson.segments = []
            store.save(folder, lesson)

            service, events, ready_calls = self._service(store)
            result = service.diarize_existing_lesson(folder, lesson, 2)

            self.assertFalse(result)
            self.assertEqual(ready_calls, [])
            kinds = self._drain(events)
            self.assertIn("error", kinds)

    def test_diarizer_failure_leaves_transcript_untouched(self) -> None:
        class RaisingDiarizer:
            def __init__(self, *args, **kwargs) -> None:
                raise RuntimeError("boom: diarization model failed to load")

        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            folder, lesson = self._make_lesson_with_audio(store)
            lesson.segments = [TranscriptSegment("seg_0001", 0.0, 5.0, "テストです。")]
            store.save(folder, lesson)

            service, events, ready_calls = self._service(store)
            with patch(
                "otoweave_app.transcription_service.SpeakerDiarizer",
                RaisingDiarizer,
            ):
                result = service.diarize_existing_lesson(folder, lesson, 2)

            self.assertFalse(result)
            self.assertEqual(lesson.segments[0].text, "テストです。")
            self.assertEqual(lesson.segments[0].speaker, "")
            self.assertEqual(ready_calls, [])
            kinds = self._drain(events)
            self.assertIn("error", kinds)


class ControllerDiarizeLessonAsyncTests(unittest.TestCase):
    """LearningAccessController.diarize_lesson_async の配線と状態管理。"""

    @staticmethod
    def write_demo_wave(path: Path) -> None:
        samples = np.zeros(16000, dtype=np.int16)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(samples.tobytes())

    def _prepared_controller(self, temporary: str) -> LearningAccessController:
        project_root = Path(__file__).resolve().parent.parent
        root = Path(temporary)
        source = root / "2026-07-01_test_lesson.wav"
        self.write_demo_wave(source)
        controller = LearningAccessController(project_root, root / "LearningAccess")
        controller.import_audio_async(source, "record_only")
        deadline = time.time() + 20
        while controller.busy and time.time() < deadline:
            time.sleep(0.05)
        # record_only の取り込みはASRを行わないため、あとから確認できる
        # 文字起こしを手動で用意しておく。
        controller.current_lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 1.0, "テストの発話です。")
        ]
        controller.store.save(controller.current_folder, controller.current_lesson)
        return controller

    def test_reaches_service_with_folder_lesson_and_num_speakers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._prepared_controller(temporary)
            folder = controller.current_folder
            with patch.object(
                controller.transcription, "diarize_existing_lesson", return_value=True
            ) as mock_diarize:
                controller.diarize_lesson_async(3)
                deadline = time.time() + 20
                while controller.busy and time.time() < deadline:
                    time.sleep(0.05)

            mock_diarize.assert_called_once()
            call_args = mock_diarize.call_args.args
            self.assertEqual(call_args[0], folder)
            self.assertEqual(call_args[2], 3)
            self.assertFalse(controller.busy)
            self.assertEqual(controller.state, SessionState.IDLE)
            self.assertIsNone(controller._processing_folder)

            kinds = []
            while not controller.events.empty():
                kind, _ = controller.events.get_nowait()
                kinds.append(kind)
            self.assertIn("diarization_started", kinds)
            self.assertIn("diarization_finished", kinds)

    def test_finished_event_fires_even_when_service_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._prepared_controller(temporary)
            with patch.object(
                controller.transcription, "diarize_existing_lesson", return_value=False
            ):
                controller.diarize_lesson_async(2)
                deadline = time.time() + 20
                while controller.busy and time.time() < deadline:
                    time.sleep(0.05)

            self.assertFalse(controller.busy)
            self.assertEqual(controller.state, SessionState.IDLE)
            kinds = []
            while not controller.events.empty():
                kind, _ = controller.events.get_nowait()
                kinds.append(kind)
            self.assertIn("diarization_finished", kinds)

    def test_invalid_num_speakers_raises_before_starting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._prepared_controller(temporary)
            with patch.object(
                controller.transcription, "diarize_existing_lesson"
            ) as mock_diarize:
                with self.assertRaises(ValueError):
                    controller.diarize_lesson_async(0)
            mock_diarize.assert_not_called()

    def test_missing_transcript_raises_before_starting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._prepared_controller(temporary)
            controller.current_lesson.segments = []
            with patch.object(
                controller.transcription, "diarize_existing_lesson"
            ) as mock_diarize:
                with self.assertRaises(RuntimeError):
                    controller.diarize_lesson_async(2)
            mock_diarize.assert_not_called()

    def test_busy_prevents_concurrent_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._prepared_controller(temporary)
            controller._state = SessionState.TRANSCRIBING
            try:
                with patch.object(
                    controller.transcription, "diarize_existing_lesson"
                ) as mock_diarize:
                    with self.assertRaises(RuntimeError):
                        controller.diarize_lesson_async(2)
                mock_diarize.assert_not_called()
            finally:
                controller._state = SessionState.IDLE

    def test_no_current_lesson_raises_runtime_error(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            controller = LearningAccessController(project_root, Path(temporary) / "LearningAccess")
            with self.assertRaises(RuntimeError):
                controller.diarize_lesson_async(2)


if __name__ == "__main__":
    unittest.main()
