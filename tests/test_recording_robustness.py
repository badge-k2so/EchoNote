"""録音の死活監視・保存スロットリング・文字起こしキャンセルの堅牢性テスト。"""
import queue as queue_module
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from otoweave_app.asr import Qwen3AsrRecognizer, RecognizedSentence
from otoweave_app.audio import SAMPLE_RATE, AudioRecorder, AudioSource, SpeechChunk
from otoweave_app.controller import LearningAccessController, SessionState
from otoweave_app.models import LessonRecord, TranscriptSegment
from otoweave_app.storage import LessonStore
from otoweave_app.transcription_service import TranscriptionService

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def wait_until(condition, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def drain_events(events: "queue_module.Queue") -> list[tuple[str, object]]:
    items = []
    while True:
        try:
            items.append(events.get_nowait())
        except queue_module.Empty:
            return items


class FakeRecorder:
    """AudioRecorder互換の最小フェイク（ウォッチドッグ・保存経路用）。"""

    def __init__(self) -> None:
        self.failed = False
        self.paused = False
        self.seconds_since_last_data = 0.0
        self.elapsed_seconds = 1.0
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def make_controller(temporary: str) -> LearningAccessController:
    return LearningAccessController(PROJECT_ROOT, Path(temporary) / "LearningAccess")


def make_recording_lesson(controller: LearningAccessController) -> Path:
    lesson = LessonRecord.create(
        "japanese", "microphone", now=datetime.fromisoformat("2026-07-07T09:00:00+09:00")
    )
    folder = controller.store.create_lesson(lesson)
    controller.current_lesson = lesson
    controller.current_folder = folder
    return folder


class RecorderLivenessSignalTests(unittest.TestCase):
    """AudioRecorder が死活監視用の情報を外部へ公開すること。"""

    def _make_recorder(self, temporary: str) -> AudioRecorder:
        source = AudioSource(
            id="microphone:0",
            label="テストマイク",
            device_index=0,
            sample_rate=SAMPLE_RATE,
            channels=1,
            kind="microphone",
        )
        return AudioRecorder(
            source=source,
            output_pcm=Path(temporary) / "out.pcm",
            on_speech_chunk=lambda chunk: None,
            on_error=lambda message: None,
        )

    def test_seconds_since_last_data_grows_from_last_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self._make_recorder(temporary)
            recorder._last_data_monotonic = time.monotonic() - 12.0
            self.assertGreaterEqual(recorder.seconds_since_last_data, 12.0)
            recorder._last_data_monotonic = time.monotonic()
            self.assertLess(recorder.seconds_since_last_data, 1.0)

    def test_failed_flag_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self._make_recorder(temporary)
            self.assertFalse(recorder.failed)
            recorder._failed.set()
            self.assertTrue(recorder.failed)


class RecordingWatchdogTests(unittest.TestCase):
    """RECORDING中のウォッチドッグが失敗検知・データ途絶警告を行うこと。"""

    def test_failed_recorder_triggers_auto_stop_and_plain_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = make_controller(temporary)
            folder = make_recording_lesson(controller)
            fake = FakeRecorder()
            fake.failed = True
            controller.recorder = fake
            controller._state = SessionState.RECORDING
            controller.WATCHDOG_INTERVAL_SECONDS = 0.05

            controller._start_recording_watchdog(fake)

            self.assertTrue(
                wait_until(lambda: controller.state is SessionState.IDLE),
                "録音が自動停止して IDLE に戻ること",
            )
            self.assertTrue(fake.stopped)
            self.assertIsNone(controller.recorder)

            events = drain_events(controller.events)
            errors = [payload for kind, payload in events if kind == "error"]
            self.assertTrue(
                any("録音を保存できなくなったため停止しました" in str(e) for e in errors)
            )
            self.assertTrue(
                any("ここまでの録音は保存されています" in str(e) for e in errors)
            )
            kinds = [kind for kind, _ in events]
            self.assertIn("lesson_finished", kinds)
            # ここまでの録音とtranscriptが確定保存されていること
            loaded = controller.store.load(folder)
            self.assertEqual(loaded.status, "complete")
            # 生徒向け文言に技術用語を出さないこと
            for error in errors:
                self.assertNotIn("Exception", str(error))
                self.assertNotIn("disk", str(error).lower())

    def test_stalled_recorder_warns_once_and_rearms_after_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = make_controller(temporary)
            make_recording_lesson(controller)
            fake = FakeRecorder()
            fake.seconds_since_last_data = 30.0
            controller.recorder = fake
            controller._state = SessionState.RECORDING
            controller.WATCHDOG_INTERVAL_SECONDS = 0.03

            controller._start_recording_watchdog(fake)
            try:
                self.assertTrue(
                    wait_until(
                        lambda: any(
                            kind == "error" for kind, _ in list(controller.events.queue)
                        )
                    )
                )
                time.sleep(0.3)  # さらに数周期回しても警告が増えないこと
                warnings = [
                    payload
                    for kind, payload in drain_events(controller.events)
                    if kind == "error"
                ]
                self.assertEqual(len(warnings), 1)
                self.assertIn("マイクの音が届いていません", str(warnings[0]))

                # 回復すると再警告できる状態に戻る
                fake.seconds_since_last_data = 0.0
                time.sleep(0.2)
                self.assertEqual(drain_events(controller.events), [])

                fake.seconds_since_last_data = 30.0
                self.assertTrue(
                    wait_until(
                        lambda: any(
                            kind == "error" for kind, _ in list(controller.events.queue)
                        )
                    ),
                    "再度の途絶で警告がもう一度出ること",
                )
            finally:
                controller._watchdog_stop.set()

    def test_stalled_warning_suppressed_while_paused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = make_controller(temporary)
            make_recording_lesson(controller)
            fake = FakeRecorder()
            fake.paused = True
            fake.seconds_since_last_data = 30.0
            controller.recorder = fake
            controller._state = SessionState.RECORDING
            controller.WATCHDOG_INTERVAL_SECONDS = 0.03

            controller._start_recording_watchdog(fake)
            try:
                time.sleep(0.3)
                kinds = [kind for kind, _ in drain_events(controller.events)]
                self.assertNotIn("error", kinds)
            finally:
                controller._watchdog_stop.set()


class LiveSaveThrottleTests(unittest.TestCase):
    """ライブ録音の確定文保存が時間ベースで間引かれること。"""

    def test_rapid_sentences_are_throttled_but_marks_save_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = make_controller(temporary)
            make_recording_lesson(controller)
            controller.recorder = FakeRecorder()
            controller._last_live_save_monotonic = 0.0
            controller._live_save_dirty = False

            with patch.object(
                controller.store, "save", wraps=controller.store.save
            ) as save_mock:
                controller._on_sentences([RecognizedSentence(0.0, 1.0, "最初の文です。")])
                self.assertEqual(save_mock.call_count, 1)
                self.assertFalse(controller._live_save_dirty)

                # 3秒未満の連続保存は dirty だけ立ててスキップされる
                controller._on_sentences([RecognizedSentence(1.0, 2.0, "二番目の文です。")])
                controller._on_sentences([RecognizedSentence(2.0, 3.0, "三番目の文です。")])
                self.assertEqual(save_mock.call_count, 1)
                self.assertTrue(controller._live_save_dirty)

                # 3秒経過後の次の契機で保存される
                controller._last_live_save_monotonic -= 4.0
                controller._on_sentences([RecognizedSentence(3.0, 4.0, "四番目の文です。")])
                self.assertEqual(save_mock.call_count, 2)
                self.assertFalse(controller._live_save_dirty)

                # ユーザー操作（マーク）は従来どおり即時保存
                segment_id = controller.current_lesson.segments[0].id
                controller.toggle_segment_mark(segment_id, "important")
                self.assertEqual(save_mock.call_count, 3)

    def test_stop_flushes_pending_sentences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = make_controller(temporary)
            folder = make_recording_lesson(controller)
            controller.recorder = FakeRecorder()
            controller._state = SessionState.RECORDING
            controller._last_live_save_monotonic = 0.0

            controller._on_sentences([RecognizedSentence(0.0, 1.0, "保存済みの文です。")])
            controller._on_sentences([RecognizedSentence(1.0, 2.0, "まだ書かれていない文です。")])
            self.assertTrue(controller._live_save_dirty)

            controller.stop_lesson_async()
            self.assertTrue(wait_until(lambda: controller.state is SessionState.IDLE))
            self.assertFalse(controller._live_save_dirty)
            loaded = controller.store.load(folder)
            texts = "".join(segment.text for segment in loaded.segments)
            self.assertIn("まだ書かれていない文です", texts)


class FakeChunkRecognizer:
    def __init__(self, text: str = "こんにちは。") -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, samples) -> str:
        self.calls += 1
        return self.text


class CancellingRecognizer(FakeChunkRecognizer):
    """最初のチャンク認識中にキャンセル要求が入った状況を再現する。"""

    def __init__(self, service: TranscriptionService, text: str = "こんにちは。") -> None:
        super().__init__(text)
        self.service = service

    def transcribe(self, samples) -> str:
        result = super().transcribe(samples)
        self.service.cancel_current()
        return result


def make_service(store: LessonStore) -> tuple[TranscriptionService, "queue_module.Queue", list]:
    events: "queue_module.Queue" = queue_module.Queue()
    ready_calls: list = []
    service = TranscriptionService(
        PROJECT_ROOT,
        store,
        events,
        ffmpeg=Path("ffmpeg"),
        correct_text=lambda text: text,
        on_lesson_ready=lambda folder, lesson: ready_calls.append((folder, lesson)),
    )
    return service, events, ready_calls


def noisy_pcm_bytes() -> bytes:
    """VADが2つの発話チャンクを切り出す 2s雑音+1.5s無音 ×2 のPCM。"""
    rng = np.random.default_rng(7)

    def noise(seconds: float) -> np.ndarray:
        return (rng.uniform(-1.0, 1.0, int(SAMPLE_RATE * seconds)) * 8000).astype(np.int16)

    def silence(seconds: float) -> np.ndarray:
        return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.int16)

    return np.concatenate(
        [noise(2.0), silence(1.5), noise(2.0), silence(1.5)]
    ).tobytes()


class ImportSaveThrottleTests(unittest.TestCase):
    """取り込みのチャンク保存が時間ベースで間引かれること。"""

    def test_chunk_saves_are_throttled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            service, events, _ = make_service(store)
            lesson = LessonRecord.create(
                "japanese", "imported", now=datetime.fromisoformat("2026-07-07T09:00:00+09:00")
            )
            folder = store.create_lesson(lesson)
            recognizer = FakeChunkRecognizer()
            chunk = SpeechChunk(0.0, 1.0, np.zeros(SAMPLE_RATE, dtype=np.int16))

            service._begin_pipeline()
            with patch.object(store, "save", wraps=store.save) as save_mock:
                service._recognize_import_chunk(recognizer, chunk, folder, lesson)
                self.assertEqual(save_mock.call_count, 1)

                service._recognize_import_chunk(recognizer, chunk, folder, lesson)
                service._recognize_import_chunk(recognizer, chunk, folder, lesson)
                self.assertEqual(save_mock.call_count, 1)
                self.assertTrue(service._progress_save_dirty)

                service._last_progress_save_monotonic -= 4.0
                service._recognize_import_chunk(recognizer, chunk, folder, lesson)
                self.assertEqual(save_mock.call_count, 2)
                self.assertFalse(service._progress_save_dirty)

            # UIイベントはチャンクごとに出続けること
            kinds = [kind for kind, _ in drain_events(events)]
            self.assertEqual(kinds.count("segments_changed"), 4)


class CancelRetranscribeTests(unittest.TestCase):
    """あとから文字起こしのキャンセルで既存segmentsが無傷なこと。"""

    def test_cancel_keeps_existing_segments_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            service, events, ready_calls = make_service(store)
            lesson = LessonRecord.create(
                "english", "microphone", now=datetime.fromisoformat("2026-07-07T09:00:00+09:00")
            )
            lesson.status = "complete"
            lesson.audio_file = "recording.pcm"
            lesson.segments = [
                TranscriptSegment("seg_0001", 0.0, 3.0, "元の大事な文字起こしです。", important=True),
                TranscriptSegment("seg_0002", 3.0, 6.0, "二つ目の文です。"),
            ]
            folder = store.create_lesson(lesson)
            (folder / "recording.pcm").write_bytes(noisy_pcm_bytes())
            original_texts = [segment.text for segment in lesson.segments]

            recognizer = CancellingRecognizer(service)
            with patch(
                "otoweave_app.transcription_service.JapaneseRecognizer",
                return_value=recognizer,
            ):
                service.transcribe_existing_audio(
                    folder / "recording.pcm", "japanese", folder, lesson
                )

            # 最初のチャンクの後、チャンク境界で停止していること
            self.assertEqual(recognizer.calls, 1)
            # 既存の文字起こし・マーク・言語モードは無傷
            self.assertEqual(
                [segment.text for segment in lesson.segments], original_texts
            )
            self.assertTrue(lesson.segments[0].important)
            self.assertEqual(lesson.language_mode, "english")
            stored = store.load(folder)
            self.assertEqual(
                [segment.text for segment in stored.segments], original_texts
            )
            self.assertEqual(ready_calls, [])
            # 一時ファイルのクリーンアップ
            self.assertFalse((folder / "transcription_input.pcm").exists())

            collected = drain_events(events)
            kinds = [kind for kind, _ in collected]
            self.assertIn("transcription_finished", kinds)
            self.assertNotIn("error", kinds)
            statuses = [str(payload) for kind, payload in collected if kind == "status"]
            self.assertTrue(
                any("あとから文字起こしを中止しました" in status for status in statuses)
            )
            self.assertTrue(
                any("元の文字起こしはそのまま残っています" in status for status in statuses)
            )


class CancelImportTests(unittest.TestCase):
    """取り込みキャンセルで途中までの結果が保存され complete になること。"""

    def test_cancel_saves_partial_result_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            service, events, _ = make_service(store)
            source = Path(temporary) / "授業音声.wav"
            source.write_bytes(b"dummy")
            lesson = LessonRecord.create(
                "japanese", "imported", now=datetime.fromisoformat("2026-07-07T09:00:00+09:00")
            )
            folder = store.create_lesson(lesson)

            recognizer = CancellingRecognizer(service, text="取り込み中の文です。")

            def fake_to_pcm(ffmpeg, source_path, pcm_path):
                Path(pcm_path).write_bytes(noisy_pcm_bytes())

            def fake_to_opus(ffmpeg, pcm_path, opus_path):
                Path(opus_path).write_bytes(b"opus")

            with patch(
                "otoweave_app.transcription_service.convert_audio_to_pcm",
                side_effect=fake_to_pcm,
            ), patch(
                "otoweave_app.transcription_service.convert_pcm_to_opus",
                side_effect=fake_to_opus,
            ), patch(
                "otoweave_app.transcription_service.JapaneseRecognizer",
                return_value=recognizer,
            ):
                service.import_audio(source, "japanese", folder, lesson)

            # 最初のチャンクだけ認識され、2つ目には進まないこと
            self.assertEqual(recognizer.calls, 1)
            # 途中までの認識結果を保存して complete
            self.assertEqual(lesson.status, "complete")
            self.assertEqual(lesson.audio_file, "audio.opus")
            self.assertTrue(any("取り込み中の文です" in s.text for s in lesson.segments))
            stored = store.load(folder)
            self.assertTrue(any("取り込み中の文です" in s.text for s in stored.segments))
            self.assertFalse((folder / "recording.pcm").exists())

            collected = drain_events(events)
            kinds = [kind for kind, _ in collected]
            self.assertIn("import_finished", kinds)
            self.assertNotIn("error", kinds)
            statuses = [str(payload) for kind, payload in collected if kind == "status"]
            self.assertTrue(
                any("取り込みを中止しました" in status for status in statuses)
            )
            self.assertTrue(
                any("途中までの文字起こしを保存しています" in status for status in statuses)
            )


class ControllerCancelApiTests(unittest.TestCase):
    """controller.cancel_transcription() の公開APIと close() 連携。"""

    def test_cancel_only_when_import_or_transcribe_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = make_controller(temporary)
            with patch.object(controller.transcription, "cancel_current") as cancel:
                self.assertFalse(controller.cancel_transcription())
                cancel.assert_not_called()

                controller._state = SessionState.IMPORTING
                self.assertTrue(controller.cancel_transcription())
                self.assertEqual(cancel.call_count, 1)

                controller._state = SessionState.TRANSCRIBING
                self.assertTrue(controller.cancel_transcription())
                self.assertEqual(cancel.call_count, 2)
                controller._state = SessionState.IDLE

    def test_close_requests_cancellation_of_running_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = make_controller(temporary)
            controller._state = SessionState.TRANSCRIBING
            with patch.object(controller.transcription, "cancel_current") as cancel:
                controller.close()
                cancel.assert_called_once()
            controller._state = SessionState.IDLE


class Qwen3AsrJobObjectTests(unittest.TestCase):
    """llama-server が kill-on-close Job へ登録されること。"""

    class FakeServerProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    def test_server_process_is_assigned_to_kill_on_close_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "engines/llama-b9763-cpu/llama-server.exe",
                "models/qwen3-asr-gguf/Qwen3-ASR-1.7B-Q8_0.gguf",
                "models/qwen3-asr-gguf/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"stub")

            fake_process = self.FakeServerProcess()
            with patch(
                "otoweave_app.asr.subprocess.Popen",
                return_value=fake_process,
            ), patch.object(
                Qwen3AsrRecognizer, "_wait_until_ready", return_value=None
            ), patch(
                "otoweave_app.asr._asr_kill_on_close_job",
                return_value=4321,
            ), patch(
                "otoweave_app.asr.assign_process_to_job"
            ) as assign:
                recognizer = Qwen3AsrRecognizer(root, root / "logs")
                assign.assert_called_once_with(4321, fake_process)
                recognizer.close()
            self.assertTrue(fake_process.terminated)


if __name__ == "__main__":
    unittest.main()
