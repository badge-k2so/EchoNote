import json
import queue as queue_module
import tempfile
import threading
import time
import unittest
import wave
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from otoweave_app.asr import (
    LanguageDecision,
    QWEN3_ASR_17_MMPROJ,
    QWEN3_ASR_17_MODEL,
    QWEN3_ASR_SERVER,
    Qwen3AsrRecognizer,
    SpeechBrainLanguageIdentifier,
    select_asr_threads,
    split_text_with_times,
)
from otoweave_app.audio import (
    AdaptiveVad,
    AudioProcessingOptions,
    AudioRecorder,
    AudioSource,
    convert_pcm_to_opus,
    process_audio_samples,
)
from otoweave_app.controller import LearningAccessController, SessionState
from otoweave_app.demo_content import DemoContent
from otoweave_app.display_settings import (
    DisplaySettings,
    available_reading_fonts,
    load_display_settings,
    save_display_settings,
)
from otoweave_app.models import LessonRecord, TranscriptSegment, append_readable_segment
from otoweave_app.storage import LessonStore, filter_lessons, safe_name
from otoweave_app.otoweave_app import OtoWeaveApp
from otoweave_app.customtkinter_views import MainPane
from otoweave_app.windows_process import decode_windows_process_output


class AudioProcessingOptionTests(unittest.TestCase):
    def test_sensitivity_changes_recorded_level(self) -> None:
        samples = np.array([1000, -1000, 2000, -2000], dtype=np.int16)

        processed = process_audio_samples(
            samples,
            AudioProcessingOptions(sensitivity=1.5),
        )

        np.testing.assert_array_equal(
            processed,
            np.array([1500, -1500, 3000, -3000], dtype=np.int16),
        )

    def test_noise_reduction_attenuates_quiet_input(self) -> None:
        samples = np.full(160, 100, dtype=np.int16)

        processed = process_audio_samples(
            samples,
            AudioProcessingOptions(noise_reduction=True),
        )

        self.assertLess(np.max(np.abs(processed)), 100)

    def test_agc_raises_quiet_input_without_clipping(self) -> None:
        samples = np.array([500, -500, 1000, -1000], dtype=np.int16)

        processed = process_audio_samples(
            samples,
            AudioProcessingOptions(automatic_gain_control=True),
        )

        self.assertGreater(np.max(np.abs(processed)), 1000)
        self.assertLessEqual(np.max(np.abs(processed)), 32767)


class WindowsProcessOutputTests(unittest.TestCase):
    def test_decodes_utf8_japanese_file_path(self) -> None:
        value = "C:\\録音\\英語授業.ogg".encode("utf-8")
        self.assertEqual(decode_windows_process_output(value), "C:\\録音\\英語授業.ogg")

    def test_decodes_cp932_error_message(self) -> None:
        value = "ファイルを開けません".encode("cp932")
        self.assertEqual(decode_windows_process_output(value), "ファイルを開けません")


class AudioRecorderStopSafetyTests(unittest.TestCase):
    @staticmethod
    def _recorder(folder: Path) -> tuple[AudioRecorder, list[str]]:
        source = AudioSource(
            id="microphone:0",
            label="test",
            device_index=0,
            sample_rate=16000,
            channels=1,
            kind="microphone",
        )
        errors: list[str] = []
        recorder = AudioRecorder(
            source,
            folder / "recording.pcm",
            on_speech_chunk=lambda chunk: None,
            on_error=errors.append,
        )
        return recorder, errors

    def test_stop_does_not_block_when_worker_died_and_queue_is_full(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder, errors = self._recorder(Path(temporary))

            class FailingFile:
                @staticmethod
                def write(_data) -> None:
                    raise OSError("No space left on device")

                @staticmethod
                def flush() -> None:
                    pass

                @staticmethod
                def close() -> None:
                    pass

            recorder._file = FailingFile()
            recorder._worker = threading.Thread(
                target=recorder._process_audio, daemon=True
            )
            recorder._worker.start()
            block = b"\x00\x00" * 1600
            recorder._queue.put(block)
            # Simulate the PortAudio callback filling the queue afterwards.
            for _ in range(200):
                try:
                    recorder._queue.put_nowait(block)
                except queue_module.Full:
                    break

            started = time.monotonic()
            recorder.stop()

            self.assertLess(time.monotonic() - started, 10.0)
            self.assertTrue(recorder.failed)
            self.assertTrue(any("書き込めません" in value for value in errors))

    def test_failed_recorder_callback_stops_enqueueing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder, _ = self._recorder(Path(temporary))
            recorder._failed.set()
            # stop() with no worker and a pre-filled queue must still return.
            for _ in range(200):
                try:
                    recorder._queue.put_nowait(b"\x00\x00")
                except queue_module.Full:
                    break
            recorder.stop()
            self.assertIsNone(recorder._worker)

    def test_start_failure_releases_pcm_file_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder, _ = self._recorder(Path(temporary))

            class FakePyAudio:
                @staticmethod
                def open(**_kwargs):
                    raise OSError("device unavailable")

                @staticmethod
                def terminate() -> None:
                    pass

            with patch("pyaudiowpatch.PyAudio", return_value=FakePyAudio()):
                with self.assertRaises(OSError):
                    recorder.start()

            self.assertIsNone(recorder._file)
            # The handle is released, so the lesson folder can be removed.
            (Path(temporary) / "recording.pcm").unlink()


class LiveAsrWorkerQueueTests(unittest.TestCase):
    @staticmethod
    def _worker() -> tuple["LiveAsrWorker", list[str], list[str]]:
        from otoweave_app.asr import LiveAsrWorker

        errors: list[str] = []
        statuses: list[str] = []
        worker = LiveAsrWorker(
            mode="japanese",
            project_root=Path("."),
            on_sentences=lambda sentences: None,
            on_status=statuses.append,
            on_error=errors.append,
        )
        return worker, errors, statuses

    @staticmethod
    def _chunk() -> "SpeechChunk":
        from otoweave_app.audio import SpeechChunk

        return SpeechChunk(start=0.0, end=1.0, samples=np.zeros(16, dtype=np.int16))

    def test_full_queue_drops_chunk_with_single_notice(self) -> None:
        worker, errors, _ = self._worker()
        chunk = self._chunk()
        for _ in range(worker.MAX_PENDING_CHUNKS):
            worker.submit(chunk)
        self.assertEqual(worker._queue.qsize(), worker.MAX_PENDING_CHUNKS)

        worker.submit(chunk)
        worker.submit(chunk)

        # Bounded: nothing beyond the cap, and the warning is throttled.
        self.assertEqual(worker._queue.qsize(), worker.MAX_PENDING_CHUNKS)
        self.assertEqual(len(errors), 1)
        self.assertIn("追いつかない", errors[0])

    def test_submit_after_close_is_noop(self) -> None:
        worker, errors, _ = self._worker()
        worker._closed.set()
        worker.submit(self._chunk())
        self.assertEqual(worker._queue.qsize(), 0)
        self.assertEqual(errors, [])

    def test_stop_with_full_queue_and_no_worker_does_not_hang(self) -> None:
        worker, _, _ = self._worker()
        chunk = self._chunk()
        for _ in range(worker.MAX_PENDING_CHUNKS):
            worker.submit(chunk)

        started = time.monotonic()
        worker.stop(timeout=1.0)

        self.assertLess(time.monotonic() - started, 15.0)
        self.assertTrue(worker._closed.is_set())
        worker.submit(chunk)
        self.assertLessEqual(worker._queue.qsize(), worker.MAX_PENDING_CHUNKS)


class LessonStoreDurabilityTests(unittest.TestCase):
    @staticmethod
    def _lesson() -> LessonRecord:
        lesson = LessonRecord.create(
            "japanese",
            "microphone",
            now=datetime.fromisoformat("2026-07-01T09:00:00+09:00"),
        )
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 3.0, "最初の文です。")
        ]
        return lesson

    def test_save_keeps_previous_version_as_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            lesson = self._lesson()
            folder = store.create_lesson(lesson)

            lesson.segments.append(
                TranscriptSegment("seg_0002", 3.0, 6.0, "二つ目の文です。")
            )
            store.save(folder, lesson)

            backup = folder / "transcript.json.bak"
            self.assertTrue(backup.exists())
            previous = json.loads(backup.read_text(encoding="utf-8"))
            self.assertEqual(len(previous["segments"]), 1)

    def test_corrupted_transcript_falls_back_to_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            lesson = self._lesson()
            folder = store.create_lesson(lesson)
            store.save(folder, lesson)  # creates the .bak

            # Simulate a power cut that truncated the main file.
            (folder / "transcript.json").write_text("", encoding="utf-8")

            restored = store.load(folder)
            self.assertEqual(restored.segments[0].text, "最初の文です。")

    def test_corrupted_lesson_stays_visible_as_needs_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            lesson = self._lesson()
            folder = store.create_lesson(lesson)

            # No .bak yet (single save) — destroy the only transcript.
            (folder / "transcript.json").write_text("{broken", encoding="utf-8")

            lessons = store.list_lessons()
            self.assertEqual(len(lessons), 1)
            listed_folder, listed = lessons[0]
            self.assertEqual(listed_folder, folder)
            self.assertEqual(listed.status, "needs_repair")
            # Title/date survive via metadata.json.
            self.assertEqual(listed.date, "2026-07-01")

    def test_totally_destroyed_folder_gets_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            lesson = self._lesson()
            folder = store.create_lesson(lesson)
            (folder / "transcript.json").write_text("{broken", encoding="utf-8")
            (folder / "metadata.json").write_text("{broken", encoding="utf-8")

            lessons = store.list_lessons()
            self.assertEqual(len(lessons), 1)
            self.assertEqual(lessons[0][1].status, "needs_repair")

    def test_invalid_date_lesson_is_skipped_by_filter_not_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            good = self._lesson()
            folder = store.create_lesson(good)
            broken = store.load(folder)
            broken.date = "not-a-date"

            result = filter_lessons(
                [(folder, good), (folder, broken)],
                "week",
                today=date.fromisoformat("2026-07-01"),
            )

            self.assertEqual(len(result), 1)


class TrashTests(unittest.TestCase):
    @staticmethod
    def _lesson() -> LessonRecord:
        lesson = LessonRecord.create(
            "japanese",
            "microphone",
            now=datetime.fromisoformat("2026-07-03T09:00:00+09:00"),
        )
        lesson.segments = [TranscriptSegment("seg_0001", 0.0, 3.0, "本文")]
        return lesson

    def test_trash_lesson_moves_folder_and_hides_it_from_listing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            folder = store.create_lesson(self._lesson())

            target = store.trash_lesson(folder)

            self.assertFalse(folder.exists())
            self.assertTrue(target.exists())
            self.assertTrue((target / "transcript.json").exists())
            self.assertEqual(store.list_lessons(), [])
            self.assertEqual(store.list_lesson_metadata(), [])
            self.assertEqual(store.search_transcripts("本文"), set())

    def test_purge_trash_removes_only_old_entries(self) -> None:
        import os as os_module

        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            old_folder = store.trash_lesson(store.create_lesson(self._lesson()))
            new_folder = store.trash_lesson(store.create_lesson(self._lesson()))

            ancient = time.time() - 40 * 86400
            os_module.utime(old_folder, (ancient, ancient))

            removed = store.purge_trash(keep_days=30)

            self.assertEqual(removed, 1)
            self.assertFalse(old_folder.exists())
            self.assertTrue(new_folder.exists())


class VadShortUtteranceTests(unittest.TestCase):
    @staticmethod
    def _loud_block(duration_seconds: float) -> np.ndarray:
        count = int(16000 * duration_seconds)
        return (np.sin(np.arange(count) * 0.5) * 20000).astype(np.int16)

    def test_short_utterance_is_emitted_on_forced_flush(self) -> None:
        vad = AdaptiveVad()
        chunks = vad.process(self._loud_block(0.4), 0.0)
        self.assertEqual(chunks, [])

        emitted = vad.flush(force=True)

        self.assertEqual(len(emitted), 1)
        self.assertLess(emitted[0].end - emitted[0].start, 0.8)

    def test_short_utterance_is_dropped_without_force_but_carried_over(self) -> None:
        vad = AdaptiveVad()
        vad.process(self._loud_block(0.4), 0.0)

        self.assertEqual(vad.flush(), [])
        # Carried over as pre-roll instead of thrown away.
        self.assertTrue(vad.pre_roll)


class DictionaryAliasSafetyTests(unittest.TestCase):
    def test_one_character_alias_is_rejected_on_save(self) -> None:
        from otoweave_app.user_dictionary import normalize_entry

        with self.assertRaises(ValueError):
            normalize_entry({"term": "感覚過敏", "aliases": "か"})

    def test_one_character_alias_is_dropped_leniently_on_load(self) -> None:
        from otoweave_app.user_dictionary import normalize_entry

        entry = normalize_entry(
            {"term": "感覚過敏", "aliases": ["か", "かんかくかびん"]},
            strict=False,
        )
        self.assertEqual(entry["aliases"], ["かんかくかびん"])

    def test_correct_text_never_uses_single_character_alias(self) -> None:
        from otoweave_app.user_dictionary import correct_text

        entries = [
            {"term": "感覚過敏", "aliases": ["か"]},
        ]
        self.assertEqual(correct_text("時間の話", entries), "時間の話")

    def test_description_is_flattened_to_one_line(self) -> None:
        from otoweave_app.user_dictionary import normalize_entry

        entry = normalize_entry(
            {"term": "用語", "description": "1行目\n2行目\n3行目"}
        )
        self.assertNotIn("\n", entry["description"])


class TemplateRecordQualityTests(unittest.TestCase):
    def test_all_empty_record_is_detected(self) -> None:
        from scripts.production.template_summarize import record_sections_all_empty

        record = "\n".join(
            f"## {section}\n- 該当なし"
            for section in (
                "今日のテーマ",
                "大事なポイント",
                "出てきた用語",
                "先生が強調したこと",
                "宿題・提出物",
                "あとで確認すること",
            )
        )
        self.assertTrue(record_sections_all_empty(record, "lesson_record"))

    def test_record_with_content_is_not_flagged(self) -> None:
        from scripts.production.template_summarize import record_sections_all_empty

        record = (
            "## 今日のテーマ\n- 分数の計算\n"
            "## 大事なポイント\n- 該当なし\n"
        )
        self.assertFalse(record_sections_all_empty(record, "lesson_record"))

    def test_truncated_record_gets_visible_warning(self) -> None:
        from scripts.production.template_summarize import add_record_note

        note = add_record_note("## 要約\n- 内容", truncated=True)
        self.assertIn("内容の一部が欠けている可能性", note)
        untouched = add_record_note("## 要約\n- 内容", truncated=False)
        self.assertNotIn("内容の一部が欠けている可能性", untouched)


class LightSaveTests(unittest.TestCase):
    def test_light_save_skips_markdown_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            lesson = LessonRecord.create(
                "japanese",
                "microphone",
                now=datetime.fromisoformat("2026-07-03T10:00:00+09:00"),
            )
            folder = store.create_lesson(lesson)
            markdown = folder / "transcript.md"
            baseline = markdown.read_text(encoding="utf-8")

            lesson.segments = [TranscriptSegment("seg_0001", 0.0, 3.0, "新しい文")]
            store.save(folder, lesson, light=True)

            # markdown untouched, transcript.json updated
            self.assertEqual(markdown.read_text(encoding="utf-8"), baseline)
            transcript = json.loads(
                (folder / "transcript.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(transcript["segments"]), 1)

            store.save(folder, lesson)
            self.assertIn("新しい文", markdown.read_text(encoding="utf-8"))


class MetadataListingTests(unittest.TestCase):
    @staticmethod
    def _lesson_with_segments() -> LessonRecord:
        lesson = LessonRecord.create(
            "japanese",
            "microphone",
            now=datetime.fromisoformat("2026-07-02T09:00:00+09:00"),
        )
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 90.0, "重要な説明です。", important=True),
            TranscriptSegment("seg_0002", 90.0, 200.0, "氷山モデルの話。"),
        ]
        return lesson

    def test_metadata_contains_listing_fields(self) -> None:
        metadata = self._lesson_with_segments().metadata_dict()
        self.assertEqual(metadata["duration_seconds"], 200.0)
        self.assertTrue(metadata["has_important"])
        self.assertFalse(metadata["has_question"])
        self.assertEqual(metadata["schema_version"], 4)

    def test_list_lesson_metadata_reads_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            lesson = self._lesson_with_segments()
            folder = store.create_lesson(lesson)

            # Corrupting the transcript must not affect the metadata list
            # (proof that transcripts are not read on the fast path).
            (folder / "transcript.json").write_text("{broken", encoding="utf-8")
            (folder / "transcript.json.bak").unlink(missing_ok=True)

            entries = store.list_lesson_metadata()

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0][1]["duration_seconds"], 200.0)

    def test_legacy_metadata_is_backfilled_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            lesson = self._lesson_with_segments()
            folder = store.create_lesson(lesson)

            # Simulate a schema-3 store: no listing fields yet.
            legacy = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            for key in ("duration_seconds", "has_important", "has_unclear", "has_question"):
                legacy.pop(key, None)
            (folder / "metadata.json").write_text(
                json.dumps(legacy, ensure_ascii=False),
                encoding="utf-8",
            )

            entries = store.list_lesson_metadata()
            self.assertEqual(entries[0][1]["duration_seconds"], 200.0)

            # Backfilled on disk: the next call is metadata-only.
            updated = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn("duration_seconds", updated)

    def test_search_transcripts_matches_body_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            lesson = self._lesson_with_segments()
            folder = store.create_lesson(lesson)
            del folder

            matched = store.search_transcripts("氷山モデル")
            self.assertEqual(matched, {lesson.lesson_id})
            self.assertEqual(store.search_transcripts("存在しない語"), set())

    def test_metadata_note_shows_duration_and_marks_without_body(self) -> None:
        from otoweave_app.otoweave_app import _metadata_to_note

        metadata = self._lesson_with_segments().metadata_dict()
        note = _metadata_to_note(Path("/tmp/lesson"), metadata)

        self.assertIn("03:20", note["meta"])
        self.assertIn("★重要", note["keywords"])
        self.assertFalse(note["_loaded"])
        self.assertFalse(note["has_transcript"])


class SummaryCancellationTests(unittest.TestCase):
    class FakeProcess:
        def __init__(self) -> None:
            self.killed = False

        def poll(self):
            return None if not self.killed else 1

        def kill(self) -> None:
            self.killed = True

    @staticmethod
    def _session() -> tuple:
        from otoweave_app.llm_session import LlmSession

        events: queue_module.Queue = queue_module.Queue()
        session = LlmSession(Path(__file__).resolve().parent.parent, events)
        return session, events

    def test_cancel_kills_registered_process(self) -> None:
        session, _ = self._session()
        process = self.FakeProcess()
        with session._lock:
            session._busy = True
        session._register_summary_process(process)

        self.assertTrue(session.cancel_summary())
        self.assertTrue(process.killed)

    def test_cancel_before_registration_kills_at_registration(self) -> None:
        session, _ = self._session()
        with session._lock:
            session._busy = True
        self.assertTrue(session.cancel_summary())

        process = self.FakeProcess()
        session._register_summary_process(process)

        self.assertTrue(process.killed)

    def test_cancel_when_idle_returns_false(self) -> None:
        session, _ = self._session()
        self.assertFalse(session.cancel_summary())

    def test_cancelled_summary_emits_llm_cancelled_event(self) -> None:
        session, events = self._session()
        folder = Path("lesson_x")

        def fake_run(lesson, lesson_folder, project_root, model_path,
                     on_process=None, on_progress=None):
            del lesson, lesson_folder, project_root, model_path
            del on_process, on_progress
            session.cancel_summary()
            raise RuntimeError("process killed")

        with patch(
            "otoweave_app.llm_chat.run_summarize_subprocess",
            side_effect=fake_run,
        ):
            session.summarize_async(None, folder, Path("model.gguf"))
            kinds: list[str] = []
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    kind, _payload = events.get(timeout=0.5)
                except queue_module.Empty:
                    continue
                kinds.append(kind)
                if kind in {"llm_cancelled", "llm_error"}:
                    break

        self.assertIn("llm_cancelled", kinds)
        self.assertNotIn("llm_error", kinds)
        self.assertFalse(session.busy)

    def test_failed_summary_without_cancel_emits_llm_error(self) -> None:
        session, events = self._session()

        with patch(
            "otoweave_app.llm_chat.run_summarize_subprocess",
            side_effect=RuntimeError("boom"),
        ):
            session.summarize_async(None, Path("lesson_y"), Path("model.gguf"))
            kinds: list[str] = []
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    kind, _payload = events.get(timeout=0.5)
                except queue_module.Empty:
                    continue
                kinds.append(kind)
                if kind in {"llm_cancelled", "llm_error"}:
                    break

        self.assertIn("llm_error", kinds)

    def test_kill_on_close_job_is_available_on_windows(self) -> None:
        from otoweave_app.windows_job import create_kill_on_close_job

        self.assertIsNotNone(create_kill_on_close_job())


class OtoWeaveUiSafetyTests(unittest.TestCase):
    def test_elapsed_timer_reads_recorder_reference_once(self) -> None:
        recorder = SimpleNamespace(elapsed_seconds=12.0)

        class FlakyController:
            def __init__(self) -> None:
                self.read_count = 0
                self.recording = True

            @property
            def recorder(self):
                self.read_count += 1
                return recorder if self.read_count == 1 else None

        controller = FlakyController()
        main_pane = SimpleNamespace(set_elapsed=Mock())
        app = SimpleNamespace(
            controller=controller,
            main_pane=main_pane,
            after=Mock(return_value="after-id"),
            _elapsed_start=0.0,
            _elapsed_after_id="",
            _tick_elapsed=Mock(),
        )

        OtoWeaveApp._tick_elapsed(app)

        self.assertEqual(controller.read_count, 1)
        main_pane.set_elapsed.assert_called_once_with("00:12")

    def test_empty_transcript_edit_is_not_saved(self) -> None:
        pane = SimpleNamespace(
            textbox=SimpleNamespace(get=lambda *_args: "   \n  "),
            status_label=SimpleNamespace(configure=Mock()),
            request_transcript_save=Mock(),
        )

        MainPane._save_transcript_edit(pane)

        pane.request_transcript_save.assert_not_called()
        pane.status_label.configure.assert_called_once()

    def test_live_transcript_appends_only_new_tail(self) -> None:
        inserts: list[tuple[str, str]] = []
        deletes: list[tuple[str, str]] = []
        raw = SimpleNamespace(
            tag_remove=Mock(),
            tag_configure=Mock(),
            tag_add=Mock(),
        )
        textbox = SimpleNamespace(
            configure=Mock(),
            insert=lambda index, text: inserts.append((index, text)),
            delete=lambda start, end: deletes.append((start, end)),
            yview=lambda: (0.0, 1.0),
            see=Mock(),
            _textbox=raw,
        )
        pane = SimpleNamespace(
            textbox=textbox,
            return_live_button=SimpleNamespace(
                place_forget=Mock(),
                place=Mock(),
                lift=Mock(),
            ),
            _live_text="00:00  こんにちは",
            _live_active=True,
            _live_follow_enabled=True,
            _live_highlight_enabled=False,
            _live_at_bottom=lambda: True,
            _apply_live_highlight=lambda _text: None,
        )

        MainPane.update_live_transcript(
            pane,
            "00:00  こんにちは\n\n00:05  次の発話",
        )

        self.assertEqual(deletes, [])
        self.assertEqual(inserts, [("end", "\n\n00:05  次の発話")])
        self.assertEqual(pane._live_text, "00:00  こんにちは\n\n00:05  次の発話")
        textbox.see.assert_called_once_with("end")


class DisplaySettingsTests(unittest.TestCase):
    def test_reading_fonts_are_ordered_for_japanese_first(self) -> None:
        fonts = available_reading_fonts(
            ["Meiryo UI", "@BIZ UDPゴシック", "OpenDyslexic", "BIZ UDPゴシック", "Yu Gothic UI"]
        )
        self.assertEqual(
            fonts,
            ("BIZ UDPゴシック", "OpenDyslexic", "Yu Gothic UI", "Meiryo UI"),
        )

    def test_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "display_settings.json"
            expected = DisplaySettings(
                text_size="Extra Large",
                font_family="BIZ UDPゴシック",
                color_mode="Dark",
                live_follow=False,
            )
            save_display_settings(path, expected)
            self.assertEqual(load_display_settings(path), expected)

    def test_invalid_settings_return_safe_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "display_settings.json"
            path.write_text('{"text_size":"Huge","color_mode":"Black"}', encoding="utf-8")
            loaded = load_display_settings(path)
            self.assertEqual(loaded.text_size, "Standard")
            self.assertEqual(loaded.color_mode, "Light")


class AsrThreadSelectionTests(unittest.TestCase):
    def test_live_mode_uses_at_most_two_threads(self) -> None:
        self.assertEqual(select_asr_threads("live", logical_cpus=1).num_threads, 1)
        self.assertEqual(select_asr_threads("live", logical_cpus=2).num_threads, 2)
        self.assertEqual(select_asr_threads("live", logical_cpus=8).num_threads, 2)

    def test_file_mode_uses_at_most_four_threads(self) -> None:
        self.assertEqual(select_asr_threads("file", logical_cpus=1).num_threads, 1)
        self.assertEqual(select_asr_threads("file", logical_cpus=2).num_threads, 2)
        self.assertEqual(select_asr_threads("file", logical_cpus=8).num_threads, 4)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_asr_threads("unknown", logical_cpus=4)


class LanguageRoutingTests(unittest.TestCase):
    @staticmethod
    def identifier_with_scores(japanese: float, english: float):
        identifier = SpeechBrainLanguageIdentifier.__new__(SpeechBrainLanguageIdentifier)
        identifier.japanese_index = 0
        identifier.english_index = 1

        class FakeSession:
            @staticmethod
            def run(_outputs, _inputs):
                return [np.asarray([[japanese, english]], dtype=np.float32)]

        identifier.session = FakeSession()
        return identifier

    def test_clear_english_score_routes_to_english(self) -> None:
        identifier = self.identifier_with_scores(0.02, 0.70)
        decision = identifier.detect(np.ones(32000, dtype=np.int16))
        self.assertEqual(decision.language, "english")
        self.assertFalse(decision.uncertain)

    def test_short_clear_english_score_still_routes_to_english(self) -> None:
        identifier = self.identifier_with_scores(0.01, 0.90)
        decision = identifier.detect(np.ones(8000, dtype=np.int16))
        self.assertEqual(decision.language, "english")
        self.assertTrue(decision.uncertain)
        self.assertIn("short_audio", decision.reason)
        self.assertNotIn("japanese_default_route", decision.reason)

    def test_close_scores_use_japanese_default_and_require_review(self) -> None:
        identifier = self.identifier_with_scores(0.001, 0.006)
        decision = identifier.detect(np.ones(32000, dtype=np.int16))
        self.assertEqual(decision.language, "japanese")
        self.assertTrue(decision.uncertain)
        self.assertIn("japanese_default_route", decision.reason)


class LessonStoreTests(unittest.TestCase):
    def test_duplicate_lesson_id_gets_unique_folder_and_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            recorded_at = datetime.fromisoformat("2026-06-23T10:12:00+09:00")
            first = LessonRecord.create(
                "record_only",
                "imported",
                now=recorded_at,
            )
            second = LessonRecord.create(
                "record_only",
                "imported",
                now=recorded_at,
            )

            first_folder = store.create_lesson(first)
            second_folder = store.create_lesson(second)

            self.assertNotEqual(first_folder, second_folder)
            self.assertEqual(first.lesson_id, "2026-06-23_101200")
            self.assertEqual(second.lesson_id, "2026-06-23_101200_2")
            self.assertEqual(store.load(second_folder).lesson_id, second.lesson_id)

    def test_existing_fragmented_transcript_is_coalesced_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            lesson = LessonRecord.create(
                "japanese", "microphone", now=datetime.fromisoformat("2026-06-23T10:12:00+09:00")
            )
            lesson.segments = [
                TranscriptSegment("seg_0001", 1.0, 2.0, "今日の"),
                TranscriptSegment("seg_0002", 2.1, 3.0, "授業です", important=True),
            ]
            folder = store.create_lesson(lesson)
            loaded = store.load(folder)
            self.assertEqual(len(loaded.segments), 1)
            self.assertEqual(loaded.segments[0].text, "今日の、授業です")
            self.assertTrue(loaded.segments[0].important)

    def test_short_japanese_fragments_grow_into_one_readable_card(self) -> None:
        segments = [TranscriptSegment("seg_0001", 1.0, 2.0, "読み書きには")]
        result = append_readable_segment(
            segments, TranscriptSegment("seg_0002", 2.3, 3.5, "いろいろな方法があります")
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(result.text, "読み書きには、いろいろな方法があります")
        self.assertEqual(result.end, 3.5)

    def test_complete_english_sentences_stay_on_separate_cards(self) -> None:
        segments = [TranscriptSegment("seg_0001", 1.0, 2.0, "This is the first point.")]
        append_readable_segment(
            segments, TranscriptSegment("seg_0002", 2.2, 3.5, "This is the second point.")
        )
        self.assertEqual(len(segments), 2)

    def test_merged_card_does_not_grow_beyond_twelve_seconds(self) -> None:
        segments = [TranscriptSegment("seg_0001", 1.0, 8.0, "最初の説明")]
        append_readable_segment(
            segments, TranscriptSegment("seg_0002", 8.2, 13.2, "次の説明")
        )
        self.assertEqual(len(segments), 2)

    def test_lesson_files_and_marks_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary))
            lesson = LessonRecord.create(
                "japanese",
                "microphone",
                now=datetime.fromisoformat("2026-06-23T10:12:00+09:00"),
            )
            lesson.asr_processing_mode = "live"
            lesson.asr_threads = 2
            lesson.detected_logical_cpus = 8
            lesson.segments.append(
                TranscriptSegment(
                    id="seg_0001",
                    start=12.3,
                    end=17.8,
                    text="今日は読み書きの方法を確認します。",
                    important=True,
                    important_at="2026-06-23T10:13:20+09:00",
                    question=True,
                    question_at="2026-06-23T10:13:25+09:00",
                )
            )
            folder = store.create_lesson(lesson)

            self.assertIn("2026", folder.parts)
            self.assertTrue((folder / "transcript.json").exists())
            self.assertTrue((folder / "transcript.md").exists())
            self.assertTrue((folder / "marks.json").exists())
            self.assertTrue((folder / "metadata.json").exists())

            loaded = store.load(folder)
            self.assertTrue(loaded.segments[0].important)
            self.assertTrue(loaded.segments[0].question)
            self.assertEqual(loaded.segments[0].important_at, "2026-06-23T10:13:20+09:00")
            self.assertEqual(loaded.asr_processing_mode, "live")
            self.assertEqual(loaded.asr_threads, 2)
            self.assertEqual(loaded.detected_logical_cpus, 8)
            marks = json.loads((folder / "marks.json").read_text(encoding="utf-8"))
            self.assertEqual(marks["marks"][0]["type"], "important")
            self.assertEqual(marks["marks"][0]["created_at"], "2026-06-23T10:13:20+09:00")
            self.assertEqual(marks["marks"][1]["type"], "question")

    def test_review_edit_speaker_split_and_merge_round_trip(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = LearningAccessController(project_root, root / "LearningAccess")
            lesson = LessonRecord.create(
                "japanese", "microphone", now=datetime.fromisoformat("2026-06-27T10:00:00+09:00")
            )
            lesson.status = "complete"
            lesson.segments = [
                TranscriptSegment("seg_0001", 0.0, 8.0, "前半の説明。後半の説明。"),
                TranscriptSegment("seg_0002", 8.0, 12.0, "次の説明。", question=True),
            ]
            folder = controller.store.create_lesson(lesson)
            controller.select_lesson(folder)

            controller.update_segment_speaker("seg_0001", "講師")
            controller.split_segment("seg_0001", "前半の説明。", "後半の説明。")
            self.assertEqual(len(controller.current_lesson.segments), 3)
            self.assertEqual(controller.current_lesson.segments[0].speaker, "講師")
            self.assertTrue(controller.current_lesson.segments[0].edited)

            controller.merge_segment_with_next("seg_0001")
            merged_text = controller.current_lesson.segments[0].text
            controller.split_segment("seg_0001", "前半の説明。", merged_text[len("前半の説明。"):])
            self.assertEqual(
                len({segment.id for segment in controller.current_lesson.segments}),
                len(controller.current_lesson.segments),
            )
            controller.merge_segment_with_next("seg_0001")
            loaded = controller.store.load(folder)
            self.assertEqual(len(loaded.segments), 2)
            self.assertIn("前半の説明", loaded.segments[0].text)
            self.assertIn("後半の説明", loaded.segments[0].text)
            self.assertEqual(loaded.segments[0].speaker, "講師")

    def test_plain_review_edit_preserves_matching_segments(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            controller = LearningAccessController(project_root, Path(temporary) / "LearningAccess")
            lesson = LessonRecord.create(
                "japanese", "microphone", now=datetime.fromisoformat("2026-06-27T10:00:00+09:00")
            )
            lesson.segments = [
                TranscriptSegment("seg_0001", 0.0, 5.0, "最初の文章。"),
                TranscriptSegment("seg_0002", 8.0, 12.0, "次の文章。"),
            ]
            folder = controller.store.create_lesson(lesson)
            controller.select_lesson(folder)
            postprocess = folder / "postprocess"
            postprocess.mkdir()
            summary = postprocess / "school_record.md"
            summary.write_text("古い要約", encoding="utf-8")

            controller.replace_transcript_text("最初の文章を訂正。\n\n次の文章も訂正。")

            self.assertEqual(len(controller.current_lesson.segments), 2)
            self.assertEqual(controller.current_lesson.segments[0].start, 0.0)
            self.assertEqual(controller.current_lesson.segments[1].end, 12.0)
            self.assertTrue(controller.current_lesson.segments[0].edited)
            corrections = (folder / "corrections.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(corrections), 2)
            self.assertFalse(summary.exists())

    def test_plain_review_edit_can_replace_paragraph_structure(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            controller = LearningAccessController(project_root, Path(temporary) / "LearningAccess")
            lesson = LessonRecord.create(
                "japanese", "microphone", now=datetime.fromisoformat("2026-06-27T10:00:00+09:00")
            )
            lesson.segments = [
                TranscriptSegment("seg_0001", 1.0, 5.0, "前半。", important=True),
                TranscriptSegment("seg_0002", 8.0, 15.0, "後半。", unclear=True),
            ]
            folder = controller.store.create_lesson(lesson)
            controller.select_lesson(folder)

            controller.replace_transcript_text("全文を一つの文章として書き直しました。")

            self.assertEqual(len(controller.current_lesson.segments), 1)
            merged = controller.current_lesson.segments[0]
            self.assertEqual((merged.start, merged.end), (1.0, 15.0))
            self.assertTrue(merged.important)
            self.assertTrue(merged.unclear)
            correction = json.loads((folder / "corrections.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(correction["segment_id"], "full_transcript")

    def test_plain_review_edit_keeps_each_mark_when_paragraph_count_changes(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            controller = LearningAccessController(project_root, Path(temporary) / "LearningAccess")
            lesson = LessonRecord.create(
                "japanese", "microphone", now=datetime.fromisoformat("2026-06-27T11:00:00+09:00")
            )
            lesson.segments = [
                TranscriptSegment("seg_0001", 0.0, 10.0, "重要な前半。", important=True),
                TranscriptSegment("seg_0002", 10.0, 20.0, "不明瞭な中盤。", unclear=True),
                TranscriptSegment("seg_0003", 20.0, 30.0, "質問の後半。", question=True),
            ]
            folder = controller.store.create_lesson(lesson)
            controller.select_lesson(folder)

            controller.replace_transcript_text("書き直した前半です。\n\n書き直した後半です。")

            edited = controller.current_lesson.segments
            self.assertEqual(len(edited), 2)
            self.assertEqual(sum(segment.important for segment in edited), 1)
            self.assertEqual(sum(segment.unclear for segment in edited), 1)
            self.assertEqual(sum(segment.question for segment in edited), 1)
            self.assertEqual((edited[0].start, edited[-1].end), (0.0, 30.0))

            loaded = controller.store.load(folder)
            self.assertEqual(sum(segment.important for segment in loaded.segments), 1)
            self.assertEqual(sum(segment.unclear for segment in loaded.segments), 1)
            self.assertEqual(sum(segment.question for segment in loaded.segments), 1)

    def test_correction_append_repairs_incomplete_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            store = LessonStore(folder / "records")
            lesson_folder = folder / "lesson"
            lesson_folder.mkdir()
            (lesson_folder / "corrections.jsonl").write_text(
                '{"lesson_id":"old"}\n{"incomplete":',
                encoding="utf-8",
            )

            store.append_correction(
                lesson_folder,
                "new",
                "seg_0001",
                "before",
                "after",
            )

            # Append-only history: the broken line stays as-is, but the
            # new entry lands on its own valid line after it.
            entries = []
            for line in (lesson_folder / "corrections.jsonl").read_text(
                encoding="utf-8"
            ).splitlines():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            self.assertEqual([entry["lesson_id"] for entry in entries], ["old", "new"])

    def test_safe_name_blocks_windows_reserved_device_names(self) -> None:
        self.assertEqual(safe_name("CON"), "_CON")
        self.assertEqual(safe_name("nul.txt"), "_nul.txt")
        self.assertEqual(safe_name("COM1"), "_COM1")

    def test_filters_find_week_and_marks(self) -> None:
        lesson = LessonRecord.create(
            "english", "loopback", now=datetime.fromisoformat("2026-06-23T09:00:00+09:00")
        )
        lesson.segments.append(
            TranscriptSegment("seg_0001", 0, 1, "Important", unclear=True)
        )
        lessons = [(Path("lesson"), lesson)]
        self.assertEqual(len(filter_lessons(lessons, "week", date(2026, 6, 23))), 1)
        self.assertEqual(len(filter_lessons(lessons, "unclear", date(2026, 6, 23))), 1)
        self.assertEqual(len(filter_lessons(lessons, "important", date(2026, 6, 23))), 0)


class AudioPipelineTests(unittest.TestCase):
    def test_vad_emits_timestamped_speech_without_changing_timeline(self) -> None:
        vad = AdaptiveVad(
            pre_roll_seconds=0.2,
            end_silence_seconds=0.3,
            min_chunk_seconds=0.4,
        )
        chunks = []
        cursor = 10.0
        silence = np.zeros(1600, dtype=np.int16)
        speech = np.full(1600, 3000, dtype=np.int16)
        for block in [silence] * 3 + [speech] * 5 + [silence] * 4:
            chunks.extend(vad.process(block, cursor))
            cursor += 0.1
        chunks.extend(vad.flush())

        self.assertEqual(len(chunks), 1)
        self.assertGreaterEqual(chunks[0].start, 10.0)
        self.assertLess(chunks[0].start, 10.31)
        self.assertGreater(chunks[0].end, 10.8)
        self.assertLessEqual(chunks[0].end, 11.2)

    def test_sentence_times_cover_original_chunk(self) -> None:
        sentences = split_text_with_times(
            "Brain drain means skilled workers leave. This can cause shortages.",
            100.0,
            108.0,
        )
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].start, 100.0)
        self.assertEqual(sentences[-1].end, 108.0)

    def test_pcm_converts_to_opus(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        ffmpeg = project_root / "engines" / "ffmpeg" / "ffmpeg.exe"
        if not ffmpeg.exists():
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pcm = root / "recording.pcm"
            opus = root / "audio.opus"
            samples = (np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000) * 2000).astype(np.int16)
            pcm.write_bytes(samples.tobytes())
            convert_pcm_to_opus(ffmpeg, pcm, opus)
            self.assertGreater(opus.stat().st_size, 100)


class ControllerDataTests(unittest.TestCase):
    @staticmethod
    def write_demo_wave(path: Path) -> None:
        samples = np.zeros(16000, dtype=np.int16)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(samples.tobytes())

    def test_demo_lesson_includes_retranscribable_audio(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            controller = LearningAccessController(project_root, Path(temporary) / "LearningAccess")
            with patch.object(
                DemoContent,
                "_synthesize_demo_audio",
                side_effect=self.write_demo_wave,
            ):
                controller.create_demo_lesson()

            folder, lesson = controller.store.list_lessons()[0]
            self.assertEqual(lesson.title, "English S&E")
            self.assertEqual(lesson.audio_file, "demo_english.wav")
            self.assertTrue((folder / lesson.audio_file).is_file())

    def test_existing_demo_without_audio_is_repaired(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "LearningAccess"
            store = LessonStore(data_root)
            lesson = LessonRecord.create(
                "english", "microphone", now=datetime.fromisoformat("2026-06-28T09:00:00+09:00")
            )
            lesson.title = "English S&E"
            lesson.status = "complete"
            lesson.audio_file = ""
            lesson.segments = [
                TranscriptSegment(
                    "seg_0001",
                    742.0,
                    748.0,
                    "Brain drain means skilled workers leave their home country.",
                )
            ]
            folder = store.create_lesson(lesson)

            with patch.object(
                DemoContent,
                "_synthesize_demo_audio",
                side_effect=self.write_demo_wave,
            ):
                controller = LearningAccessController(project_root, data_root)

            repaired = controller.store.load(folder)
            self.assertEqual(repaired.audio_file, "demo_english.wav")
            self.assertTrue((folder / repaired.audio_file).is_file())
            self.assertTrue(repaired.is_demo)
            self.assertEqual(repaired.segments[0].start, 0.0)

    def test_demo_flag_survives_title_and_transcript_edits(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "LearningAccess"
            store = LessonStore(data_root)
            lesson = LessonRecord.create(
                "english", "microphone", now=datetime.fromisoformat("2026-06-28T09:15:00+09:00")
            )
            lesson.title = "Renamed sample"
            lesson.is_demo = True
            lesson.audio_file = ""
            lesson.segments = [
                TranscriptSegment("seg_0001", 0.0, 3.0, "Edited sample text.")
            ]
            folder = store.create_lesson(lesson)

            with patch.object(
                DemoContent,
                "_synthesize_demo_audio",
                side_effect=self.write_demo_wave,
            ):
                controller = LearningAccessController(project_root, data_root)

            repaired = controller.store.load(folder)
            self.assertTrue(repaired.is_demo)
            self.assertEqual(repaired.audio_file, "demo_english.wav")

    def test_select_lesson_rejects_folder_outside_store(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = LearningAccessController(project_root, root / "LearningAccess")
            with self.assertRaises(ValueError):
                controller.select_lesson(root / "outside")

    def test_recording_only_import_preserves_source_and_can_be_deleted(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "2026-06-23_test_lesson.wav"
            samples = (np.sin(np.arange(16000) * 2 * np.pi * 220 / 16000) * 2000).astype(np.int16)
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(samples.tobytes())

            controller = LearningAccessController(project_root, root / "LearningAccess")
            controller.import_audio_async(source, "record_only")
            deadline = time.time() + 20
            while controller.busy and time.time() < deadline:
                time.sleep(0.05)

            self.assertFalse(controller.busy)
            self.assertTrue(source.exists())
            folder = controller.current_folder
            self.assertIsNotNone(folder)
            self.assertTrue((folder / "audio.opus").exists())
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_audio_name"], source.name)
            self.assertEqual(metadata["audio_source"], "imported")

            class FakeJapaneseRecognizer:
                @staticmethod
                def transcribe(samples):
                    return "保存した録音をあとから文字起こししました。"

            with patch(
                "otoweave_app.transcription_service.JapaneseRecognizer",
                return_value=FakeJapaneseRecognizer(),
            ):
                controller.transcribe_current_audio_async("japanese")
                deadline = time.time() + 20
                while controller.busy and time.time() < deadline:
                    time.sleep(0.05)

            self.assertFalse(controller.busy)
            transcript = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(transcript["language_mode"], "japanese")
            self.assertEqual(
                transcript["segments"][0]["text"],
                "保存した録音をあとから文字起こししました。",
            )

            class FakeEnglishRecognizer:
                def __init__(self, model_dir, num_threads=2):
                    self.model_dir = model_dir
                    self.num_threads = num_threads

                @staticmethod
                def transcribe(samples):
                    return "This is an English classroom discussion."

            class FakeLanguageIdentifier:
                def __init__(self, model_dir, num_threads=2):
                    self.model_dir = model_dir
                    self.num_threads = num_threads

                @staticmethod
                def detect(samples):
                    return LanguageDecision(
                        language="english",
                        confidence=0.9,
                        margin=0.8,
                        japanese_score=0.1,
                        english_score=0.9,
                        uncertain=False,
                        reason="",
                    )

            with patch(
                "otoweave_app.transcription_service.JapaneseRecognizer",
                return_value=FakeJapaneseRecognizer(),
            ), patch(
                "otoweave_app.transcription_service.EnglishRecognizer",
                FakeEnglishRecognizer,
            ), patch(
                "otoweave_app.transcription_service.SpeechBrainLanguageIdentifier",
                FakeLanguageIdentifier,
            ):
                controller.transcribe_current_audio_async("mixed")
                deadline = time.time() + 20
                while controller.busy and time.time() < deadline:
                    time.sleep(0.05)

            mixed_transcript = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(mixed_transcript["language_mode"], "mixed")
            self.assertEqual(
                mixed_transcript["segments"][0]["text"],
                "This is an English classroom discussion.",
            )

            segment_id = controller.current_lesson.segments[0].id
            controller.update_segment_text(segment_id, "This is a corrected classroom discussion.")
            corrections = (folder / "corrections.jsonl").read_text(encoding="utf-8").splitlines()
            correction = json.loads(corrections[-1])
            self.assertEqual(correction["before"], "This is an English classroom discussion.")
            self.assertEqual(correction["after"], "This is a corrected classroom discussion.")

            class FakeQwenRecognizer:
                def __init__(self) -> None:
                    self.closed = False

                @staticmethod
                def transcribe(samples):
                    return "日本語と English を一度に文字起こしします。"

                def close(self) -> None:
                    self.closed = True

            fake_qwen = FakeQwenRecognizer()
            with patch(
                "otoweave_app.transcription_service.Qwen3AsrRecognizer",
                return_value=fake_qwen,
            ):
                controller.transcribe_current_audio_async("mixed_qwen17")
                deadline = time.time() + 20
                while controller.busy and time.time() < deadline:
                    time.sleep(0.05)

            qwen_transcript = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(qwen_transcript["language_mode"], "mixed_qwen17")
            self.assertEqual(
                qwen_transcript["segments"][0]["text"],
                "日本語と English を一度に文字起こしします。",
            )
            self.assertTrue(fake_qwen.closed)

            controller.delete_current_lesson()
            self.assertFalse(folder.exists())
            self.assertTrue(source.exists())


class ControllerDiarizationWiringTests(unittest.TestCase):
    """transcribe_current_audio_async / _run_transcribe_existing の
    diarization_speakers 配管が TranscriptionService まで届くこと。"""

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
        return controller

    def test_diarization_speakers_reaches_service_when_given(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._prepared_controller(temporary)
            with patch.object(
                controller.transcription, "transcribe_existing_audio"
            ) as mock_transcribe:
                controller.transcribe_current_audio_async("japanese", diarization_speakers=2)
                deadline = time.time() + 20
                while controller.busy and time.time() < deadline:
                    time.sleep(0.05)

            mock_transcribe.assert_called_once()
            self.assertEqual(
                mock_transcribe.call_args.kwargs["diarization_speakers"], 2
            )

    def test_diarization_speakers_defaults_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._prepared_controller(temporary)
            with patch.object(
                controller.transcription, "transcribe_existing_audio"
            ) as mock_transcribe:
                controller.transcribe_current_audio_async("japanese")
                deadline = time.time() + 20
                while controller.busy and time.time() < deadline:
                    time.sleep(0.05)

            mock_transcribe.assert_called_once()
            self.assertIsNone(
                mock_transcribe.call_args.kwargs["diarization_speakers"]
            )

    def test_invalid_diarization_speakers_raises_before_starting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._prepared_controller(temporary)
            with patch.object(
                controller.transcription, "transcribe_existing_audio"
            ) as mock_transcribe:
                with self.assertRaises(ValueError):
                    controller.transcribe_current_audio_async(
                        "japanese", diarization_speakers=0
                    )

            mock_transcribe.assert_not_called()


class QwenRecognizerLifecycleTests(unittest.TestCase):
    def test_popen_failure_closes_log_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            for relative_path in (
                QWEN3_ASR_SERVER,
                QWEN3_ASR_17_MODEL,
                QWEN3_ASR_17_MMPROJ,
            ):
                path = project_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
            log_dir = project_root / "logs"

            with patch(
                "otoweave_app.asr.subprocess.Popen",
                side_effect=OSError("cannot start"),
            ):
                with self.assertRaises(OSError):
                    Qwen3AsrRecognizer(project_root, log_dir, num_threads=1)

            for log_path in log_dir.iterdir():
                renamed = log_path.with_suffix(log_path.suffix + ".closed")
                log_path.rename(renamed)
                self.assertTrue(renamed.exists())


class LlmLifecycleTests(unittest.TestCase):
    class FakeLlm:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def test_resident_chat_model_is_released_immediately(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            controller = LearningAccessController(
                project_root,
                Path(temporary) / "LearningAccess",
            )
            llm = self.FakeLlm()
            controller.llm_session._llm = llm
            controller.llm_session._model_path = project_root / "models" / "chat.gguf"
            controller.llm_session._chat_messages = [{"role": "user", "content": "test"}]

            released = controller.release_chat_model()

            self.assertTrue(released)
            self.assertTrue(llm.closed)
            self.assertIsNone(controller.llm_session._llm)
            self.assertIsNone(controller.llm_session._model_path)
            self.assertEqual(controller.llm_session._chat_messages, [])

    def test_release_is_deferred_until_chat_finishes(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            controller = LearningAccessController(
                project_root,
                Path(temporary) / "LearningAccess",
            )
            llm = self.FakeLlm()
            controller.llm_session._llm = llm
            controller.llm_session._busy = True

            released = controller.release_chat_model()

            self.assertFalse(released)
            self.assertFalse(llm.closed)
            self.assertTrue(controller.llm_session._release_pending)

            controller.llm_session._finish_task()

            self.assertTrue(llm.closed)
            self.assertIsNone(controller.llm_session._llm)
            self.assertFalse(controller.llm_session._busy)

    def test_chat_history_is_not_overwritten_after_session_reset(self) -> None:
        """Fix 1B: chat_active_folder guard prevents stale response overwriting reset history."""
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            controller = LearningAccessController(
                project_root,
                Path(temporary) / "LearningAccess",
            )
            folder_a = Path(temporary) / "lesson_a"
            folder_b = Path(temporary) / "lesson_b"

            controller.llm_session._chat_active_folder = folder_a
            controller.llm_session._chat_messages = [{"role": "user", "content": "Q from A"}]

            controller.reset_chat()

            self.assertEqual(controller.llm_session._chat_messages, [])
            self.assertIsNone(controller.llm_session._chat_active_folder)

            controller.llm_session._chat_messages = [{"role": "user", "content": "Q from B"}]
            controller.llm_session._chat_active_folder = folder_b

            fake_updated = [{"role": "user", "content": "Q from A"}, {"role": "assistant", "content": "A1"}]
            with controller.llm_session._lock:
                if controller.llm_session._chat_active_folder == folder_a:
                    controller.llm_session._chat_messages = fake_updated

            self.assertEqual(controller.llm_session._chat_messages[0]["content"], "Q from B")

    def test_summary_and_chat_are_blocked_during_audio_processing(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            controller = LearningAccessController(
                project_root,
                Path(temporary) / "LearningAccess",
            )
            lesson = LessonRecord.create("japanese", "microphone")
            controller._state = SessionState.TRANSCRIBING

            with self.assertRaises(RuntimeError):
                controller.summarize_async(
                    lesson,
                    Path(temporary),
                    project_root / "models" / "Qwen3.5-4B-Q4_K_M.gguf",
                )
            with self.assertRaises(RuntimeError):
                controller.chat_async(
                    "質問",
                    Path(temporary),
                    project_root / "models" / "Qwen3.5-2B-Q4_K_M.gguf",
                )


if __name__ == "__main__":
    unittest.main()
