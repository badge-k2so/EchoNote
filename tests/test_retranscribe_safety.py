import queue as queue_module
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from otoweave_app.models import LessonRecord, TranscriptSegment
from otoweave_app.segment_editing import rename_speaker, transfer_marks
from otoweave_app.storage import LessonStore
from otoweave_app.transcription_service import TranscriptionService


class EmptyRetranscribeGuardTests(unittest.TestCase):
    """再文字起こしが空の結果を返しても既存の文字起こしを失わないこと。"""

    def _make_lesson(self, store: LessonStore) -> tuple[Path, LessonRecord]:
        lesson = LessonRecord.create(
            "english",
            "microphone",
            now=datetime.fromisoformat("2026-07-01T09:00:00+09:00"),
        )
        lesson.status = "complete"
        lesson.audio_file = "recording.pcm"
        lesson.segments = [
            TranscriptSegment(
                "seg_0001",
                0.0,
                3.0,
                "手で直した大事な文字起こしです。",
                important=True,
                important_at="2026-07-01T10:00:00+09:00",
            ),
            TranscriptSegment(
                "seg_0002",
                3.0,
                6.0,
                "ここは質問した場所です。",
                question=True,
                question_at="2026-07-01T10:01:00+09:00",
            ),
        ]
        folder = store.create_lesson(lesson)
        (folder / "recording.pcm").write_bytes(b"\x00\x00" * 1600)
        return folder, lesson

    def test_empty_result_keeps_existing_segments_and_reports_error(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            folder, lesson = self._make_lesson(store)
            original_texts = [segment.text for segment in lesson.segments]

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

            with patch.object(
                TranscriptionService,
                "_transcribe_single_language_pcm",
                return_value=[],
            ), patch(
                "otoweave_app.transcription_service.JapaneseRecognizer",
                return_value=object(),
            ):
                service.transcribe_existing_audio(
                    folder / "recording.pcm", "japanese", folder, lesson
                )

            # 既存の文字起こしとマークが残っていること
            self.assertEqual(
                [segment.text for segment in lesson.segments], original_texts
            )
            self.assertTrue(lesson.segments[0].important)
            self.assertTrue(lesson.segments[1].question)
            # 失敗扱い: language_mode / status は書き換えない
            self.assertEqual(lesson.language_mode, "english")
            self.assertEqual(lesson.status, "complete")
            # ディスク上の transcript.json も守られていること
            stored = store.load(folder)
            self.assertEqual(
                [segment.text for segment in stored.segments], original_texts
            )
            # 成功時の選択切り替えは呼ばれないこと
            self.assertEqual(ready_calls, [])
            # 作業用PCMが片付けられていること
            self.assertFalse((folder / "transcription_input.pcm").exists())

            kinds = []
            payloads = {}
            while not events.empty():
                kind, payload = events.get_nowait()
                kinds.append(kind)
                payloads.setdefault(kind, payload)
            # 「処理中」のまま残らないよう transcription_finished が出ること
            self.assertIn("transcription_finished", kinds)
            self.assertIn("error", kinds)
            self.assertIn(
                "元の文字起こしはそのまま残しています", payloads["error"]
            )
            # 内部用語（例外クラス名やモデル名）を生徒に見せないこと
            self.assertNotIn("Exception", payloads["error"])
            self.assertNotIn("Recognizer", payloads["error"])

    def test_non_empty_result_still_replaces_segments(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            folder, lesson = self._make_lesson(store)

            new_segments = [
                TranscriptSegment("seg_0001", 0.0, 3.0, "新しい文字起こしです。"),
                TranscriptSegment("seg_0002", 4.0, 6.0, "二つ目の文です。"),
            ]
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

            with patch.object(
                TranscriptionService,
                "_transcribe_single_language_pcm",
                return_value=new_segments,
            ), patch(
                "otoweave_app.transcription_service.JapaneseRecognizer",
                return_value=object(),
            ):
                service.transcribe_existing_audio(
                    folder / "recording.pcm", "japanese", folder, lesson
                )

            self.assertEqual(lesson.segments, new_segments)
            self.assertEqual(lesson.language_mode, "japanese")
            self.assertEqual(lesson.status, "complete")
            self.assertEqual(len(ready_calls), 1)
            # マーク（★と?）が新しいセグメントへ引き継がれること
            self.assertTrue(lesson.segments[0].important)
            self.assertTrue(lesson.segments[1].question)

            kinds = [events.get_nowait()[0] for _ in range(events.qsize())]
            self.assertIn("transcription_finished", kinds)
            self.assertNotIn("error", kinds)


class TransferMarksQuestionTests(unittest.TestCase):
    """質問マークだけの区間も再文字起こしで引き継がれること。"""

    def test_question_only_mark_is_transferred(self) -> None:
        previous = [
            TranscriptSegment(
                "seg_0001",
                0.0,
                4.0,
                "先生に質問した場所です。",
                question=True,
                question_at="2026-07-01T10:05:00+09:00",
            )
        ]
        new_segments = [
            TranscriptSegment("seg_0001", 0.0, 2.0, "前半の文です。"),
            TranscriptSegment("seg_0002", 2.0, 4.0, "後半の文です。"),
        ]

        transfer_marks(previous, new_segments)

        marked = [segment for segment in new_segments if segment.question]
        self.assertEqual(len(marked), 1)
        self.assertEqual(marked[0].question_at, "2026-07-01T10:05:00+09:00")
        self.assertFalse(any(segment.important for segment in new_segments))
        self.assertFalse(any(segment.unclear for segment in new_segments))

    def test_all_mark_kinds_are_transferred_together(self) -> None:
        previous = [
            TranscriptSegment(
                "seg_0001",
                0.0,
                2.0,
                "大事で質問もした場所です。",
                important=True,
                question=True,
                important_at="2026-07-01T10:06:00+09:00",
                question_at="2026-07-01T10:07:00+09:00",
            )
        ]
        new_segments = [TranscriptSegment("seg_0001", 0.0, 2.0, "新しい文です。")]

        transfer_marks(previous, new_segments)

        self.assertTrue(new_segments[0].important)
        self.assertTrue(new_segments[0].question)
        self.assertEqual(new_segments[0].important_at, "2026-07-01T10:06:00+09:00")
        self.assertEqual(new_segments[0].question_at, "2026-07-01T10:07:00+09:00")

    def test_unmarked_segments_are_skipped(self) -> None:
        previous = [TranscriptSegment("seg_0001", 0.0, 2.0, "マークなしの文です。")]
        new_segments = [TranscriptSegment("seg_0001", 0.0, 2.0, "新しい文です。")]

        transfer_marks(previous, new_segments)

        self.assertFalse(new_segments[0].important)
        self.assertFalse(new_segments[0].unclear)
        self.assertFalse(new_segments[0].question)


class RenameSpeakerTests(unittest.TestCase):
    """rename_speaker(): renames every segment matching `old`, in place."""

    def test_matching_segments_are_renamed_and_marked_edited(self) -> None:
        segments = [
            TranscriptSegment("seg_0001", 0.0, 2.0, "おはよう", speaker="話者1"),
            TranscriptSegment("seg_0002", 2.0, 4.0, "どうも", speaker="話者2"),
            TranscriptSegment("seg_0003", 4.0, 6.0, "またね", speaker="話者1"),
        ]

        changed = rename_speaker(segments, "話者1", "先生")

        self.assertEqual(changed, 2)
        self.assertEqual(segments[0].speaker, "先生")
        self.assertEqual(segments[2].speaker, "先生")
        self.assertTrue(segments[0].edited)
        self.assertTrue(segments[2].edited)
        # The unmatched speaker and its edited flag are left untouched.
        self.assertEqual(segments[1].speaker, "話者2")
        self.assertFalse(segments[1].edited)

    def test_no_matching_segments_returns_zero(self) -> None:
        segments = [
            TranscriptSegment("seg_0001", 0.0, 2.0, "おはよう", speaker="話者2"),
        ]

        changed = rename_speaker(segments, "話者1", "先生")

        self.assertEqual(changed, 0)
        self.assertEqual(segments[0].speaker, "話者2")
        self.assertFalse(segments[0].edited)

    def test_renaming_to_the_same_name_is_a_no_op(self) -> None:
        segments = [
            TranscriptSegment("seg_0001", 0.0, 2.0, "おはよう", speaker="話者1"),
        ]

        changed = rename_speaker(segments, "話者1", "話者1")

        self.assertEqual(changed, 0)
        self.assertFalse(segments[0].edited)

    def test_empty_segments_list_is_handled(self) -> None:
        self.assertEqual(rename_speaker([], "話者1", "先生"), 0)


if __name__ == "__main__":
    unittest.main()
