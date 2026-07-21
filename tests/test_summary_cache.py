import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from otoweave_app.models import LessonRecord, TranscriptSegment
from otoweave_app.summary_cache import (
    activate_cached_summary,
    inspect_cached_summary,
    save_cached_summary,
)


class SummaryCacheTests(unittest.TestCase):
    def _lesson(self) -> LessonRecord:
        lesson = LessonRecord.create(
            "japanese",
            "microphone",
            now=datetime.fromisoformat("2026-07-02T12:00:00+09:00"),
        )
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 3.0, "光合成の説明です。")
        ]
        return lesson

    def _template(self) -> dict:
        return {
            "id": "lesson_record",
            "name": "授業の要点",
            "instruction": "授業内容を整理してください。",
            "sections": ["テーマ", "要点"],
            "dictionary": "",
        }

    def test_generated_summary_is_reused_when_inputs_are_unchanged(self) -> None:
        lesson = self._lesson()
        template = self._template()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_cached_summary(
                folder,
                lesson,
                template,
                "保存した要約",
                Path("model.gguf"),
            )

            result = inspect_cached_summary(folder, lesson, template)

        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["text"], "保存した要約")

    def test_integer_and_loaded_float_timestamps_have_same_fingerprint(self) -> None:
        lesson = self._lesson()
        lesson.segments[0].start = 0
        lesson.segments[0].end = 3
        template = self._template()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_cached_summary(
                folder,
                lesson,
                template,
                "保存した要約",
                Path("model.gguf"),
            )
            lesson.segments[0].start = 0.0
            lesson.segments[0].end = 3.0

            result = inspect_cached_summary(folder, lesson, template)

        self.assertEqual(result["status"], "generated")

    def test_transcript_edit_keeps_old_summary_but_marks_it_stale(self) -> None:
        lesson = self._lesson()
        template = self._template()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_cached_summary(
                folder,
                lesson,
                template,
                "以前の要約",
                Path("model.gguf"),
            )
            lesson.segments[0].text = "編集後の文字起こしです。"

            result = inspect_cached_summary(folder, lesson, template)

        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["text"], "以前の要約")

    def test_template_or_dictionary_change_marks_summary_stale(self) -> None:
        lesson = self._lesson()
        template = self._template()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_cached_summary(
                folder,
                lesson,
                template,
                "以前の要約",
                Path("model.gguf"),
            )
            changed = {
                **template,
                "dictionary": "- ディスレクシア: 読み書きの困難",
            }

            result = inspect_cached_summary(folder, lesson, changed)

        self.assertEqual(result["status"], "stale")

    def test_only_current_summary_is_activated_for_chat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            activate_cached_summary(
                folder,
                {"status": "generated", "text": "現在の要約"},
            )
            active = folder / "postprocess" / "school_record.md"
            self.assertEqual(active.read_text(encoding="utf-8"), "現在の要約")

            activate_cached_summary(
                folder,
                {"status": "stale", "text": "古い要約"},
            )

            self.assertFalse(active.exists())


if __name__ == "__main__":
    unittest.main()
