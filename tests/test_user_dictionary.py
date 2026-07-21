import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from otoweave_app.asr import RecognizedSentence
from otoweave_app.controller import LearningAccessController
from otoweave_app.models import LessonRecord
from otoweave_app.user_dictionary import (
    correct_text,
    dictionary_path,
    glossary_prompt,
    load_dictionary,
    normalize_entry,
    save_dictionary,
)


class UserDictionaryTests(unittest.TestCase):
    def _entry(self) -> dict:
        return normalize_entry(
            {
                "id": "dyslexia",
                "term": "ディスレクシア",
                "reading": "でぃすれくしあ",
                "aliases": ["ディスレキシア", "ディスレクシヤ"],
                "description": "読み書きに特有の困難がある状態",
                "category": "読み書き支援",
            }
        )

    def test_dictionary_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "user_dictionary.json"
            save_dictionary(path, [self._entry()])

            restored = load_dictionary(path)

        self.assertEqual(restored[0]["term"], "ディスレクシア")
        self.assertEqual(restored[0]["aliases"], ["ディスレキシア", "ディスレクシヤ"])

    def test_registered_alias_corrects_recognized_text(self) -> None:
        corrected = correct_text(
            "ディスレキシアへの支援を考えます。",
            [self._entry()],
        )

        self.assertEqual(corrected, "ディスレクシアへの支援を考えます。")

    def test_english_alias_does_not_replace_inside_another_word(self) -> None:
        entry = normalize_entry(
            {
                "id": "ict",
                "term": "ICT",
                "aliases": ["IT"],
                "category": "学習用語",
            }
        )

        corrected = correct_text("IT is useful, but WITHIN stays.", [entry])

        self.assertEqual(corrected, "ICT is useful, but WITHIN stays.")

    def test_one_correction_does_not_trigger_another_dictionary_entry(self) -> None:
        entries = [
            normalize_entry(
                {
                    "id": "first",
                    "term": "正式語A",
                    "aliases": ["誤認語"],
                    "category": "学習用語",
                }
            ),
            normalize_entry(
                {
                    "id": "second",
                    "term": "正式語B",
                    "aliases": ["正式語A"],
                    "category": "学習用語",
                }
            ),
        ]

        self.assertEqual(correct_text("誤認語", entries), "正式語A")

    def test_glossary_is_available_to_summary_prompt(self) -> None:
        prompt = glossary_prompt([self._entry()])

        self.assertIn("ディスレクシア", prompt)
        self.assertIn("読み書きに特有の困難", prompt)

    def test_controller_applies_dictionary_to_live_asr_result(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "LearningAccess"
            save_dictionary(dictionary_path(data_root), [self._entry()])
            controller = LearningAccessController(project_root, data_root)
            lesson = LessonRecord.create(
                "japanese",
                "microphone",
                now=datetime.fromisoformat("2026-07-02T10:00:00+09:00"),
            )
            folder = controller.store.create_lesson(lesson)
            controller.select_lesson(folder)

            controller._on_sentences(
                [RecognizedSentence(0.0, 2.0, "ディスレキシアの説明")]
            )

            self.assertEqual(
                controller.current_lesson.segments[0].text,
                "ディスレクシアの説明",
            )


if __name__ == "__main__":
    unittest.main()
