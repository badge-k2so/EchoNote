import tempfile
import unittest
from pathlib import Path

from otoweave_app.summary_templates import (
    load_templates,
    normalize_template,
    save_custom_templates,
    template_by_id,
)


class SummaryTemplateTests(unittest.TestCase):
    def test_defaults_include_learning_templates(self) -> None:
        templates = load_templates(Path("missing-summary-templates.json"))

        ids = {value["id"] for value in templates}
        self.assertIn("lesson_record", ids)
        self.assertIn("easy_japanese", ids)
        self.assertIn("exam_review", ids)

    def test_custom_template_round_trip(self) -> None:
        custom = {
            "id": "custom_weekly",
            "name": "今週の復習",
            "instruction": "一週間の学習内容として整理してください。",
            "sections": ["できたこと", "次に覚えること"],
            "builtin": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary_templates.json"

            save_custom_templates(path, [custom])
            loaded = load_templates(path)

        restored = template_by_id(loaded, "custom_weekly")
        self.assertEqual(restored["name"], "今週の復習")
        self.assertFalse(restored["builtin"])

    def test_invalid_template_requires_name_and_instruction(self) -> None:
        with self.assertRaisesRegex(ValueError, "必要"):
            normalize_template(
                {
                    "id": "custom_empty",
                    "name": "",
                    "instruction": "",
                    "sections": [],
                },
                builtin=False,
            )


if __name__ == "__main__":
    unittest.main()
