import datetime as dt
import tempfile
import unittest
from pathlib import Path

from scripts.production.record_filename import build_filename_suggestion, extract_recorded_date, extract_title


class RecordFilenameTests(unittest.TestCase):
    def test_date_from_audio_filename_and_title_from_transcript(self) -> None:
        transcript = """
===== chunk_000 =====
よろしくお願いします。
今日は読み書きに困難のある子どもの支援について話します。
氷山モデルと合理的配慮について確認します。
"""
        result = build_filename_suggestion(
            Path("2025-11-22 15_19_13.ogg"), transcript
        )
        self.assertEqual(result["recorded_date"], "2025-11-22")
        self.assertEqual(result["date_source"], "audio_filename")
        self.assertIn("読み書き", result["title"])
        self.assertEqual(result["suggested_audio_filename"][-4:], ".ogg")
        self.assertTrue(result["requires_user_confirmation"])

    def test_provided_date_takes_priority(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            date, source = extract_recorded_date(
                Path(audio.name), "2026-06-23T09:30:00+09:00"
            )
        self.assertEqual(date, dt.date(2026, 6, 23))
        self.assertEqual(source, "provided_recorded_at")

    def test_empty_transcript_has_safe_fallback(self) -> None:
        title, source = extract_title("===== chunk_000 =====\nはい。\n")
        self.assertEqual(title, "録音")
        self.assertEqual(source, "fallback_no_meaningful_text")

    def test_windows_invalid_characters_are_removed(self) -> None:
        result = build_filename_suggestion(
            Path("20260623_audio.wav"), '授業で「読む/書く？」について確認します。'
        )
        for invalid in '<>:"/\\|?*':
            self.assertNotIn(invalid, result["suggested_audio_filename"])

    def test_descriptive_audio_filename_label_is_preserved(self) -> None:
        result = build_filename_suggestion(
            Path("20260522ひかり学園学習サポート面談.ogg"),
            "冒頭にはあいさつと自己紹介が含まれます。",
        )
        self.assertEqual(result["title"], "ひかり学園学習サポート面談")
        self.assertEqual(result["title_source"], "audio_filename_label")

    def test_conversational_prefix_is_removed_from_topic_term(self) -> None:
        title, source = extract_title(
            "アメリカではボランティア活動があります。多分中学受験も話題です。"
        )
        self.assertEqual(title, "ボランティア活動・中学受験")
        self.assertEqual(source, "transcript_topic_terms")


if __name__ == "__main__":
    unittest.main()
