"""UDデジタル教科書体のフォント候補追加に関するテスト。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from otoweave_app.display_settings import available_reading_fonts


class UdDigitalKyokashoFontTests(unittest.TestCase):
    def test_ud_digital_kyokasho_is_top_priority(self) -> None:
        fonts = available_reading_fonts(
            [
                "Meiryo UI",
                "Yu Gothic UI",
                "BIZ UDPゴシック",
                "UD デジタル 教科書体 N-R",
            ]
        )
        self.assertEqual(fonts[0], "UD デジタル 教科書体 N-R")
        self.assertEqual(
            fonts,
            (
                "UD デジタル 教科書体 N-R",
                "BIZ UDPゴシック",
                "Yu Gothic UI",
                "Meiryo UI",
            ),
        )

    def test_np_variant_is_recognized(self) -> None:
        fonts = available_reading_fonts(
            ["Yu Gothic UI", "UD デジタル 教科書体 NP-R"]
        )
        self.assertEqual(fonts[0], "UD デジタル 教科書体 NP-R")

    def test_english_family_name_is_recognized(self) -> None:
        fonts = available_reading_fonts(
            ["Yu Gothic UI", "UD Digi Kyokasho N-R"]
        )
        self.assertEqual(fonts[0], "UD Digi Kyokasho N-R")

    def test_n_r_comes_before_other_widths(self) -> None:
        fonts = available_reading_fonts(
            [
                "UD デジタル 教科書体 NP-R",
                "UD デジタル 教科書体 NK-R",
                "UD デジタル 教科書体 N-R",
                "Yu Gothic UI",
            ]
        )
        self.assertEqual(fonts[0], "UD デジタル 教科書体 N-R")
        self.assertEqual(
            fonts[:3],
            (
                "UD デジタル 教科書体 N-R",
                "UD デジタル 教科書体 NK-R",
                "UD デジタル 教科書体 NP-R",
            ),
        )

    def test_bold_variants_are_not_selected(self) -> None:
        # 読み上げ用途では太字（-B）は候補にしない。
        fonts = available_reading_fonts(
            ["UD デジタル 教科書体 N-B", "Yu Gothic UI"]
        )
        self.assertEqual(fonts, ("Yu Gothic UI",))

    def test_fallback_order_without_ud_font(self) -> None:
        fonts = available_reading_fonts(
            ["Meiryo UI", "Yu Gothic UI", "BIZ UDPゴシック", "OpenDyslexic"]
        )
        self.assertEqual(
            fonts,
            ("BIZ UDPゴシック", "OpenDyslexic", "Yu Gothic UI", "Meiryo UI"),
        )

    def test_no_matching_fonts_falls_back_to_default(self) -> None:
        fonts = available_reading_fonts(["Arial", "Times New Roman"])
        self.assertEqual(fonts, ("TkDefaultFont",))


if __name__ == "__main__":
    unittest.main()
