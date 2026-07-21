"""Tests for the dyslexia-support line-spacing fix (spacing2 for wrapped lines).

tkinter Text's spacing3 only affects the gap after a paragraph (newline);
wrapped lines inside a long utterance are spaced by spacing2. These tests
cover the helper that derives spacing2 from the paragraph spacing so the
"標準/ゆったり" toggle visibly changes wrapped-line spacing too.
"""
from __future__ import annotations

import unittest

from otoweave_app.customtkinter_views import wrapped_line_spacing


class WrappedLineSpacingTest(unittest.TestCase):
    def test_is_smaller_than_paragraph_spacing(self):
        # Wrapped-line spacing should be a bit tighter than paragraph spacing.
        for spacing3 in (6, 7, 9, 12):
            with self.subTest(spacing3=spacing3):
                self.assertLess(wrapped_line_spacing(spacing3), spacing3)

    def test_is_roughly_60_percent(self):
        self.assertEqual(wrapped_line_spacing(9), 5)   # body-text default
        self.assertEqual(wrapped_line_spacing(7), 4)   # summary-box default
        self.assertEqual(wrapped_line_spacing(12), 7)  # "ゆったり" body text
        self.assertEqual(wrapped_line_spacing(6), 4)   # "標準" body text
        self.assertEqual(wrapped_line_spacing(8), 5)   # "ゆったり" summary

    def test_comfortable_differs_from_standard(self):
        # otoweave_app maps 標準->6 / ゆったり->12 for the body text and
        # max(4, spacing - 4) (=4/8) for the summary. Both pairs must yield
        # different spacing2 values or the toggle stays invisible on
        # wrapped lines.
        self.assertNotEqual(wrapped_line_spacing(6), wrapped_line_spacing(12))
        self.assertNotEqual(wrapped_line_spacing(4), wrapped_line_spacing(8))

    def test_never_below_one(self):
        self.assertEqual(wrapped_line_spacing(0), 1)
        self.assertEqual(wrapped_line_spacing(1), 1)


if __name__ == "__main__":
    unittest.main()
