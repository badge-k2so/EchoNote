"""AI要約のゲーティング（4B限定化）のテスト。

方針: 2Bの要約は品質不足（A/B検証）のため、AI要約は
「4Bモデルのファイルが存在 かつ 低メモリ機でない」環境でのみ有効。
使えない環境では生成UIを隠して平易な案内を出す。チャット（2B）と
キャッシュ済み要約の閲覧・読み上げは従来どおり。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from otoweave_app.llm_chat import (
    SUMMARIZE_UNAVAILABLE_LOW_MEMORY,
    SUMMARIZE_UNAVAILABLE_MODEL_MISSING,
    find_chat_model,
    find_summarize_model,
    summarize_availability,
)
from otoweave_app.customtkinter_views import (
    SUMMARY_UNAVAILABLE_MESSAGE,
    summary_controls_visibility,
)
from otoweave_app.otoweave_app import summary_display_text
from customtkinter_app import OtoWeaveApp


_HIGH_RAM = 16 * 1024**3
_LOW_RAM = int(7.9 * 1024**3)  # 実8GB機はWindows上で約7.9GBに見える
# 実16GB機はWindows上で約15.7GBに見える（9B採用の高メモリ閾値15GBを超える最小例）。
_HIGH_RAM_9B = int(15.7 * 1024**3)
# 12GB機: 4Bは使えるが、9B閾値(15GB)未満のため9Bは使わない。
_MID_RAM = 12 * 1024**3


def _patch_ram(ram_bytes: int):
    return patch(
        "otoweave_app.llm_chat._total_physical_ram_bytes",
        return_value=ram_bytes,
    )


# TemporaryDirectory はプロセス終了時にまとめて片付ける
# （Path インスタンスへは属性を追加できないため）。
_TEMP_DIRS: list[tempfile.TemporaryDirectory] = []


def _make_root(with_4b: bool = True, with_2b: bool = True, with_9b: bool = False) -> Path:
    temporary = tempfile.TemporaryDirectory()
    _TEMP_DIRS.append(temporary)
    root = Path(temporary.name)
    (root / "models").mkdir()
    if with_4b:
        (root / "models" / "Qwen3.5-4B-Q4_K_M.gguf").write_bytes(b"x")
    if with_2b:
        (root / "models" / "Qwen3.5-2B-Q4_K_M.gguf").write_bytes(b"x")
    if with_9b:
        (root / "models" / "Qwen3.5-9B-Q4_K_M.gguf").write_bytes(b"x")
    return root


class SummarizeAvailabilityTests(unittest.TestCase):
    def test_available_with_4b_and_high_ram(self) -> None:
        root = _make_root(with_4b=True)
        with _patch_ram(_HIGH_RAM):
            available, reason = summarize_availability(root)
        self.assertTrue(available)
        self.assertEqual(reason, "")

    def test_unavailable_without_4b_even_if_2b_exists(self) -> None:
        root = _make_root(with_4b=False, with_2b=True)
        with _patch_ram(_HIGH_RAM):
            available, reason = summarize_availability(root)
        self.assertFalse(available)
        self.assertEqual(reason, SUMMARIZE_UNAVAILABLE_MODEL_MISSING)

    def test_unavailable_on_low_memory_machine(self) -> None:
        root = _make_root(with_4b=True)
        with _patch_ram(_LOW_RAM):
            available, reason = summarize_availability(root)
        self.assertFalse(available)
        self.assertEqual(reason, SUMMARIZE_UNAVAILABLE_LOW_MEMORY)

    def test_unavailable_when_ram_query_fails(self) -> None:
        # RAM取得失敗（0）は安全側＝低メモリ扱いで非対応にする。
        root = _make_root(with_4b=True)
        with _patch_ram(0):
            available, reason = summarize_availability(root)
        self.assertFalse(available)
        self.assertEqual(reason, SUMMARIZE_UNAVAILABLE_LOW_MEMORY)

    def test_find_summarize_model_never_falls_back_to_2b(self) -> None:
        # 4Bが無ければ高RAM機でも None（2Bで要約は生成しない）。
        root = _make_root(with_4b=False, with_2b=True)
        with _patch_ram(_HIGH_RAM):
            self.assertIsNone(find_summarize_model(root))

    def test_find_summarize_model_returns_4b_when_available(self) -> None:
        root = _make_root(with_4b=True)
        with _patch_ram(_HIGH_RAM):
            chosen = find_summarize_model(root)
        self.assertIsNotNone(chosen)
        self.assertIn("4B", chosen.name)

    def test_chat_model_selection_is_unaffected(self) -> None:
        # チャットは従来どおり: 低RAM・Lite構成でも2Bが選ばれる。
        root = _make_root(with_4b=False, with_2b=True)
        for ram in (_LOW_RAM, _HIGH_RAM):
            with _patch_ram(ram):
                chat_model = find_chat_model(root)
            self.assertIsNotNone(chat_model)
            self.assertIn("2B", chat_model.name)


class SummarizeModel9BTests(unittest.TestCase):
    """上位棚(9B)の準備実装。

    4B/9Bベンチマークの結果次第で採用されるため、現時点では「9Bファイルが
    存在し、かつ高メモリ機(~15GB以上)」のときだけ9Bが選ばれる。9Bが無い/
    低メモリ/中間RAMのときは既存どおり4Bにフォールスルーする（挙動不変）。
    """

    def test_high_ram_with_9b_file_selects_9b(self) -> None:
        root = _make_root(with_4b=True, with_9b=True)
        with _patch_ram(_HIGH_RAM_9B):
            chosen = find_summarize_model(root)
            available, reason = summarize_availability(root)
        self.assertIsNotNone(chosen)
        self.assertIn("9B", chosen.name)
        self.assertTrue(available)
        self.assertEqual(reason, "")

    def test_high_ram_without_9b_file_falls_back_to_4b(self) -> None:
        root = _make_root(with_4b=True, with_9b=False)
        with _patch_ram(_HIGH_RAM_9B):
            chosen = find_summarize_model(root)
            available, reason = summarize_availability(root)
        self.assertIsNotNone(chosen)
        self.assertIn("4B", chosen.name)
        self.assertTrue(available)
        self.assertEqual(reason, "")

    def test_mid_ram_with_9b_file_still_uses_4b(self) -> None:
        # 15GB未満では9Bファイルがあっても使わない。
        root = _make_root(with_4b=True, with_9b=True)
        with _patch_ram(_MID_RAM):
            chosen = find_summarize_model(root)
        self.assertIsNotNone(chosen)
        self.assertIn("4B", chosen.name)

    def test_low_ram_with_9b_file_is_still_unavailable(self) -> None:
        # 低メモリ機は9B/4Bどちらのファイルがあっても要約非対応のまま。
        root = _make_root(with_4b=True, with_9b=True)
        with _patch_ram(_LOW_RAM):
            chosen = find_summarize_model(root)
            available, reason = summarize_availability(root)
        self.assertIsNone(chosen)
        self.assertFalse(available)
        self.assertEqual(reason, SUMMARIZE_UNAVAILABLE_LOW_MEMORY)

    def test_high_ram_with_only_9b_file_is_available(self) -> None:
        # 4Bが無くても、高メモリ機で9Bファイルがあれば要約可能と判定する。
        root = _make_root(with_4b=False, with_9b=True)
        with _patch_ram(_HIGH_RAM_9B):
            chosen = find_summarize_model(root)
            available, reason = summarize_availability(root)
        self.assertIsNotNone(chosen)
        self.assertIn("9B", chosen.name)
        self.assertTrue(available)
        self.assertEqual(reason, "")


class SummaryControlsVisibilityTests(unittest.TestCase):
    """可用性 → 右パネル表示状態の純ロジック。"""

    def test_available_shows_generation_controls(self) -> None:
        visibility = summary_controls_visibility(True)
        self.assertTrue(visibility["summarize_button"])
        self.assertTrue(visibility["template_menu"])
        self.assertTrue(visibility["manage_templates_button"])
        self.assertFalse(visibility["unavailable_notice"])

    def test_unavailable_hides_generation_and_shows_notice(self) -> None:
        visibility = summary_controls_visibility(False)
        self.assertFalse(visibility["summarize_button"])
        self.assertFalse(visibility["template_menu"])
        self.assertFalse(visibility["manage_templates_button"])
        self.assertTrue(visibility["unavailable_notice"])

    def test_cached_summary_viewing_and_tts_stay_available(self) -> None:
        # 生成不可でも、要約本文の表示と読み上げボタンは残す。
        visibility = summary_controls_visibility(False)
        self.assertTrue(visibility["summary_box"])
        self.assertTrue(visibility["speak_summary_button"])

    def test_notice_message_is_plain_japanese(self) -> None:
        # 技術用語・モデル名を生徒に見せない。
        for banned in ("Qwen", "4B", "2B", "GB", "RAM", "メモリ", "モデル"):
            self.assertNotIn(banned, SUMMARY_UNAVAILABLE_MESSAGE)
        self.assertIn("じゅんび中", SUMMARY_UNAVAILABLE_MESSAGE)


class SummaryDisplayTextTests(unittest.TestCase):
    """キャッシュ済み要約は、生成が使えない環境でも閲覧できる。"""

    def test_generated_summary_is_shown_unchanged_when_unavailable(self) -> None:
        text = "## 授業の要点\n- 光合成のしくみ"
        self.assertEqual(
            summary_display_text("generated", text, summarize_available=False),
            text,
        )

    def test_missing_summary_shows_preparing_notice_when_unavailable(self) -> None:
        shown = summary_display_text("missing", "", summarize_available=False)
        self.assertIn("じゅんび中", shown)
        self.assertNotIn("要約を作成", shown)

    def test_missing_summary_prompts_button_when_available(self) -> None:
        shown = summary_display_text("missing", "", summarize_available=True)
        self.assertIn("「要約を作成」を押してください", shown)

    def test_stale_summary_stays_readable_without_update_prompt(self) -> None:
        cached = "以前の要約本文"
        shown = summary_display_text("stale", cached, summarize_available=False)
        self.assertIn(cached, shown)
        self.assertNotIn("更新する", shown)

    def test_stale_summary_prompts_update_when_available(self) -> None:
        cached = "以前の要約本文"
        shown = summary_display_text("stale", cached, summarize_available=True)
        self.assertIn(cached, shown)
        self.assertIn("「更新する」を押してください", shown)


class OnSummarizeGuardTests(unittest.TestCase):
    """生成UI非表示でも、万一ハンドラが呼ばれたら安全に何もしない。"""

    def test_on_summarize_is_noop_when_unavailable(self) -> None:
        app = SimpleNamespace(
            _summarize_available=False,
            _active_folder=Path("/tmp/lesson"),
            controller=SimpleNamespace(
                store=SimpleNamespace(load=Mock()),
                summarize_async=Mock(),
            ),
            right_pane=SimpleNamespace(set_summarizing=Mock()),
            main_pane=SimpleNamespace(
                set_transcribing_blocked=Mock(),
                update_status=Mock(),
            ),
        )
        with patch(
            "otoweave_app.otoweave_app.messagebox"
        ) as fake_messagebox:
            OtoWeaveApp._on_summarize(app)

        fake_messagebox.showinfo.assert_called_once()
        message = fake_messagebox.showinfo.call_args.args[1]
        self.assertIn("じゅんび中", message)
        app.controller.store.load.assert_not_called()
        app.controller.summarize_async.assert_not_called()
        app.right_pane.set_summarizing.assert_not_called()


if __name__ == "__main__":
    unittest.main()
