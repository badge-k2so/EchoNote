"""macOSらしいUI対応（ファイルダイアログ・フォント・メニューバー・TTSカタログ）のテスト。

開発機はWindowsのため、macOS固有パスは sys.platform / os.name をモックして
検証する。Windows側の挙動（既定値・分岐先）が変わっていないことも合わせて
確認する。
"""

from __future__ import annotations

import tkinter
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from otoweave_app import model_catalog
from otoweave_app import otoweave_app as otoweave_app_module
from otoweave_app.display_settings import (
    _default_font_family,
    available_reading_fonts,
)
from otoweave_app.customtkinter_views import (
    _default_ui_font,
    _undo_key_hint,
)
from otoweave_app.otoweave_app import AUDIO_EXTENSIONS, OtoWeaveApp


# ---------------------------------------------------------------------------
# 2. フォントフォールバック
# ---------------------------------------------------------------------------
class FontFallbackMacTests(unittest.TestCase):
    def test_hiragino_sans_is_recognized_when_installed(self) -> None:
        fonts = available_reading_fonts(["Hiragino Sans", "Hiragino Kaku Gothic ProN"])
        self.assertEqual(fonts[0], "Hiragino Sans")
        self.assertEqual(fonts[1], "Hiragino Kaku Gothic ProN")

    def test_hiragino_kaku_gothic_pron_is_recognized_alone(self) -> None:
        fonts = available_reading_fonts(["Hiragino Kaku Gothic ProN"])
        self.assertEqual(fonts, ("Hiragino Kaku Gothic ProN",))

    def test_ud_font_still_wins_over_hiragino(self) -> None:
        # アクセシビリティ用のUDフォントは、Mac上にあってもなお最優先。
        fonts = available_reading_fonts(
            ["Hiragino Sans", "UD デジタル 教科書体 N-R"]
        )
        self.assertEqual(fonts[0], "UD デジタル 教科書体 N-R")

    def test_windows_only_fonts_are_unaffected_by_hiragino_addition(self) -> None:
        # Windowsの候補・順序が変わっていないことの回帰確認。
        fonts = available_reading_fonts(
            ["Meiryo UI", "Yu Gothic UI", "BIZ UDPゴシック", "OpenDyslexic"]
        )
        self.assertEqual(
            fonts,
            ("BIZ UDPゴシック", "OpenDyslexic", "Yu Gothic UI", "Meiryo UI"),
        )

    def test_no_matching_fonts_falls_back_to_system_default(self) -> None:
        fonts = available_reading_fonts(["Arial", "Times New Roman"])
        self.assertEqual(fonts, ("TkDefaultFont",))

    def test_default_font_family_is_hiragino_on_darwin(self) -> None:
        self.assertEqual(_default_font_family("darwin"), "Hiragino Sans")

    def test_default_font_family_is_unchanged_on_windows(self) -> None:
        self.assertEqual(_default_font_family("win32"), "Yu Gothic UI")

    def test_default_font_family_uses_real_platform_by_default(self) -> None:
        # 引数省略時は実行環境の sys.platform を見る（Windows開発機では
        # "Yu Gothic UI" のまま）。
        self.assertEqual(_default_font_family(), "Yu Gothic UI")


class UiFontConstantsTests(unittest.TestCase):
    def test_ui_font_is_hiragino_on_darwin(self) -> None:
        self.assertEqual(_default_ui_font("darwin"), "Hiragino Sans")

    def test_ui_font_is_meiryo_elsewhere(self) -> None:
        self.assertEqual(_default_ui_font("win32"), "Meiryo")
        self.assertEqual(_default_ui_font("linux"), "Meiryo")

    def test_undo_key_hint_is_command_on_darwin(self) -> None:
        self.assertEqual(_undo_key_hint("darwin"), "Cmd+Z")

    def test_undo_key_hint_is_control_elsewhere(self) -> None:
        self.assertEqual(_undo_key_hint("win32"), "Ctrl+Z")


# ---------------------------------------------------------------------------
# 1. Macファイルダイアログ
# ---------------------------------------------------------------------------
def _make_fake_app(project_root: Path = Path("C:/proj")) -> SimpleNamespace:
    app = SimpleNamespace()
    app.controller = SimpleNamespace(project_root=project_root, busy=False)
    app._file_dialog_active = False
    app._file_dialog_process = None
    app.main_pane = SimpleNamespace(update_status=Mock())
    app.route_to = Mock()
    app.run_background = Mock()
    app.after = lambda _delay, fn: fn()
    app._start_selected_audio_import = Mock()
    app._show_friendly_error = Mock()
    app._open_audio_file_dialog_native = Mock()
    app._finish_audio_file_dialog = Mock()
    return app


class FileDialogBranchTests(unittest.TestCase):
    def test_windows_uses_powershell_helper_branch(self) -> None:
        app = _make_fake_app()
        with patch("otoweave_app.otoweave_app.os.name", "nt"):
            OtoWeaveApp._open_audio_file_dialog(app)
        app._open_audio_file_dialog_native.assert_not_called()
        app.run_background.assert_called_once()

    def test_macos_uses_native_tkinter_dialog_branch(self) -> None:
        app = _make_fake_app()
        with patch("otoweave_app.otoweave_app.os.name", "posix"):
            OtoWeaveApp._open_audio_file_dialog(app)
        app._open_audio_file_dialog_native.assert_called_once()
        app.run_background.assert_not_called()

    def test_linux_also_uses_native_dialog_branch(self) -> None:
        # 仕様上 macOS/Linux の両方が対象。
        app = _make_fake_app()
        with patch("otoweave_app.otoweave_app.os.name", "posix"):
            OtoWeaveApp._open_audio_file_dialog(app)
        app._open_audio_file_dialog_native.assert_called_once()


class NativeFileDialogTests(unittest.TestCase):
    def test_selected_file_starts_import_with_same_8_extensions(self) -> None:
        app = _make_fake_app()
        with patch(
            "otoweave_app.otoweave_app.filedialog.askopenfilename",
            return_value="C:/proj/audio.wav",
        ) as mock_ask:
            OtoWeaveApp._open_audio_file_dialog_native(app)
        mock_ask.assert_called_once()
        _, kwargs = mock_ask.call_args
        pattern = kwargs["filetypes"][0][1]
        for ext in AUDIO_EXTENSIONS:
            self.assertIn(f"*{ext}", pattern)
        self.assertEqual(len(AUDIO_EXTENSIONS), 8)
        app._start_selected_audio_import.assert_called_once_with(
            Path("C:/proj/audio.wav")
        )
        self.assertFalse(app._file_dialog_active)

    def test_cancelled_dialog_returns_to_notes(self) -> None:
        app = _make_fake_app()
        with patch(
            "otoweave_app.otoweave_app.filedialog.askopenfilename",
            return_value="",
        ):
            OtoWeaveApp._open_audio_file_dialog_native(app)
        app.route_to.assert_called_once_with("notes")
        app._start_selected_audio_import.assert_not_called()
        self.assertFalse(app._file_dialog_active)

    def test_dialog_error_shows_friendly_message(self) -> None:
        app = _make_fake_app()
        with patch(
            "otoweave_app.otoweave_app.filedialog.askopenfilename",
            side_effect=RuntimeError("boom"),
        ):
            OtoWeaveApp._open_audio_file_dialog_native(app)
        app._show_friendly_error.assert_called_once()
        app.route_to.assert_called_once_with("notes")
        self.assertFalse(app._file_dialog_active)


# ---------------------------------------------------------------------------
# 3. Macネイティブメニューバー
# ---------------------------------------------------------------------------
class MacMenuBarTests(unittest.TestCase):
    def test_menu_bar_is_only_set_up_on_darwin(self) -> None:
        with patch.object(OtoWeaveApp, "_setup_mac_menu_bar", Mock()) as mocked:
            app = OtoWeaveApp(None)
            try:
                self.assertEqual(mocked.call_count, 0)
            finally:
                app.destroy()

    def test_menu_bar_is_set_up_when_platform_is_darwin(self) -> None:
        # customtkinter自体も内部でsys.platformを見てカーソル指定などを
        # 切り替えるため、実機がWindowsのままだと本物のsys.platformを
        # "darwin"に差し替えると壊れる（例: Aqua専用カーソル名の指定）。
        # otoweave_app モジュール内の `sys` 参照だけを差し替えることで、
        # customtkinter 側の実際の実行環境判定には影響を与えない。
        with patch.object(otoweave_app_module, "sys") as mock_sys:
            mock_sys.platform = "darwin"
            with patch.object(OtoWeaveApp, "_setup_mac_menu_bar", Mock()) as mocked:
                app = OtoWeaveApp(None)
                try:
                    self.assertEqual(mocked.call_count, 1)
                finally:
                    app.destroy()

    def test_setup_mac_menu_bar_builds_apple_menu_without_error(self) -> None:
        root = tkinter.Tk()
        try:
            root.route_to = Mock()
            root._show_context = Mock()
            root._show_about = Mock()
            root._on_close = Mock()
            OtoWeaveApp._setup_mac_menu_bar(root)
            menu_path = root["menu"]
            self.assertTrue(menu_path)
        finally:
            root.destroy()

    def test_show_about_routes_to_license_screen(self) -> None:
        app = SimpleNamespace()
        app.route_to = Mock()
        app._show_context = Mock()
        OtoWeaveApp._show_about(app)
        app.route_to.assert_called_once_with("settings")
        app._show_context.assert_called_once_with("settings", "モデルとライセンス")


# ---------------------------------------------------------------------------
# 4. model_catalog.py の Mac TTS エントリ
# ---------------------------------------------------------------------------
class MacTtsCatalogTests(unittest.TestCase):
    def _entry(self, root: Path = Path.cwd()) -> model_catalog.ModelDisclosure:
        return next(
            m
            for m in model_catalog.model_disclosures(root)
            if m.key == "macos-tts-kyoko"
        )

    def test_entry_metadata(self) -> None:
        entry = self._entry()
        self.assertEqual(entry.name, "macOS標準音声 Kyoko（say）")
        self.assertEqual(entry.purpose, "読み上げ（TTS）")
        self.assertEqual(entry.license_name, "macOS標準機能（OSに同梱）")
        self.assertTrue(entry.source_url.startswith("https://developer.apple.com/"))
        self.assertFalse(entry.required)

    def test_available_true_on_darwin(self) -> None:
        with patch("otoweave_app.model_catalog.sys.platform", "darwin"):
            self.assertTrue(self._entry().available)

    def test_available_false_on_windows(self) -> None:
        with patch("otoweave_app.model_catalog.sys.platform", "win32"):
            self.assertFalse(self._entry().available)

    def test_windows_haruka_entry_still_present_and_unaffected(self) -> None:
        haruka = next(
            m
            for m in model_catalog.model_disclosures(Path.cwd())
            if m.key == "windows-tts-haruka"
        )
        self.assertEqual(haruka.name, "Microsoft Haruka（System.Speech）")


# ---------------------------------------------------------------------------
# 5. OtoWeaveApp(None) スモークテスト（Windowsで壊れないこと）
# ---------------------------------------------------------------------------
class OtoWeaveAppSmokeTests(unittest.TestCase):
    def test_instantiates_and_destroys_without_error(self) -> None:
        app = OtoWeaveApp(None)
        try:
            self.assertEqual(app.current_route, "notes")
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
