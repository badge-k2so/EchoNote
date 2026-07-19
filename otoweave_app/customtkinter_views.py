from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

import customtkinter as ctk
from PIL import Image, ImageDraw

from .customtkinter_mock_data import NOTE_BY_ID, NOTES
from .display_settings import DisplaySettings


# Each entry is a (light, dark) pair; CustomTkinter picks the active one
# based on ctk.set_appearance_mode(). Use theme_color() when passing a
# value to a plain tkinter API that needs a single color string.
COLORS = {
    "page": ("#e9edf1", "#1e242b"),
    "activity": ("#f8f9fa", "#252b33"),
    "detail": ("#f1f3f5", "#232930"),
    "main": ("#ffffff", "#1b2127"),
    "right": ("#f8f9fa", "#252b33"),
    "surface": ("#ffffff", "#2a3139"),
    "line": ("#dce1e6", "#3a424c"),
    "text": ("#20262d", "#e6eaee"),
    "muted": ("#68727c", "#9aa4ae"),
    "blue": ("#2f73c9", "#4f8fd9"),
    "blue_hover": ("#255fa8", "#3c78c4"),
    "blue_soft": ("#e4f0fd", "#274158"),
    "green": ("#167a5b", "#3aa583"),
    "chat_user": ("#e7f1ff", "#2b3d52"),
    "chat_ai": ("#eef1f4", "#2e353d"),
    "hover": ("#e5e9ed", "#333c46"),
    "hover_soft": ("#edf1f4", "#2e363f"),
    "textbox": ("#fbfcfd", "#20262d"),
    "input_border": ("#cbd2d9", "#46505a"),
    "box_border": ("#e2e7eb", "#3a424c"),
    "danger_hover": ("#fde9e6", "#4a2c28"),
    "danger_text": ("#a13d32", "#e08a7f"),
    "danger_border": ("#e4bbb7", "#6b4a45"),
    "accent_text": ("#075baa", "#8fc1f0"),
    "link_text": ("#245994", "#a8c8e8"),
    "note_text": ("#46515b", "#b9c2cb"),
    "card": ("#f1f5f8", "#242c34"),
    "player": ("#f7f9fb", "#232a31"),
    "record_bar": ("#fff0f0", "#3a2626"),
    "summary_text": ("#394149", "#c8d0d8"),
    # Speaker-label colors for the transcript. Color is a supplementary
    # cue only (the "話者1:" / renamed prefix text is always kept, and the
    # prefix is bolded too), so palette choice isn't safety-critical, but
    # hues are still spread out for colorblind users and each pair keeps
    # >6.5:1 contrast against both the main/textbox light and dark
    # backgrounds (checked against #ffffff/#fbfcfd and #1b2127/#20262d).
    "speaker_1": ("#175a3c", "#5fd6a5"),  # teal / minty green
    "speaker_2": ("#8a4b08", "#f0ad5c"),  # brown / amber
    "speaker_3": ("#5b3fa0", "#c3a6f5"),  # indigo / violet
    "speaker_4": ("#9c2f66", "#f093c0"),  # plum / pink
}

# Cycled in order of first appearance in a transcript; wraps around if a
# lesson somehow has more distinct speakers than colors.
SPEAKER_COLOR_KEYS = ["speaker_1", "speaker_2", "speaker_3", "speaker_4"]

def _default_ui_font(platform: str = sys.platform) -> str:
    """OS既定のUIフォント名。

    macOS には Meiryo が存在しないため、標準UIフォントのヒラギノ角ゴシックを
    使う。Windows側の既定 (Meiryo) は変えない。
    TODO(platform_support): platform_support.py 導入後は共通の is_macos()
    判定に置き換える。
    """
    if platform == "darwin":
        return "Hiragino Sans"
    return "Meiryo"


def _undo_key_hint(platform: str = sys.platform) -> str:
    """文字起こし編集中の「取り消し」操作のキー表示。OSの慣習に合わせる。"""
    if platform == "darwin":
        return "Cmd+Z"
    return "Ctrl+Z"


FONT = _default_ui_font()
BASE_FONT_SIZE = 14

UNDO_KEY_HINT = _undo_key_hint()


def theme_color(value: tuple[str, str] | str) -> str:
    """Resolve a (light, dark) color pair for plain tkinter APIs."""
    if isinstance(value, tuple):
        return value[1] if ctk.get_appearance_mode() == "Dark" else value[0]
    return value
BackgroundRunner = Callable[
    [Callable[[], Any], Callable[[Any], None] | None],
    None,
]
RouteRequester = Callable[[str], None]
NoteRequester = Callable[..., None]
ContextRequester = Callable[[str, str], None]
RecordingRequester = Callable[[dict[str, Any]], None]


def wrapped_line_spacing(paragraph_spacing: int) -> int:
    """折り返し行の行間 (spacing2) を段落間隔 (spacing3) から算出する。

    tkinter Text の spacing3 は段落（改行）後の間隔にしか効かず、
    長い発話が折り返された行の間隔は spacing2 で決まる。読みやすさの
    通例に合わせ、折り返し行間は段落間隔よりやや小さめ (60%) にする。
    """
    return max(1, round(paragraph_spacing * 0.6))


def replace_read_only_text(textbox: ctk.CTkTextbox, text: str) -> None:
    textbox.configure(state="normal")
    textbox.delete("1.0", "end")
    textbox.insert("1.0", text)
    textbox.configure(state="disabled")


def processing_indicator_state(
    transcribing: bool,
    importing: bool,
    cancel_pending: bool,
) -> dict[str, Any]:
    """取り込み・再文字起こし中の進捗バーと「中止」ボタンの状態（純関数）。

    UI なしでテストできるよう、表示判定とボタンの文言・状態をここで決める。
    """
    return {
        "visible": bool(transcribing or importing),
        "button_text": "中止中…" if cancel_pending else "中止",
        "button_state": "disabled" if cancel_pending else "normal",
    }


# AI要約が使えない環境（4Bモデル非同梱のLite版・メモリの少ない端末）で
# 生成ボタンの代わりに表示する案内。技術用語・モデル名は出さない。
SUMMARY_UNAVAILABLE_MESSAGE = (
    "AIようやくは、この端末ではじゅんび中です。\n"
    "つかえるようになるまで待っていてね。"
)


def summary_controls_visibility(available: bool) -> dict[str, bool]:
    """AI要約の可用性 → 右パネル各部品の表示可否（純関数）。

    使えない環境では生成系（作成ボタン・テンプレート選択・管理）を隠して
    案内文だけを出す。キャッシュ済み要約の閲覧と読み上げは残すため、
    要約本文ボックスと読み上げボタンは常に表示する。チャットは対象外。
    """
    return {
        "summarize_button": available,
        "template_menu": available,
        "manage_templates_button": available,
        "unavailable_notice": not available,
        "summary_box": True,
        "speak_summary_button": True,
    }


# マイク音量テストの判定 → 生徒向けの言葉。RMS/Peak の生値は見せない。
AUDIO_LEVEL_TEXTS = {
    "Good": "良好です。このまま録音できます。",
    "Caution": "少し小さめです。マイクに近づいてみてください。",
    "Poor": "小さすぎるか、音が割れています。マイクの位置を調整してください。",
}


def audio_level_text(result: dict[str, Any] | None) -> str:
    """マイク音量テストの測定結果を平易な日本語の一文へ変換する純関数。"""
    if not isinstance(result, dict):
        return "測定できませんでした。もう一度試してください。"
    state = str(result.get("state", ""))
    return AUDIO_LEVEL_TEXTS.get(
        state,
        "測定しました。もう一度試すこともできます。",
    )


class HoverTooltip:
    def __init__(
        self,
        widget,
        text: str,
        delay_ms: int = 450,
    ) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._window: ctk.CTkToplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel_schedule()
        self._after_id = self.widget.after(
            self.delay_ms,
            self._show,
        )

    def _cancel_schedule(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._window is not None or not self.widget.winfo_exists():
            return
        window = ctk.CTkToplevel(self.widget)
        window.withdraw()
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.configure(fg_color="#25313c")
        ctk.CTkLabel(
            window,
            text=self.text,
            height=32,
            corner_radius=7,
            fg_color="#25313c",
            text_color="#ffffff",
            font=ctk.CTkFont(FONT, 9, "bold"),
        ).pack(padx=10, pady=2)
        window.update_idletasks()
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 9
        y = (
            self.widget.winfo_rooty()
            + (self.widget.winfo_height() - window.winfo_reqheight()) // 2
        )
        window.geometry(f"+{x}+{max(0, y)}")
        window.deiconify()
        self._window = window

    def _hide(self, _event=None) -> None:
        self._cancel_schedule()
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None


class ActivityBar(ctk.CTkFrame):
    ITEMS = (
        ("record", "🔴", "録音"),
        ("import", "📥", "音声ファイルを取り込む"),
        ("notes", "📁", "ノート"),
        ("dictionary", "📖", "補正辞書"),
        ("settings", "⚙", "設定"),
    )

    def __init__(self, parent, request_route: RouteRequester) -> None:
        super().__init__(
            parent,
            width=72,
            fg_color=COLORS["activity"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["line"],
        )
        self.request_route = request_route
        self.buttons: dict[str, ctk.CTkButton] = {}
        self.icons: dict[str, ctk.CTkImage] = {}
        self.tooltips: dict[str, HoverTooltip] = {}
        self.grid_propagate(False)

        # OtoWeaveアイコン。アセットが無い環境では頭文字の「O」で代替する。
        brand_icon = self._load_brand_icon()
        if brand_icon is not None:
            ctk.CTkLabel(
                self,
                text="",
                image=brand_icon,
                width=40,
                height=40,
            ).pack(side="top", padx=14, pady=(16, 14))
        else:
            ctk.CTkLabel(
                self,
                text="O",
                width=40,
                height=40,
                corner_radius=8,
                fg_color=COLORS["blue"],
                text_color="#ffffff",
                font=ctk.CTkFont(FONT, 18, "bold"),
            ).pack(side="top", padx=14, pady=(16, 14))

        top_group = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        top_group.pack(side="top", fill="x")

        for key, icon, label in self.ITEMS:
            self.buttons[key] = self._activity_button(top_group, key, icon, label)
            self.buttons[key].pack(padx=10, pady=4)

    def _load_brand_icon(self) -> ctk.CTkImage | None:
        """otoweave_app/assets/icon_128.png を読み込む。無ければNone。"""
        try:
            icon_path = Path(__file__).resolve().parent / "assets" / "icon_128.png"
            if not icon_path.is_file():
                return None
            image = Image.open(icon_path)
            brand = ctk.CTkImage(light_image=image, dark_image=image, size=(40, 40))
            self.icons["_brand"] = brand
            return brand
        except Exception:
            return None

    def _activity_button(
        self,
        parent,
        key: str,
        icon: str,
        label: str,
    ) -> ctk.CTkButton:
        del icon
        image = ctk.CTkImage(
            light_image=self._draw_activity_icon(key),
            dark_image=self._draw_activity_icon(key),
            size=(28, 28),
        )
        self.icons[key] = image
        button = ctk.CTkButton(
            parent,
            text="",
            image=image,
            width=48,
            height=48,
            corner_radius=8,
            fg_color="transparent",
            hover_color="#e9edf1",
            text_color=COLORS["text"],
            font=ctk.CTkFont(FONT, 17),
            command=lambda value=key: self.request_route(value),
        )
        button._route_label = label
        self.tooltips[key] = HoverTooltip(button, label)
        return button

    @staticmethod
    def _draw_activity_icon(key: str) -> Image.Image:
        size = 64
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        if key == "record":
            draw.ellipse((10, 10, 54, 54), fill="#e44d4d", outline="#b82f36", width=4)
            draw.ellipse((18, 16, 31, 29), fill="#ff9696")
        elif key == "import":
            draw.rounded_rectangle(
                (8, 42, 56, 56),
                radius=5,
                fill="#d9f3e9",
                outline="#168267",
                width=4,
            )
            draw.polygon(
                [(25, 8), (39, 8), (39, 31), (49, 31), (32, 48), (15, 31), (25, 31)],
                fill="#20a47f",
            )
        elif key == "notes":
            draw.rounded_rectangle(
                (7, 18, 57, 54),
                radius=6,
                fill="#f3b83f",
                outline="#cc8a18",
                width=3,
            )
            draw.polygon(
                [(8, 17), (8, 11), (29, 11), (36, 19), (56, 19), (56, 25), (8, 25)],
                fill="#ffd66d",
            )
        elif key == "dictionary":
            draw.rounded_rectangle(
                (8, 9, 31, 55),
                radius=5,
                fill="#7867d5",
                outline="#5141a7",
                width=3,
            )
            draw.rounded_rectangle(
                (33, 9, 56, 55),
                radius=5,
                fill="#9b86ea",
                outline="#5141a7",
                width=3,
            )
            draw.line((32, 12, 32, 53), fill="#ffffff", width=3)
            draw.line((14, 20, 26, 20), fill="#e8e3ff", width=3)
            draw.line((38, 20, 50, 20), fill="#eeeaff", width=3)
        elif key == "settings":
            center = (32, 32)
            for step in range(8):
                angle = math.radians(step * 45)
                x1 = center[0] + math.cos(angle) * 17
                y1 = center[1] + math.sin(angle) * 17
                x2 = center[0] + math.cos(angle) * 25
                y2 = center[1] + math.sin(angle) * 25
                draw.line((x1, y1, x2, y2), fill="#e8842e", width=8)
            draw.ellipse((14, 14, 50, 50), fill="#f39a45", outline="#c76622", width=3)
            draw.ellipse((25, 25, 39, 39), fill="#fff4e8", outline="#c76622", width=3)
        return image

    def set_active(self, route: str) -> None:
        for key, button in self.buttons.items():
            active = key == route
            button.configure(
                fg_color=COLORS["blue_soft"] if active else "transparent",
                hover_color=COLORS["blue_soft"] if active else "#e9edf1",
                text_color=COLORS["accent_text"] if active else COLORS["text"],
                border_width=1 if active else 0,
                border_color="#bcd6f2",
            )


class DetailPane(ctk.CTkFrame):
    VIEWS = ("notes", "search", "dictionary", "settings")
    LINE_SPACING_LABELS = {"Standard": "標準", "Comfortable": "ゆったり"}
    TEXT_WIDTH_LABELS = {"Wide": "ワイド", "Reading Width": "読書幅"}
    COLOR_MODE_LABELS = {"Light": "ライト", "Dark": "ダーク"}

    def __init__(
        self,
        parent,
        run_background: BackgroundRunner,
        request_note: NoteRequester,
        request_context: ContextRequester,
        font_families: list[str] | None = None,
        display_settings: DisplaySettings | None = None,
        selected_font_size: int = BASE_FONT_SIZE,
        request_display_settings: Callable[[dict[str, Any]], None] | None = None,
        request_dictionary_manage: Callable[[], None] | None = None,
        request_content_search: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(
            parent,
            width=280,
            fg_color=COLORS["detail"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["line"],
        )
        self.run_background = run_background
        self.request_note = request_note
        self.request_context = request_context
        self.display_settings = display_settings or DisplaySettings(font_family=FONT)
        self.font_families = font_families or [self.display_settings.font_family]
        self.selected_font = self.display_settings.font_family
        self.selected_font_size = selected_font_size
        self.request_display_settings = request_display_settings
        self.request_dictionary_manage = request_dictionary_manage
        self.request_content_search = request_content_search
        self.active_note_id = "kokoro"
        self.note_buttons: dict[str, ctk.CTkButton] = {}
        self.folder_buttons: dict[str, ctk.CTkButton] = {}
        self.context_buttons: dict[tuple[str, str], ctk.CTkButton] = {}
        self._search_notes: list[dict] | None = None
        self._note_groups: dict[str, list[dict]] = {}
        self._visible_note_groups: dict[str, list[dict]] = {}
        self._collapsed_note_groups: set[str] = set()
        self._active_note_query = ""
        self.view_frames: dict[str, ctk.CTkFrame] = {}

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_propagate(False)
        self.title_label = ctk.CTkLabel(
            self,
            text="ノート",
            font=ctk.CTkFont(FONT, 17, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        )
        self.title_label.grid(row=0, column=0, sticky="ew", padx=18, pady=(20, 12))

        self.notes_search_bar = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        self.notes_search_bar.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=16,
            pady=(0, 10),
        )
        self.notes_search_bar.grid_columnconfigure(0, weight=1)
        self.notes_search_entry = ctk.CTkEntry(
            self.notes_search_bar,
            placeholder_text="ノート・元ファイル名を検索",
            height=38,
            corner_radius=8,
            border_color=COLORS["input_border"],
            font=ctk.CTkFont(FONT, 10),
        )
        self.notes_search_entry.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 6),
        )
        self.notes_search_entry.bind(
            "<Return>",
            self._filter_notes_from_event,
        )
        ctk.CTkButton(
            self.notes_search_bar,
            text="検索",
            command=self._filter_notes,
            width=62,
            height=38,
            corner_radius=8,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            font=ctk.CTkFont(FONT, 10, "bold"),
        ).grid(row=0, column=1)
        ctk.CTkLabel(
            self.notes_search_bar,
            text="ノート名・元ファイル名・文字起こし本文を検索します",
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
            anchor="w",
        ).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(5, 0),
        )

        self.content_host = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.content_host.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 12))
        self.content_host.grid_columnconfigure(0, weight=1)
        self.content_host.grid_rowconfigure(0, weight=1)

        self.view_frames["notes"] = self._build_notes_view()
        self.view_frames["search"] = self._build_search_view()
        self.view_frames["dictionary"] = self._build_dictionary_view()
        self.view_frames["settings"] = self._build_settings_view()
        # Starts empty; real data arrives via populate_notes. Mock notes
        # are injected only by the controller-less demo shell.
        self._note_groups = {}
        self._render_note_groups(self._note_groups)
        for frame in self.view_frames.values():
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_view("notes")

    def _scroll_frame(self) -> ctk.CTkScrollableFrame:
        frame = ctk.CTkScrollableFrame(
            self.content_host,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color="#c2c9d0",
            scrollbar_button_hover_color="#aab3bc",
        )
        frame.grid_columnconfigure(0, weight=1)
        return frame

    def _build_notes_view(self) -> ctk.CTkFrame:
        return self._scroll_frame()

    def _build_search_view(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.content_host, fg_color="transparent", corner_radius=0)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1)
        search_row = ctk.CTkFrame(frame, fg_color="transparent", corner_radius=0)
        search_row.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 10))
        search_row.grid_columnconfigure(0, weight=1)
        self.search_entry = ctk.CTkEntry(
            search_row,
            placeholder_text="ノートを検索",
            height=38,
            corner_radius=8,
            border_color=COLORS["input_border"],
            font=ctk.CTkFont(FONT, 10),
        )
        self.search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.search_entry.bind("<Return>", self._request_search_from_event)
        ctk.CTkButton(
            search_row,
            text="検索",
            width=64,
            height=38,
            corner_radius=8,
            command=self._request_search,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            font=ctk.CTkFont(FONT, 10, "bold"),
        ).grid(row=0, column=1)
        self.search_status = ctk.CTkLabel(
            frame,
            text="タイトルと文字起こしを検索します",
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
            anchor="w",
        )
        self.search_status.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        self.search_results = ctk.CTkScrollableFrame(
            frame,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color="#c2c9d0",
            scrollbar_button_hover_color="#aab3bc",
        )
        self.search_results.grid(row=2, column=0, sticky="nsew")
        self.search_results.grid_columnconfigure(0, weight=1)
        self._render_search_results([])
        return frame

    def _build_dictionary_view(self) -> ctk.CTkFrame:
        frame = self._scroll_frame()
        ctk.CTkButton(
            frame,
            text="＋ 用語を登録・管理",
            command=self.request_dictionary_manage or (lambda: None),
            height=42,
            corner_radius=8,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            font=ctk.CTkFont(FONT, 10, "bold"),
        ).grid(row=0, column=0, sticky="ew", padx=6, pady=(8, 12))
        categories = (
            ("学習用語", "学習用語"),
            ("読み書き支援", "読み書き支援"),
            ("英語表現", "英語表現"),
            ("登録した言葉", "登録した言葉"),
        )
        for row, (key, label) in enumerate(categories, start=1):
            button = ctk.CTkButton(
                frame,
                text=f"📖  {label}",
                command=lambda value=key: self.request_context("dictionary", value),
                height=42,
                corner_radius=8,
                fg_color="transparent",
                hover_color=COLORS["hover"],
                text_color=COLORS["text"],
                anchor="w",
                font=ctk.CTkFont(FONT, 11),
            )
            button.grid(row=row, column=0, sticky="ew", padx=6, pady=3)
            self.context_buttons[("dictionary", key)] = button
        return frame

    def _build_settings_view(self) -> ctk.CTkFrame:
        frame = self._scroll_frame()
        ctk.CTkLabel(
            frame,
            text="フォント",
            font=ctk.CTkFont(FONT, 11, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=6, pady=(10, 4))
        self.font_menu = ctk.CTkOptionMenu(
            frame,
            values=self.font_families,
            command=lambda _value: self._request_display_change(),
            height=38,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 10),
        )
        self.font_menu.set(self.selected_font)
        self.font_menu.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 12))

        ctk.CTkLabel(
            frame,
            text="文字サイズ",
            font=ctk.CTkFont(FONT, 11, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=6, pady=(4, 4))
        self.font_size_menu = ctk.CTkOptionMenu(
            frame,
            values=["12", "14", "16", "18"],
            command=lambda _value: self._request_display_change(),
            height=38,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 10),
        )
        self.font_size_menu.set(str(self.selected_font_size))
        self.font_size_menu.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=6,
            pady=(0, 12),
        )

        def option_row(row: int, label: str, values: list[str], current: str) -> ctk.CTkOptionMenu:
            ctk.CTkLabel(
                frame,
                text=label,
                font=ctk.CTkFont(FONT, 11, "bold"),
                text_color=COLORS["text"],
                anchor="w",
            ).grid(row=row, column=0, sticky="ew", padx=6, pady=(4, 4))
            menu = ctk.CTkOptionMenu(
                frame,
                values=values,
                command=lambda _value: self._request_display_change(),
                height=38,
                corner_radius=8,
                font=ctk.CTkFont(FONT, 10),
            )
            menu.set(current)
            menu.grid(row=row + 1, column=0, sticky="ew", padx=6, pady=(0, 12))
            return menu

        settings = self.display_settings
        self.line_spacing_menu = option_row(
            4,
            "行間",
            list(self.LINE_SPACING_LABELS.values()),
            self.LINE_SPACING_LABELS.get(settings.line_spacing, "ゆったり"),
        )
        self.text_width_menu = option_row(
            6,
            "文字はば",
            list(self.TEXT_WIDTH_LABELS.values()),
            self.TEXT_WIDTH_LABELS.get(settings.text_width, "読書幅"),
        )
        self.color_mode_menu = option_row(
            8,
            "配色",
            list(self.COLOR_MODE_LABELS.values()),
            self.COLOR_MODE_LABELS.get(settings.color_mode, "ライト"),
        )
        self.live_follow_switch = ctk.CTkSwitch(
            frame,
            text="最新の文字起こしを追う",
            command=self._request_display_change,
            font=ctk.CTkFont(FONT, 10),
        )
        if settings.live_follow:
            self.live_follow_switch.select()
        self.live_follow_switch.grid(
            row=10,
            column=0,
            sticky="ew",
            padx=6,
            pady=(4, 6),
        )
        self.highlight_switch = ctk.CTkSwitch(
            frame,
            text="現在の発話を強調する",
            command=self._request_display_change,
            font=ctk.CTkFont(FONT, 10),
        )
        if settings.highlight_current:
            self.highlight_switch.select()
        self.highlight_switch.grid(
            row=11,
            column=0,
            sticky="ew",
            padx=6,
            pady=(0, 18),
        )

        for row, (key, label) in enumerate(
            (
                ("モデルとライセンス", "モデルとライセンス"),
                ("システム情報", "システム情報"),
            ),
            start=12,
        ):
            button = ctk.CTkButton(
                frame,
                text=f"ⓘ  {label}",
                command=lambda value=key: self.request_context("settings", value),
                height=42,
                corner_radius=8,
                fg_color="transparent",
                hover_color=COLORS["hover"],
                text_color=COLORS["text"],
                anchor="w",
                font=ctk.CTkFont(FONT, 11),
            )
            button.grid(row=row, column=0, sticky="ew", padx=6, pady=3)
            self.context_buttons[("settings", key)] = button
        return frame

    @staticmethod
    def _label_to_value(labels: dict[str, str], label: str, default: str) -> str:
        return next(
            (value for value, text in labels.items() if text == label),
            default,
        )

    def display_preferences(self) -> dict[str, Any]:
        return {
            "font_family": self.font_menu.get(),
            "font_size": int(self.font_size_menu.get()),
            "line_spacing": self._label_to_value(
                self.LINE_SPACING_LABELS,
                self.line_spacing_menu.get(),
                "Comfortable",
            ),
            "text_width": self._label_to_value(
                self.TEXT_WIDTH_LABELS,
                self.text_width_menu.get(),
                "Reading Width",
            ),
            "color_mode": self._label_to_value(
                self.COLOR_MODE_LABELS,
                self.color_mode_menu.get(),
                "Light",
            ),
            "live_follow": bool(self.live_follow_switch.get()),
            "highlight_current": bool(self.highlight_switch.get()),
        }

    def _request_display_change(self) -> None:
        if self.request_display_settings is not None:
            self.request_display_settings(self.display_preferences())

    def populate_notes(self, groups: dict[str, list[dict]]) -> None:
        """Rebuild the notes list from live data (replaces mock content)."""
        self._note_groups = groups
        self._collapsed_note_groups.intersection_update(groups)
        self._search_notes: list[dict] = [note for notes in groups.values() for note in notes]
        if self.notes_search_entry.get().strip():
            self._filter_notes()
        else:
            self._active_note_query = ""
            self._render_note_groups(groups)

    def _render_note_groups(self, groups: dict[str, list[dict]]) -> None:
        self._visible_note_groups = groups
        notes_frame = self.view_frames["notes"]
        query = self._active_note_query
        for child in notes_frame.winfo_children():
            child.destroy()
        self.note_buttons.clear()
        self.folder_buttons.clear()
        row = 0
        for group_name, notes in groups.items():
            collapsed = (
                group_name in self._collapsed_note_groups
                and not self._active_note_query
            )
            folder_button = ctk.CTkButton(
                notes_frame,
                text=f"{'▶' if collapsed else '▼'}  📁  {group_name}",
                command=lambda value=group_name: self._toggle_note_group(value),
                height=36,
                corner_radius=8,
                fg_color="transparent",
                hover_color=COLORS["hover"],
                font=ctk.CTkFont(FONT, 11, "bold"),
                text_color=COLORS["muted"],
                anchor="w",
            )
            folder_button.grid(
                row=row,
                column=0,
                sticky="ew",
                padx=6,
                pady=(12, 4),
            )
            self.folder_buttons[group_name] = folder_button
            row += 1
            if collapsed:
                continue
            for note in notes:
                source_name = note.get("source_audio_name", "")
                compact_source = source_name
                if len(compact_source) > 28:
                    compact_source = f"{compact_source[:25]}…"
                source_line = (
                    f"\n   🎧 {compact_source}"
                    if compact_source
                    else ""
                )
                btn = ctk.CTkButton(
                    notes_frame,
                    text=f"📄  {note['label']}{source_line}",
                    command=lambda nid=note["id"], value=query: self.request_note(
                        nid,
                        value,
                    ),
                    height=54 if source_line else 38,
                    corner_radius=8,
                    fg_color="transparent",
                    hover_color=COLORS["hover"],
                    text_color=COLORS["note_text"],
                    anchor="w",
                    font=ctk.CTkFont(FONT, 10),
                )
                btn.grid(row=row, column=0, sticky="ew", padx=8, pady=2)
                self.note_buttons[note["id"]] = btn
                row += 1
        if not groups:
            ctk.CTkLabel(
                notes_frame,
                text="一致するノートはありません",
                font=ctk.CTkFont(FONT, 10),
                text_color=COLORS["muted"],
            ).grid(row=0, column=0, padx=8, pady=18)

    def _toggle_note_group(self, group_name: str) -> None:
        if group_name in self._collapsed_note_groups:
            self._collapsed_note_groups.remove(group_name)
        else:
            self._collapsed_note_groups.add(group_name)
        self._render_note_groups(self._visible_note_groups)
        self.set_active_note(self.active_note_id)
        self._request_display_change()

    def _filter_notes_from_event(self, _event=None) -> str:
        self._filter_notes()
        return "break"

    def _filter_notes(self) -> None:
        raw_query = self.notes_search_entry.get().strip()
        query = raw_query.lower()
        self._active_note_query = raw_query
        if not query:
            self._render_note_groups(self._note_groups)
            self._request_display_change()
            return
        # Immediate pass over metadata fields; transcript bodies are
        # searched in the background and merged in via
        # apply_content_search once ready.
        self._render_filtered_note_groups(query, set())
        if self.request_content_search is not None:
            self.request_content_search(raw_query)

    def _render_filtered_note_groups(
        self,
        query: str,
        content_matched_ids: set[str],
    ) -> None:
        filtered: dict[str, list[dict]] = {}
        for group_name, notes in self._note_groups.items():
            matches = [
                note
                for note in notes
                if note.get("id") in content_matched_ids
                or query
                in (
                    note.get("title", "")
                    + note.get("source_audio_name", "")
                    + note.get("keywords", "")
                    + (note.get("transcript") or "")
                ).lower()
            ]
            if matches:
                filtered[group_name] = matches
        self._render_note_groups(filtered)
        self._request_display_change()

    def apply_content_search(self, query: str, matched_ids: set[str]) -> None:
        """Merge background transcript-search results into the note list."""
        if query != self._active_note_query:
            return
        self._render_filtered_note_groups(query.lower(), matched_ids)

    def show_view(self, view_name: str) -> None:
        if view_name not in self.view_frames:
            return
        titles = {
            "notes": "ノート",
            "search": "検索",
            "dictionary": "辞書",
            "settings": "設定",
        }
        self.title_label.configure(text=titles[view_name])
        if view_name == "notes":
            self.notes_search_bar.grid()
        else:
            self.notes_search_bar.grid_remove()
        frame = self.view_frames[view_name]
        outer_frame = getattr(frame, "_parent_frame", frame)
        outer_frame.tkraise()

    def set_active_note(self, note_id: str) -> None:
        self.active_note_id = note_id
        for key, button in self.note_buttons.items():
            active = key == note_id
            button.configure(
                fg_color=COLORS["blue_soft"] if active else "transparent",
                hover_color=COLORS["blue_soft"] if active else COLORS["hover"],
                text_color=COLORS["accent_text"] if active else COLORS["note_text"],
                font=ctk.CTkFont(FONT, 10, "bold" if active else "normal"),
            )

    def set_active_context(self, route: str, item: str) -> None:
        for (button_route, key), button in self.context_buttons.items():
            active = button_route == route and key == item
            button.configure(
                fg_color=COLORS["blue_soft"] if active else "transparent",
                hover_color=COLORS["blue_soft"] if active else COLORS["hover"],
                text_color=COLORS["accent_text"] if active else COLORS["text"],
                font=ctk.CTkFont(FONT, 11, "bold" if active else "normal"),
            )

    def _request_search_from_event(self, _event=None) -> str:
        self._request_search()
        return "break"

    def _request_search(self) -> None:
        query = self.search_entry.get().strip().lower()
        source = self._search_notes if self._search_notes is not None else list(NOTE_BY_ID.values())

        def worker() -> list[dict]:
            time.sleep(0.08)
            if not query:
                return list(source)
            return [
                note
                for note in source
                if query
                in (
                    note["title"]
                    + note.get("keywords", "")
                    + note.get("transcript", "")
                    + note.get("source_audio_name", "")
                ).lower()
            ]

        self.run_background(worker, self._render_search_results)

    def _render_search_results(self, notes: list[dict[str, str]]) -> None:
        for child in self.search_results.winfo_children():
            child.destroy()
        self.search_status.configure(text=f"{len(notes)}件のノート")
        if not notes:
            ctk.CTkLabel(
                self.search_results,
                text="一致するノートはありません",
                font=ctk.CTkFont(FONT, 10),
                text_color=COLORS["muted"],
            ).grid(row=0, column=0, padx=8, pady=18)
            return
        for row, note in enumerate(notes):
            ctk.CTkButton(
                self.search_results,
                text=f"{note['title']}\n{note['meta']}",
                command=lambda note_id=note["id"]: self.request_note(note_id),
                height=76 if note.get("source_audio_name") else 58,
                corner_radius=8,
                fg_color=COLORS["surface"],
                hover_color=COLORS["hover"],
                border_width=1,
                border_color=COLORS["line"],
                text_color=COLORS["text"],
                anchor="w",
                font=ctk.CTkFont(FONT, 9),
            ).grid(row=row, column=0, sticky="ew", padx=4, pady=4)


class MainPane(ctk.CTkFrame):
    def __init__(
        self,
        parent,
        run_background: BackgroundRunner,
        request_right_toggle: Callable[[], None],
        request_transcribe: Callable[[], None],
        request_rename: Callable[[], None],
        request_delete: Callable[[], None],
        request_record_start: RecordingRequester | None = None,
        request_audio_test: RecordingRequester | None = None,
        request_save_location: Callable[[], None] | None = None,
        request_transcript_save: Callable[[str], bool] | None = None,
        request_play_toggle: Callable[[], None] | None = None,
        request_segment_play: Callable[[float], None] | None = None,
        request_speak_transcript: Callable[[str], None] | None = None,
        request_cancel_processing: Callable[[], None] | None = None,
        request_speaker_rename: Callable[[], None] | None = None,
        request_diarize: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            parent,
            fg_color=COLORS["main"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["line"],
        )
        self.run_background = run_background
        self.request_right_toggle = request_right_toggle
        self.request_transcribe = request_transcribe
        self.request_rename = request_rename
        self.request_delete = request_delete
        self.request_record_start = request_record_start
        self.request_audio_test = request_audio_test
        self.request_save_location = request_save_location
        self.request_transcript_save = request_transcript_save
        self.request_play_toggle = request_play_toggle
        self.request_segment_play = request_segment_play
        self.request_speak_transcript = request_speak_transcript
        self.request_cancel_processing = request_cancel_processing
        self.request_speaker_rename = request_speaker_rename
        self.request_diarize = request_diarize
        self._player_duration = 0.0
        self._editing_transcript = False
        self._display_transcript = ""
        self._editable_transcript = ""
        self._transcript_editable = False
        self._has_speakers = False
        self._speaker_lines: list[str] = []
        self._diarization_available = False
        self._can_diarize_note = False
        self._diarizing = False
        self._recording_sources: list[Any] = []
        self._recording_source_by_label: dict[str, Any] = {}
        self._recording_details_visible = False
        self._transcribe_available = False
        self._transcribe_is_retry = False
        self._transcribing = False
        self._transcribing_blocked = False
        self._importing = False
        self._cancel_processing_pending = False
        self._search_matches: list[tuple[str, str]] = []
        self._search_match_index = -1
        self._search_query = ""
        self._live_text = ""
        self._live_active = False
        self._live_follow_enabled = True
        self._live_highlight_enabled = True
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 4))
        header.grid_columnconfigure(0, weight=1)
        self.title_label = ctk.CTkLabel(
            header,
            text="",
            font=ctk.CTkFont(FONT, 24, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        )
        self.title_label.grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="ew",
        )
        self.transcribe_button = ctk.CTkButton(
            header,
            text="文字起こしを開始",
            command=self.request_transcribe,
            width=132,
            height=34,
            corner_radius=8,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            font=ctk.CTkFont(FONT, 10, "bold"),
        )
        self.transcribe_button.grid(
            row=1,
            column=1,
            padx=(10, 0),
            pady=(6, 0),
        )
        self.transcribe_button.grid_remove()
        self.right_toggle_button = ctk.CTkButton(
            header,
            text="💡 右パネルを隠す",
            command=self.request_right_toggle,
            width=128,
            height=34,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["hover_soft"],
            text_color=COLORS["muted"],
            border_width=1,
            border_color=COLORS["line"],
            font=ctk.CTkFont(FONT, 9),
        )
        self.right_toggle_button.grid(
            row=1,
            column=2,
            padx=(10, 0),
            pady=(6, 0),
        )
        self.meta_row = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        self.meta_row.grid(row=1, column=0, sticky="ew", padx=24)
        self.meta_row.grid_columnconfigure(0, weight=1)
        self.meta_label = ctk.CTkLabel(
            self.meta_row,
            text="",
            font=ctk.CTkFont(FONT, 10),
            text_color=COLORS["muted"],
            anchor="w",
            justify="left",
        )
        self.meta_label.grid(row=0, column=0, sticky="ew")
        self.rename_button = ctk.CTkButton(
            self.meta_row,
            text="名前変更",
            command=self.request_rename,
            width=76,
            height=30,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["hover_soft"],
            text_color=COLORS["muted"],
            border_width=1,
            border_color=COLORS["line"],
            font=ctk.CTkFont(FONT, 9),
        )
        self.rename_button.grid(row=0, column=1, padx=(8, 0))
        self.delete_button = ctk.CTkButton(
            self.meta_row,
            text="削除",
            command=self.request_delete,
            width=58,
            height=30,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["danger_hover"],
            text_color=COLORS["danger_text"],
            border_width=1,
            border_color=COLORS["danger_border"],
            font=ctk.CTkFont(FONT, 9),
        )
        self.delete_button.grid(row=0, column=2, padx=(8, 0))
        self.rename_button.grid_remove()
        self.delete_button.grid_remove()
        self.keyword_label = ctk.CTkLabel(
            self,
            text="",
            height=36,
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["link_text"],
            fg_color=COLORS["blue_soft"],
            corner_radius=8,
            anchor="w",
        )
        self.keyword_label.grid(row=2, column=0, sticky="ew", padx=24, pady=(12, 14))

        self.player = ctk.CTkFrame(self, fg_color=COLORS["player"], corner_radius=8)
        self.player.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 16))
        self.player.grid_columnconfigure(2, weight=1)
        self.play_button = ctk.CTkButton(
            self.player,
            text="▶",
            width=40,
            height=36,
            corner_radius=18,
            command=self._request_play_toggle,
            fg_color=COLORS["text"],
            hover_color="#3d454d",
            font=ctk.CTkFont(FONT, 11, "bold"),
        )
        self.play_button.grid(row=0, column=0, padx=(12, 8), pady=10)
        self.player_position_label = ctk.CTkLabel(
            self.player,
            text="00:00",
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
        )
        self.player_position_label.grid(row=0, column=1, padx=(0, 10))
        self.player_progress = ctk.CTkProgressBar(
            self.player,
            height=6,
            corner_radius=3,
            progress_color=COLORS["blue"],
            fg_color=COLORS["line"],
        )
        self.player_progress.grid(row=0, column=2, sticky="ew", padx=(0, 10))
        self.player_progress.set(0.0)
        self.player_duration_label = ctk.CTkLabel(
            self.player,
            text="--:--",
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
        )
        self.player_duration_label.grid(row=0, column=3, padx=(0, 12))

        self._record_bar = ctk.CTkFrame(self, fg_color=COLORS["record_bar"], corner_radius=8)
        self._record_bar.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 16))
        self._record_bar.grid_columnconfigure(2, weight=1)
        self.stop_button = ctk.CTkButton(
            self._record_bar,
            text="⏹ 停止",
            width=80,
            height=36,
            corner_radius=8,
            fg_color="#c0392b",
            hover_color="#a93226",
            text_color="#ffffff",
            font=ctk.CTkFont(FONT, 11, "bold"),
            command=lambda: None,  # overridden by OtoWeaveApp
        )
        self.stop_button.grid(row=0, column=0, padx=(12, 8), pady=10)
        self.pause_button = ctk.CTkButton(
            self._record_bar,
            text="⏸",
            width=36,
            height=36,
            corner_radius=8,
            fg_color="transparent",
            hover_color="#ffe0de",
            text_color=COLORS["text"],
            border_width=1,
            border_color="#e0b0b0",
            font=ctk.CTkFont(FONT, 12),
            command=lambda: None,  # overridden by OtoWeaveApp
        )
        self.pause_button.grid(row=0, column=1, padx=(0, 10), pady=10)
        self._elapsed_label = ctk.CTkLabel(
            self._record_bar,
            text="00:00",
            font=ctk.CTkFont(FONT, 20, "bold"),
            text_color="#c0392b",
        )
        self._elapsed_label.grid(row=0, column=2, padx=10)
        self._record_bar.grid_remove()

        self.transcript_toolbar = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        self.transcript_toolbar.grid(
            row=4,
            column=0,
            sticky="ew",
            padx=24,
            pady=(0, 8),
        )
        self.transcript_toolbar.grid_columnconfigure(0, weight=1)
        self.section_label = ctk.CTkLabel(
            self.transcript_toolbar,
            text="文字起こし",
            font=ctk.CTkFont(FONT, 14, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        )
        self.section_label.grid(row=0, column=0, sticky="ew")
        self.edit_transcript_button = self._transcript_tool_button(
            self.transcript_toolbar,
            "編集",
            self._begin_transcript_edit,
        )
        self.edit_transcript_button.grid(row=0, column=1, padx=(6, 0))
        self.copy_transcript_button = self._transcript_tool_button(
            self.transcript_toolbar,
            "コピー",
            self._copy_transcript,
        )
        self.copy_transcript_button.grid(row=0, column=2, padx=(6, 0))
        self.cut_transcript_button = self._transcript_tool_button(
            self.transcript_toolbar,
            "切り取り",
            lambda: self.textbox._textbox.event_generate("<<Cut>>"),
        )
        self.cut_transcript_button.grid(row=0, column=3, padx=(6, 0))
        self.paste_transcript_button = self._transcript_tool_button(
            self.transcript_toolbar,
            "貼り付け",
            lambda: self.textbox._textbox.event_generate("<<Paste>>"),
        )
        self.paste_transcript_button.grid(row=0, column=4, padx=(6, 0))
        self.cancel_transcript_button = self._transcript_tool_button(
            self.transcript_toolbar,
            "取消",
            self._cancel_transcript_edit,
        )
        self.cancel_transcript_button.grid(row=0, column=5, padx=(6, 0))
        self.save_transcript_button = ctk.CTkButton(
            self.transcript_toolbar,
            text="保存",
            command=self._save_transcript_edit,
            width=62,
            height=32,
            corner_radius=8,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            font=ctk.CTkFont(FONT, 9, "bold"),
        )
        self.save_transcript_button.grid(row=0, column=6, padx=(6, 0))
        self.speak_transcript_button = self._transcript_tool_button(
            self.transcript_toolbar,
            "🔊 読み上げ",
            self._request_speak_transcript,
        )
        self.speak_transcript_button.configure(width=96)
        self.speak_transcript_button.grid(row=0, column=7, padx=(6, 0))
        self.diarize_button = self._transcript_tool_button(
            self.transcript_toolbar,
            "話者を推定",
            self._request_diarize,
        )
        self.diarize_button.configure(width=96)
        self.diarize_button.grid(row=0, column=8, padx=(6, 0))
        self.diarize_button.grid_remove()
        self.rename_speakers_button = self._transcript_tool_button(
            self.transcript_toolbar,
            "話者名を変更",
            self._request_speaker_rename,
        )
        self.rename_speakers_button.configure(width=104)
        self.rename_speakers_button.grid(row=0, column=9, padx=(6, 0))
        self.rename_speakers_button.grid_remove()
        self._set_transcript_editing(False)
        self.textbox = ctk.CTkTextbox(
            self,
            wrap="word",
            fg_color=COLORS["textbox"],
            text_color=COLORS["text"],
            border_width=1,
            border_color=COLORS["box_border"],
            corner_radius=8,
            font=ctk.CTkFont(FONT, 12),
            spacing2=wrapped_line_spacing(9),
            spacing3=9,
            padx=18,
            pady=16,
            cursor="xterm",
        )
        self.textbox.grid(row=5, column=0, sticky="nsew", padx=24, pady=(0, 12))
        self.textbox._textbox.configure(undo=True, maxundo=100)
        self.return_live_button = ctk.CTkButton(
            self,
            text="↓ 最新へ戻る",
            command=self._return_to_live,
            width=118,
            height=34,
            corner_radius=17,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            font=ctk.CTkFont(FONT, 10, "bold"),
        )
        # Recording only: shown by _on_live_scroll / update_live_transcript.
        self.textbox._textbox.bind(
            "<MouseWheel>",
            self._on_live_scroll,
            add="+",
        )
        # Click a leading timestamp to listen from that position.
        raw = self.textbox._textbox
        raw.tag_bind("seg_time", "<Button-1>", self._on_timestamp_click)
        raw.tag_bind(
            "seg_time",
            "<Enter>",
            lambda _event: raw.configure(cursor="hand2"),
        )
        raw.tag_bind(
            "seg_time",
            "<Leave>",
            lambda _event: raw.configure(cursor="xterm"),
        )
        self.record_setup = self._build_record_setup()
        self.record_setup.grid(
            row=5,
            column=0,
            sticky="nsew",
            padx=24,
            pady=(0, 12),
        )
        self.record_setup.grid_remove()
        self.status_row = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        self.status_row.grid(
            row=6,
            column=0,
            sticky="ew",
            padx=24,
            pady=(0, 16),
        )
        self.status_row.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(
            self.status_row,
            text="文字を選択してコピーできます",
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.search_position_label = ctk.CTkLabel(
            self.status_row,
            text="",
            width=48,
            font=ctk.CTkFont(FONT, 9, "bold"),
            text_color=COLORS["muted"],
        )
        self.search_position_label.grid(row=0, column=1, padx=(8, 4))
        self.search_previous_button = ctk.CTkButton(
            self.status_row,
            text="＜",
            command=self._show_previous_search_match,
            width=34,
            height=30,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["hover"],
            text_color=COLORS["text"],
            border_width=1,
            border_color=COLORS["line"],
            font=ctk.CTkFont(FONT, 11, "bold"),
        )
        self.search_previous_button.grid(row=0, column=2, padx=2)
        self.search_next_button = ctk.CTkButton(
            self.status_row,
            text="＞",
            command=self._show_next_search_match,
            width=34,
            height=30,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["hover"],
            text_color=COLORS["text"],
            border_width=1,
            border_color=COLORS["line"],
            font=ctk.CTkFont(FONT, 11, "bold"),
        )
        self.search_next_button.grid(row=0, column=3, padx=(2, 0))
        self._set_search_navigation_visible(False)
        # 取り込み・あとから文字起こしの実行中に表示する不確定プログレス
        # バーと「中止」ボタン。処理が動いていることを見せ、途中で
        # やめられるようにする。
        self.processing_bar = ctk.CTkProgressBar(
            self.status_row,
            width=120,
            height=6,
            corner_radius=3,
            mode="indeterminate",
            progress_color=COLORS["blue"],
            fg_color=COLORS["line"],
        )
        self.processing_bar.grid(row=0, column=4, padx=(10, 4))
        self.cancel_processing_button = ctk.CTkButton(
            self.status_row,
            text="中止",
            command=self._request_cancel_processing,
            width=58,
            height=30,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["danger_hover"],
            text_color=COLORS["danger_text"],
            border_width=1,
            border_color=COLORS["danger_border"],
            font=ctk.CTkFont(FONT, 9),
        )
        self.cancel_processing_button.grid(row=0, column=5, padx=(4, 0))
        self.processing_bar.grid_remove()
        self.cancel_processing_button.grid_remove()

    @staticmethod
    def _transcript_tool_button(
        parent,
        text: str,
        command: Callable[[], None],
    ) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=66,
            height=32,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["hover"],
            text_color=COLORS["muted"],
            border_width=1,
            border_color=COLORS["line"],
            font=ctk.CTkFont(FONT, 9),
        )

    def _build_record_setup(self) -> ctk.CTkScrollableFrame:
        frame = ctk.CTkScrollableFrame(
            self,
            fg_color=COLORS["textbox"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["box_border"],
        )
        frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            frame,
            text="通常設定",
            font=ctk.CTkFont(FONT, 14, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(18, 12))
        for column, text in enumerate(("録音対象", "入力デバイス")):
            ctk.CTkLabel(
                frame,
                text=text,
                font=ctk.CTkFont(FONT, 10, "bold"),
                text_color=COLORS["text"],
                anchor="w",
            ).grid(
                row=1,
                column=column,
                sticky="ew",
                padx=(18, 8) if column == 0 else (8, 18),
            )

        self.record_target_menu = ctk.CTkOptionMenu(
            frame,
            values=["マイク"],
            command=lambda _value: self._record_target_changed(),
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 10),
        )
        self.record_target_menu.grid(row=2, column=0, sticky="ew", padx=(18, 8), pady=(5, 14))
        self.record_device_menu = ctk.CTkOptionMenu(
            frame,
            values=["音声入力を検出中…"],
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 10),
        )
        self.record_device_menu.grid(row=2, column=1, sticky="ew", padx=(8, 18), pady=(5, 14))

        meter_card = ctk.CTkFrame(frame, fg_color=COLORS["card"], corner_radius=12)
        meter_card.grid(row=3, column=0, columnspan=2, sticky="ew", padx=18, pady=(2, 14))
        meter_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            meter_card,
            text="音量メーター",
            font=ctk.CTkFont(FONT, 10, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 4))
        self.record_level_meter = ctk.CTkProgressBar(
            meter_card,
            height=12,
            corner_radius=6,
            progress_color=COLORS["green"],
            fg_color="#d7dde2",
        )
        self.record_level_meter.grid(row=1, column=0, sticky="ew", padx=14, pady=(2, 5))
        self.record_level_meter.set(0)
        self.record_level_status = ctk.CTkLabel(
            meter_card,
            text="3秒テストで入力レベルを確認できます",
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
            anchor="w",
        )
        self.record_level_status.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.audio_test_button = ctk.CTkButton(
            meter_card,
            text="3秒テスト",
            command=self._request_audio_test,
            width=108,
            height=36,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["hover"],
            text_color=COLORS["text"],
            border_width=1,
            border_color=COLORS["input_border"],
            font=ctk.CTkFont(FONT, 10, "bold"),
        )
        self.audio_test_button.grid(row=0, column=1, rowspan=3, padx=(8, 14), pady=12)

        self.record_start_button = ctk.CTkButton(
            frame,
            text="🔴  録音を開始",
            command=self._request_record_start,
            height=46,
            corner_radius=12,
            fg_color="#c0392b",
            hover_color="#a93226",
            font=ctk.CTkFont(FONT, 12, "bold"),
        )
        self.record_start_button.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=18,
            pady=(0, 16),
        )
        self.record_details_button = ctk.CTkButton(
            frame,
            text="▶  詳細設定",
            command=self._toggle_recording_details,
            height=38,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["hover_soft"],
            text_color=COLORS["text"],
            anchor="w",
            font=ctk.CTkFont(FONT, 10, "bold"),
        )
        self.record_details_button.grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=18,
            pady=(0, 8),
        )

        self.record_details = ctk.CTkFrame(frame, fg_color=COLORS["card"], corner_radius=12)
        self.record_details.grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=18,
            pady=(0, 18),
        )
        self.record_details.grid_columnconfigure(1, weight=1)
        self.noise_reduction_switch = ctk.CTkSwitch(
            self.record_details,
            text="ノイズ除去",
            font=ctk.CTkFont(FONT, 10),
        )
        self.noise_reduction_switch.grid(
            row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(14, 10)
        )
        ctk.CTkLabel(
            self.record_details,
            text="感度",
            font=ctk.CTkFont(FONT, 10),
            text_color=COLORS["text"],
        ).grid(row=1, column=0, sticky="w", padx=14)
        self.sensitivity_slider = ctk.CTkSlider(
            self.record_details,
            from_=0.5,
            to=2.0,
            number_of_steps=15,
            command=self._sensitivity_changed,
        )
        self.sensitivity_slider.set(1.0)
        self.sensitivity_slider.grid(row=1, column=1, sticky="ew", padx=(8, 8))
        self.sensitivity_label = ctk.CTkLabel(
            self.record_details,
            text="100%",
            width=48,
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
        )
        self.sensitivity_label.grid(row=1, column=2, padx=(0, 14))
        self.agc_switch = ctk.CTkSwitch(
            self.record_details,
            text="自動音量調整（AGC）",
            font=ctk.CTkFont(FONT, 10),
        )
        self.agc_switch.grid(
            row=2, column=0, columnspan=3, sticky="w", padx=14, pady=10
        )

        ctk.CTkLabel(
            self.record_details,
            text="保存先",
            font=ctk.CTkFont(FONT, 10),
            text_color=COLORS["text"],
        ).grid(row=3, column=0, sticky="w", padx=14, pady=(5, 4))
        self.save_location_entry = ctk.CTkEntry(
            self.record_details,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 9),
        )
        self.save_location_entry.grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=(14, 8), pady=(0, 10)
        )
        self.save_location_entry.configure(state="disabled")
        ctk.CTkButton(
            self.record_details,
            text="変更",
            command=self.request_save_location or (lambda: None),
            width=64,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 9, "bold"),
        ).grid(row=4, column=2, padx=(0, 14), pady=(0, 10))

        ctk.CTkLabel(
            self.record_details,
            text="入力元ラベル（任意）",
            font=ctk.CTkFont(FONT, 10),
            text_color=COLORS["text"],
        ).grid(row=5, column=0, sticky="w", padx=14, pady=(5, 4))
        self.microphone_speaker_entry = ctk.CTkEntry(
            self.record_details,
            placeholder_text="マイク側（例：マイク音声）",
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 9),
        )
        self.microphone_speaker_entry.grid(
            row=6, column=0, columnspan=3, sticky="ew", padx=14, pady=(0, 8)
        )
        self.system_speaker_entry = ctk.CTkEntry(
            self.record_details,
            placeholder_text="PC側（例：PC音声）",
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 9),
        )
        self.system_speaker_entry.grid(
            row=7, column=0, columnspan=3, sticky="ew", padx=14, pady=(0, 14)
        )
        self.record_details.grid_remove()
        return frame

    def _sensitivity_changed(self, value: float) -> None:
        self.sensitivity_label.configure(text=f"{round(value * 100):d}%")

    def _toggle_recording_details(self) -> None:
        self._recording_details_visible = not self._recording_details_visible
        marker = "▼" if self._recording_details_visible else "▶"
        self.record_details_button.configure(text=f"{marker}  詳細設定")
        if self._recording_details_visible:
            self.record_details.grid()
        else:
            self.record_details.grid_remove()

    def configure_recording_sources(self, sources: list[Any], save_location: str) -> None:
        self._recording_sources = sources
        targets = []
        if any(source.kind == "microphone" for source in sources):
            targets.append("マイク")
        if any(source.kind == "loopback" for source in sources):
            targets.append("PC音声")
        targets = targets or ["入力なし"]
        self.record_target_menu.configure(values=targets)
        self.record_target_menu.set(targets[0])
        self.set_save_location(save_location)
        self._record_target_changed()

    def set_save_location(self, value: str) -> None:
        self.save_location_entry.configure(state="normal")
        self.save_location_entry.delete(0, "end")
        self.save_location_entry.insert(0, value)
        self.save_location_entry.configure(state="disabled")

    def _record_target_changed(self) -> None:
        kind = "loopback" if self.record_target_menu.get() == "PC音声" else "microphone"
        sources = [source for source in self._recording_sources if source.kind == kind]
        self._recording_source_by_label = {source.label: source for source in sources}
        labels = list(self._recording_source_by_label) or ["利用できる入力がありません"]
        self.record_device_menu.configure(values=labels)
        self.record_device_menu.set(labels[0])
        state = "normal" if sources else "disabled"
        self.record_start_button.configure(state=state)
        self.audio_test_button.configure(state=state)
        self.record_level_meter.set(0)
        self.record_level_status.configure(
            text=(
                "3秒テストで入力レベルを確認できます"
                if sources
                else "この録音対象に利用できる入力がありません"
            )
        )

    def recording_options(self) -> dict[str, Any]:
        source = self._recording_source_by_label.get(self.record_device_menu.get())
        kind = source.kind if source is not None else "microphone"
        speaker_entry = (
            self.system_speaker_entry
            if kind == "loopback"
            else self.microphone_speaker_entry
        )
        return {
            "source": source,
            "noise_reduction": bool(self.noise_reduction_switch.get()),
            "sensitivity": float(self.sensitivity_slider.get()),
            "automatic_gain_control": bool(self.agc_switch.get()),
            "speaker_label": speaker_entry.get().strip(),
        }

    def _request_record_start(self) -> None:
        if self.request_record_start is not None:
            self.request_record_start(self.recording_options())

    def _request_audio_test(self) -> None:
        if self.request_audio_test is None:
            return
        self.audio_test_button.configure(state="disabled", text="測定中…")
        self.record_level_status.configure(text="3秒間、入力音を測定しています…")
        self.request_audio_test(self.recording_options())

    def show_audio_test_result(
        self,
        result: dict[str, float | str] | None,
        error: str = "",
    ) -> None:
        self.audio_test_button.configure(state="normal", text="3秒テスト")
        if error:
            # 呼び出し側で平易な日本語に変換済みの文をそのまま表示する。
            self.record_level_meter.set(0)
            self.record_level_status.configure(text=error)
            return
        if result is None:
            return
        rms = float(result.get("rms", 0.0))
        self.record_level_meter.set(min(1.0, max(0.0, rms / 0.12)))
        # RMS / Peak の生値は生徒には見せず、言葉だけで伝える。
        self.record_level_status.configure(text=audio_level_text(result))

    def finish_importing(self) -> None:
        """取り込みの完了・失敗・中止時に進捗表示と中止ボタンを閉じる。

        ノート表示への切り替え（show_note）では閉じない: 取り込み中に
        別のノートを開いても、バックグラウンドの取り込みは続いており
        進捗と中止ボタンは見え続けるべきだから。"""
        self._importing = False
        self._refresh_processing_indicator()

    def show_note(
        self,
        note: dict[str, str],
        search_query: str = "",
    ) -> None:
        has_saved_folder = bool(note.get("_folder"))
        self._display_transcript = note["transcript"]
        self._editable_transcript = note.get(
            "editable_transcript",
            note["transcript"],
        )
        self._transcript_editable = bool(
            has_saved_folder and note.get("has_transcript")
        )
        self._speaker_lines = list(note.get("speaker_lines", []))
        if has_saved_folder:
            self.rename_button.grid()
            self.delete_button.grid()
        else:
            self.rename_button.grid_remove()
            self.delete_button.grid_remove()
        self._has_speakers = bool(
            has_saved_folder and note.get("has_speakers")
        )
        if self._has_speakers:
            self.rename_speakers_button.grid()
        else:
            self.rename_speakers_button.grid_remove()
        self._can_diarize_note = bool(
            has_saved_folder
            and note.get("has_audio")
            and note.get("has_transcript")
        )
        self._refresh_diarize_button()
        self._transcribe_available = bool(
            note.get("has_audio")
        )
        self._transcribe_is_retry = bool(note.get("has_transcript"))
        self._refresh_transcribe_button()
        self.reset_player(
            float(note.get("duration_seconds") or 0.0),
            bool(note.get("has_audio")),
        )
        self._show_common(
            title=note["title"],
            meta=note["meta"],
            keywords=note["keywords"],
            section="文字起こし",
            text=note["transcript"],
            show_player=True,
            show_right_toggle=True,
        )
        self._set_transcript_editing(False)
        self._highlight_search_matches(search_query)
        if note.get("has_audio") and note.get("has_transcript"):
            self._tag_segment_timestamps()
        self._tag_speaker_colors()

    def _set_transcript_editing(self, editing: bool) -> None:
        self._editing_transcript = editing
        editing_widgets = (
            self.cut_transcript_button,
            self.paste_transcript_button,
            self.cancel_transcript_button,
            self.save_transcript_button,
        )
        for widget in editing_widgets:
            if editing:
                widget.grid()
            else:
                widget.grid_remove()
        if editing:
            self.edit_transcript_button.grid_remove()
        elif self._transcript_editable:
            self.edit_transcript_button.grid()
        else:
            self.edit_transcript_button.grid_remove()

    def _begin_transcript_edit(self) -> None:
        if not self._transcript_editable:
            return
        self._set_transcript_editing(True)
        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", self._editable_transcript)
        self.textbox.focus_set()
        self.status_label.configure(
            text=f"編集後に「保存」を押してください。{UNDO_KEY_HINT}も利用できます。"
        )

    def _cancel_transcript_edit(self) -> None:
        replace_read_only_text(self.textbox, self._display_transcript)
        self._set_transcript_editing(False)
        self.status_label.configure(text="編集を取り消しました")

    def _save_transcript_edit(self) -> None:
        text = self.textbox.get("1.0", "end-1c").strip()
        if not text:
            self.status_label.configure(
                text="文字起こしを空にはできません"
            )
            return
        if self.request_transcript_save is None:
            return
        if self.request_transcript_save(text):
            self._editable_transcript = text
            self._set_transcript_editing(False)

    def _copy_transcript(self) -> None:
        raw = self.textbox._textbox
        try:
            text = raw.get("sel.first", "sel.last")
        except Exception:
            text = self.textbox.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_label.configure(text="クリップボードへコピーしました")

    def _highlight_search_matches(self, query: str) -> None:
        text_widget = self.textbox._textbox
        text_widget.tag_remove("search_match", "1.0", "end")
        text_widget.tag_remove("search_current", "1.0", "end")
        self._search_matches = []
        self._search_match_index = -1
        self._search_query = query.strip()
        self._set_search_navigation_visible(False)
        query = self._search_query
        if not query:
            return
        # Fixed dark text: the yellow highlight backgrounds are the same in
        # light and dark mode, so the text on them must stay dark too.
        text_widget.tag_configure(
            "search_match",
            background="#fff1a8",
            foreground="#20262d",
        )
        text_widget.tag_configure(
            "search_current",
            background="#ffc857",
            foreground="#20262d",
        )
        start = "1.0"
        while True:
            index = text_widget.search(
                query,
                start,
                stopindex="end",
                nocase=True,
            )
            if not index:
                break
            end = f"{index}+{len(query)}c"
            text_widget.tag_add("search_match", index, end)
            self._search_matches.append((index, end))
            start = end
        if self._search_matches:
            self._show_search_match(0)
        else:
            self.status_label.configure(
                text=f"検索「{query}」: ノート名または元ファイル名に一致"
            )

    def _show_previous_search_match(self) -> None:
        self._show_search_match(self._search_match_index - 1)

    def _show_next_search_match(self) -> None:
        self._show_search_match(self._search_match_index + 1)

    def _show_search_match(self, index: int) -> None:
        if not 0 <= index < len(self._search_matches):
            return
        self._search_match_index = index
        start, end = self._search_matches[index]
        text_widget = self.textbox._textbox
        text_widget.tag_remove("search_current", "1.0", "end")
        text_widget.tag_add("search_current", start, end)
        text_widget.see(start)
        text_widget.mark_set("insert", start)
        total = len(self._search_matches)
        self.status_label.configure(
            text=f"検索「{self._search_query}」: 本文内に{total}件"
        )
        self.search_position_label.configure(text=f"{index + 1} / {total}")
        self.search_previous_button.configure(
            state="normal" if index > 0 else "disabled"
        )
        self.search_next_button.configure(
            state="normal" if index < total - 1 else "disabled"
        )
        self._set_search_navigation_visible(total > 1)

    def _set_search_navigation_visible(self, visible: bool) -> None:
        widgets = (
            self.search_position_label,
            self.search_previous_button,
            self.search_next_button,
        )
        for widget in widgets:
            if visible:
                widget.grid()
            else:
                widget.grid_remove()

    def show_route(self, route: str, context: str = "") -> None:
        self._transcript_editable = False
        self._set_transcript_editing(False)
        self.rename_button.grid_remove()
        self.delete_button.grid_remove()
        self.rename_speakers_button.grid_remove()
        self.diarize_button.grid_remove()
        self._transcribe_available = False
        self._transcribe_is_retry = False
        self._refresh_transcribe_button()
        if route == "record":
            self.title_label.configure(text="新しい録音")
            self.meta_label.configure(
                text="録音対象と入力デバイスを確認してから開始します"
            )
            self.keyword_label.configure(text="  録音設定")
            self.player.grid_remove()
            self._record_bar.grid_remove()
            self.transcript_toolbar.grid_remove()
            self.textbox.grid_remove()
            self.record_setup.grid()
            self.right_toggle_button.grid()
            self.status_label.configure(
                text="詳細設定は必要なときだけ開いて変更できます"
            )
            return
        content = {
            "live": (
                "ライブ字幕",
                "入力: マイク（教室・対面）  |  日本語",
                "リアルタイム   字幕",
                "現在の発話",
                "Kの理想と本当の気持ちの違いが、今日の重要な手がかりになります。",
            ),
            "import": (
                "音声を取り込む",
                "対応形式: WAV / OGG / MP3 / M4A",
                "ファイル   文字起こし",
                "取り込み",
                "音声ファイルを選ぶと、文字起こし方法の選択画面を表示します。\n\n"
                "Step 1ではダミー画面です。",
            ),
            "search": (
                "ノートを検索",
                "左の検索欄からキーワードを入力してください",
                "タイトル   文字起こし",
                "検索結果",
                "検索結果からノートを選ぶと、該当する文字起こしを表示します。",
            ),
            "dictionary": (
                context or "辞書",
                "学習中に確認した言葉をまとめます",
                "用語   読み方   説明",
                "辞書",
                f"{context or '学習用語'}の項目をここに表示します。\n\n"
                "・音韻認識\n・視覚認知\n・合理的配慮",
            ),
            "settings": (
                context or "設定",
                "OtoWeaveの動作と表示を調整します",
                "ローカル設定",
                "設定内容",
                self._settings_text(context or "一般・表示"),
            ),
        }[route]
        self._show_common(
            title=content[0],
            meta=content[1],
            keywords=content[2],
            section=content[3],
            text=content[4],
            show_player=route in {"record", "live", "import"},
            show_right_toggle=route != "live",
        )

    def show_dictionary(
        self,
        entries: list[dict[str, Any]],
        category: str = "登録した言葉",
    ) -> None:
        self._transcript_editable = False
        self._set_transcript_editing(False)
        visible = (
            entries
            if category == "登録した言葉"
            else [
                entry
                for entry in entries
                if entry.get("category") == category
            ]
        )
        lines: list[str] = []
        for entry in visible:
            reading = (
                f"（{entry['reading']}）"
                if entry.get("reading")
                else ""
            )
            lines.append(f"● {entry['term']}{reading}")
            if entry.get("description"):
                lines.append(f"  {entry['description']}")
            if entry.get("aliases"):
                lines.append(
                    "  誤認識候補: "
                    + "、".join(entry["aliases"])
                )
            lines.append("")
        text = (
            "\n".join(lines).strip()
            if lines
            else "登録された用語はありません。\n\n"
            "左の「＋ 用語を登録・管理」から追加できます。"
        )
        self._show_common(
            title=category,
            meta="文字起こし補正とAI要約で使用するローカル辞書",
            keywords=f"登録語 {len(visible)}件",
            section="登録内容",
            text=text,
            show_player=False,
            show_right_toggle=True,
        )

    @staticmethod
    def _settings_text(context: str) -> str:
        values = {
            "一般・表示": (
                "左の設定欄からフォントと文字サイズを変更できます。\n\n"
                "設定はこの端末に保存され、次回起動時にも適用されます。"
            ),
            "システム情報": (
                "OtoWeave Step 1\n"
                "UI: CustomTkinter\n"
                "ローカル文字起こし・学習支援アプリ"
            ),
        }
        return values.get(context, values["一般・表示"])

    def show_license_info(self, text: str) -> None:
        self._transcript_editable = False
        self._set_transcript_editing(False)
        self._transcribe_available = False
        self._transcribe_is_retry = False
        self._refresh_transcribe_button()
        self.rename_button.grid_remove()
        self.delete_button.grid_remove()
        self.rename_speakers_button.grid_remove()
        self.diarize_button.grid_remove()
        self._show_common(
            title="モデルとライセンス",
            meta="現在使用するローカルAIモデルの情報",
            keywords="モデル選択ではなく、利用条件の表示です",
            section="使用モデル",
            text=text,
            show_player=False,
            show_right_toggle=True,
        )

    def _show_common(
        self,
        *,
        title: str,
        meta: str,
        keywords: str,
        section: str,
        text: str,
        show_player: bool,
        show_right_toggle: bool,
    ) -> None:
        self.record_setup.grid_remove()
        self.transcript_toolbar.grid()
        self.textbox.grid()
        self._live_text = ""
        self._live_active = False
        self.return_live_button.place_forget()
        self.title_label.configure(text=title)
        self.meta_label.configure(text=meta)
        self.keyword_label.configure(text=f"  {keywords}")
        self.section_label.configure(text=section)
        replace_read_only_text(self.textbox, text)
        self._highlight_search_matches("")
        self.status_label.configure(text="文字を選択してコピーできます")
        if show_player:
            self.player.grid()
        else:
            self.player.grid_remove()
        self._record_bar.grid_remove()
        if show_right_toggle:
            self.right_toggle_button.grid()
        else:
            self.right_toggle_button.grid_remove()

    def show_recording(self, title: str) -> None:
        self._transcript_editable = False
        self._set_transcript_editing(False)
        self._importing = False
        self._refresh_processing_indicator()
        self._show_common(
            title=title,
            meta="",
            keywords="録音中",
            section="文字起こし（リアルタイム）",
            text="",
            show_player=False,
            show_right_toggle=False,
        )
        self._elapsed_label.configure(text="00:00")
        self._record_bar.grid()
        self._live_active = True

    def show_importing(self, filename: str) -> None:
        self._transcript_editable = False
        self._set_transcript_editing(False)
        self._show_common(
            title="取り込み中",
            meta=filename,
            keywords="処理中",
            section="文字起こし",
            text="音声ファイルを処理しています…",
            show_player=False,
            show_right_toggle=False,
        )
        self._importing = True
        self._cancel_processing_pending = False
        self._refresh_processing_indicator()

    def set_live_options(self, follow: bool, highlight: bool) -> None:
        self._live_follow_enabled = follow
        self._live_highlight_enabled = highlight
        if not follow:
            self.return_live_button.place_forget()

    def apply_text_layout(self, *, spacing: int, padx: int) -> None:
        """Apply line-spacing and reading-width preferences to the body text."""
        self.textbox._textbox.configure(
            spacing2=wrapped_line_spacing(spacing),
            spacing3=spacing,
            padx=padx,
        )

    def _live_at_bottom(self) -> bool:
        try:
            return float(self.textbox.yview()[1]) >= 0.985
        except Exception:
            return True

    def update_live_transcript(self, text: str) -> None:
        at_bottom = self._live_at_bottom()
        self.textbox.configure(state="normal")
        if self._live_text and text.startswith(self._live_text):
            # Append only the new tail instead of re-rendering the whole
            # transcript (keeps long recordings O(n) instead of O(n^2)).
            self.textbox.insert("end", text[len(self._live_text):])
        else:
            self.textbox.delete("1.0", "end")
            self.textbox.insert("end", text)
        self._live_text = text
        self._apply_live_highlight(text)
        self.textbox.configure(state="disabled")
        if not self._live_follow_enabled:
            return
        if at_bottom:
            self.textbox.see("end")
            self.return_live_button.place_forget()
        elif self._live_active:
            self._show_return_to_live()

    def _apply_live_highlight(self, text: str) -> None:
        raw = self.textbox._textbox
        raw.tag_remove("current_block", "1.0", "end")
        if not self._live_highlight_enabled or not self._live_active or not text:
            return
        raw.tag_configure(
            "current_block",
            background=theme_color(COLORS["blue_soft"]),
        )
        offset = text.rfind("\n\n")
        start = f"1.0+{offset + 2}c" if offset >= 0 else "1.0"
        raw.tag_add("current_block", start, "end-1c")

    def _on_live_scroll(self, _event=None) -> None:
        if not self._live_active or not self._live_follow_enabled:
            return
        self.after_idle(self._refresh_return_to_live)

    def _refresh_return_to_live(self) -> None:
        if not self._live_active or not self._live_follow_enabled:
            return
        if self._live_at_bottom():
            self.return_live_button.place_forget()
        else:
            self._show_return_to_live()

    def _show_return_to_live(self) -> None:
        self.return_live_button.place(relx=0.5, rely=1.0, y=-64, anchor="s")
        self.return_live_button.lift()

    def _return_to_live(self) -> None:
        self.textbox.see("end")
        self.return_live_button.place_forget()

    def set_elapsed(self, elapsed: str) -> None:
        self._elapsed_label.configure(text=elapsed)

    def update_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def set_transcribing(self, transcribing: bool) -> None:
        self._transcribing = transcribing
        if not transcribing:
            self._cancel_processing_pending = False
        self._refresh_transcribe_button()
        self._refresh_processing_indicator()

    def set_transcribing_blocked(self, blocked: bool) -> None:
        self._transcribing_blocked = blocked
        self._refresh_transcribe_button()

    def _request_cancel_processing(self) -> None:
        if self.request_cancel_processing is not None:
            self.request_cancel_processing()

    def set_cancel_processing_pending(self) -> None:
        """「中止」ボタンを押した後、完了イベントが届くまでの表示。"""
        self._cancel_processing_pending = True
        self._refresh_processing_indicator()

    def _refresh_processing_indicator(self) -> None:
        state = processing_indicator_state(
            self._transcribing,
            self._importing,
            self._cancel_processing_pending,
        )
        if state["visible"]:
            self.processing_bar.grid()
            self.processing_bar.start()
            self.cancel_processing_button.configure(
                state=state["button_state"],
                text=state["button_text"],
            )
            self.cancel_processing_button.grid()
        else:
            self._cancel_processing_pending = False
            self.processing_bar.stop()
            self.processing_bar.grid_remove()
            self.cancel_processing_button.grid_remove()

    def _refresh_transcribe_button(self) -> None:
        if not self._transcribe_available:
            self.transcribe_button.grid_remove()
            return
        self.transcribe_button.grid()
        busy = self._transcribing or self._transcribing_blocked
        self.transcribe_button.configure(
            state="disabled" if busy else "normal",
            text=(
                "処理中…"
                if self._transcribing_blocked
                else (
                    "文字起こし中…"
                    if self._transcribing
                    else (
                        "文字起こしをやり直す"
                        if self._transcribe_is_retry
                        else "文字起こしを開始"
                    )
                )
            ),
        )

    def set_empty_note(self) -> None:
        self._transcript_editable = False
        self._set_transcript_editing(False)
        self._importing = False
        self._refresh_processing_indicator()
        self.rename_button.grid_remove()
        self.delete_button.grid_remove()
        self.rename_speakers_button.grid_remove()
        self.diarize_button.grid_remove()
        self._transcribe_available = False
        self._transcribe_is_retry = False
        self._refresh_transcribe_button()
        self._show_common(
            title="ノートはまだありません",
            meta="録音または音声ファイルの取り込みから始められます",
            keywords="",
            section="",
            text="",
            show_player=False,
            show_right_toggle=True,
        )

    def set_right_visible(self, visible: bool) -> None:
        self.right_toggle_button.configure(
            text="💡 右パネルを隠す" if visible else "💡 右パネルを表示"
        )

    @staticmethod
    def _parse_leading_timestamp(line: str) -> float | None:
        match = re.match(r"^(\d{1,3}):(\d{2})\b", line)
        if not match:
            return None
        return float(int(match.group(1)) * 60 + int(match.group(2)))

    def _tag_segment_timestamps(self) -> None:
        """Underline leading MM:SS stamps so they act as replay links."""
        raw = self.textbox._textbox
        raw.tag_remove("seg_time", "1.0", "end")
        if self.request_segment_play is None:
            return
        raw.tag_configure(
            "seg_time",
            underline=True,
            foreground=theme_color(COLORS["blue"]),
        )
        total_lines = int(raw.index("end-1c").split(".")[0])
        for lineno in range(1, total_lines + 1):
            line = raw.get(f"{lineno}.0", f"{lineno}.end")
            match = re.match(r"^\d{1,3}:\d{2}\b", line)
            if match:
                raw.tag_add("seg_time", f"{lineno}.0", f"{lineno}.{match.end()}")

    def _on_timestamp_click(self, event) -> str:
        raw = self.textbox._textbox
        index = raw.index(f"@{event.x},{event.y}")
        lineno = index.split(".")[0]
        seconds = self._parse_leading_timestamp(
            raw.get(f"{lineno}.0", f"{lineno}.end")
        )
        if seconds is not None and self.request_segment_play is not None:
            self.request_segment_play(seconds)
        return "break"

    def _tag_speaker_colors(self) -> None:
        """Color each line's leading "Speaker:" prefix by speaker.

        Color is a supplementary cue only: the prefix text itself is
        always left in place, and the prefix is bolded too so the
        distinction doesn't rely on color alone. Speakers are matched
        against `self._speaker_lines` (one entry per transcript segment,
        in order, "" when the segment has no speaker) rather than by
        parsing the displayed text, so renamed speakers or names that
        happen to contain a colon still tag correctly.
        """
        raw = self.textbox._textbox
        tag_names = [f"speaker_color_{index}" for index in range(len(SPEAKER_COLOR_KEYS))]
        for tag in tag_names:
            raw.tag_remove(tag, "1.0", "end")
        speakers = self._speaker_lines
        if not speakers or not any(speakers):
            return

        order: list[str] = []
        for name in speakers:
            if name and name not in order:
                order.append(name)
        color_index = {name: i % len(SPEAKER_COLOR_KEYS) for i, name in enumerate(order)}

        base_font = self.textbox.cget("font")
        try:
            family = base_font.cget("family")
            size = base_font.cget("size")
        except Exception:
            family, size = FONT, 12
        for index, tag in enumerate(tag_names):
            raw.tag_configure(
                tag,
                foreground=theme_color(COLORS[SPEAKER_COLOR_KEYS[index]]),
                font=(family, size, "bold"),
            )

        total_lines = int(raw.index("end-1c").split(".")[0])
        segment_index = 0
        for lineno in range(1, total_lines + 1):
            if segment_index >= len(speakers):
                break
            line = raw.get(f"{lineno}.0", f"{lineno}.end")
            timestamp_match = re.match(r"^\d{1,3}:\d{2}\s*", line)
            if not timestamp_match:
                continue
            name = speakers[segment_index]
            segment_index += 1
            if not name:
                continue
            prefix = f"{name}:"
            start_col = timestamp_match.end()
            if line[start_col:start_col + len(prefix)] != prefix:
                continue
            tag = tag_names[color_index[name]]
            raw.tag_add(
                tag,
                f"{lineno}.{start_col}",
                f"{lineno}.{start_col + len(prefix)}",
            )

    def refresh_transcript_tags(self) -> None:
        """Re-apply timestamp/speaker tag colors and fonts to the current
        transcript. Plain tkinter Text tag colors don't auto-update on a
        theme or font-size change (unlike CTk widget colors), so this is
        called again after display preferences change."""
        self._tag_segment_timestamps()
        self._tag_speaker_colors()

    def _request_speak_transcript(self) -> None:
        if self.request_speak_transcript is None:
            return
        text = self._editable_transcript or self._display_transcript
        self.request_speak_transcript(text)

    def _request_speaker_rename(self) -> None:
        if self.request_speaker_rename is not None:
            self.request_speaker_rename()

    def _request_diarize(self) -> None:
        if self.request_diarize is not None:
            self.request_diarize()

    def set_diarization_available(self, available: bool) -> None:
        """Whether the diarization models are present on this machine
        (a one-time, system-wide fact set at startup), independent of
        whichever note happens to be open."""
        self._diarization_available = bool(available)
        self._refresh_diarize_button()

    def set_diarizing(self, diarizing: bool) -> None:
        self._diarizing = diarizing
        self._refresh_diarize_button()

    def _refresh_diarize_button(self) -> None:
        if self._diarization_available and self._can_diarize_note:
            self.diarize_button.grid()
        else:
            self.diarize_button.grid_remove()
        self.diarize_button.configure(
            state="disabled" if self._diarizing else "normal",
            text="話者を推定中…" if self._diarizing else "話者を推定",
        )

    def set_speaking_transcript(self, speaking: bool) -> None:
        self.speak_transcript_button.configure(
            text="⏹ 停止" if speaking else "🔊 読み上げ"
        )

    @staticmethod
    def _format_clock(seconds: float) -> str:
        secs = max(0, int(seconds))
        h, m = divmod(secs, 3600)
        m, s = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _request_play_toggle(self) -> None:
        if self.request_play_toggle is not None:
            self.request_play_toggle()

    def reset_player(self, duration_seconds: float, has_audio: bool) -> None:
        self._player_duration = max(0.0, float(duration_seconds or 0.0))
        self.player_position_label.configure(text="00:00")
        self.player_duration_label.configure(
            text=self._format_clock(self._player_duration)
            if self._player_duration > 0
            else "--:--"
        )
        self.player_progress.set(0.0)
        self.play_button.configure(
            text="▶",
            state="normal" if has_audio else "disabled",
        )

    def set_player_playing(self, playing: bool) -> None:
        self.play_button.configure(text="Ⅱ" if playing else "▶")

    def set_player_progress(self, position: float, duration: float) -> None:
        total = duration if duration > 0 else self._player_duration
        self.player_position_label.configure(text=self._format_clock(position))
        if total > 0:
            self.player_progress.set(min(1.0, max(0.0, position / total)))


class RightPane(ctk.CTkFrame):
    def __init__(
        self,
        parent,
        run_background: BackgroundRunner,
        on_chat: Callable[[str], None] | None = None,
        on_summarize: Callable[[str], None] | None = None,
        on_cancel_summarize: Callable[[], None] | None = None,
        on_manage_templates: Callable[[], None] | None = None,
        on_focus_toggle: Callable[[], None] | None = None,
        on_template_selected: Callable[[str], None] | None = None,
        on_speak_summary: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(
            parent,
            fg_color=COLORS["right"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["line"],
        )
        self.run_background = run_background
        self._on_chat = on_chat
        self._on_summarize = on_summarize
        self._on_cancel_summarize = on_cancel_summarize
        self._on_manage_templates = on_manage_templates
        self._on_focus_toggle = on_focus_toggle
        self._on_template_selected = on_template_selected
        self._on_speak_summary = on_speak_summary
        self._template_ids_by_name: dict[str, str] = {}
        self._chat_row = 0
        self._thinking = False
        self._stream_label: ctk.CTkLabel | None = None
        self._stream_text = ""
        self._summarizing = False
        self._summary_text = ""
        # AI要約（生成）の可用性。False のとき生成系UIを隠して案内を出す。
        # set_summarize_available() で起動時に更新される。
        self._summarize_available = True
        self._note_controls_visible = False
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        title_header = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        title_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))
        title_header.grid_columnconfigure(0, weight=1)
        self.title_label = ctk.CTkLabel(
            title_header,
            text="要約とAIチューター",
            font=ctk.CTkFont(FONT, 18, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        )
        self.title_label.grid(row=0, column=0, sticky="ew")
        self.focus_button = ctk.CTkButton(
            title_header,
            text="広く表示",
            command=on_focus_toggle or (lambda: None),
            width=76,
            height=32,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["hover"],
            text_color=COLORS["muted"],
            border_width=1,
            border_color=COLORS["line"],
            font=ctk.CTkFont(FONT, 9, "bold"),
        )
        self.focus_button.grid(row=0, column=1, padx=(8, 0))
        self.summary_header = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        self.summary_header.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 8),
        )
        self.summary_header.grid_columnconfigure(0, weight=1)
        self.summary_label = ctk.CTkLabel(
            self.summary_header,
            text="全体の要約",
            font=ctk.CTkFont(FONT, 12, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        )
        self.summary_label.grid(row=0, column=0, sticky="ew")
        self.speak_summary_button = ctk.CTkButton(
            self.summary_header,
            text="🔊 読み上げ",
            command=self._request_speak_summary,
            width=92,
            height=34,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["hover"],
            text_color=COLORS["muted"],
            border_width=1,
            border_color=COLORS["line"],
            font=ctk.CTkFont(FONT, 9),
        )
        self.speak_summary_button.grid(row=0, column=1, padx=(8, 0))
        self.summarize_button = ctk.CTkButton(
            self.summary_header,
            text="要約を作成",
            command=self._request_summarize,
            width=96,
            height=34,
            corner_radius=8,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            font=ctk.CTkFont(FONT, 10, "bold"),
            state="normal" if on_summarize is not None else "disabled",
        )
        self.summarize_button.grid(row=0, column=2, padx=(8, 0))
        self.template_menu = ctk.CTkOptionMenu(
            self.summary_header,
            values=["授業の要点"],
            command=self._template_selected,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 9),
        )
        self.template_menu.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        self.manage_templates_button = ctk.CTkButton(
            self.summary_header,
            text="テンプレート管理",
            command=on_manage_templates or (lambda: None),
            width=112,
            height=34,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["hover"],
            text_color=COLORS["muted"],
            border_width=1,
            border_color=COLORS["line"],
            font=ctk.CTkFont(FONT, 9),
        )
        self.manage_templates_button.grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="e",
            padx=(8, 0),
            pady=(8, 0),
        )
        # AI要約が使えない環境で、生成ボタンの代わりに出す案内。
        # テンプレート選択と同じ行を使う（同時に表示されることはない）。
        self.summary_unavailable_label = ctk.CTkLabel(
            self.summary_header,
            text=SUMMARY_UNAVAILABLE_MESSAGE,
            font=ctk.CTkFont(FONT, 10),
            text_color=COLORS["muted"],
            anchor="w",
            justify="left",
        )
        self.summary_unavailable_label.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(8, 0),
        )
        self.summary_unavailable_label.grid_remove()
        ctk.CTkLabel(
            self.summary_header,
            text="⚠ AI要約・チャットはβ版です。大事な内容は本文と音声で確認してください。",
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
            anchor="w",
        ).grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(6, 0),
        )
        # 要約の実行中だけ表示する進捗ラベルと不確定プログレスバー。
        # summary_progress イベントでラベルの文言が更新される。
        self.summary_progress_label = ctk.CTkLabel(
            self.summary_header,
            text="要約を作成中…",
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
            anchor="w",
        )
        self.summary_progress_label.grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(8, 2),
        )
        self.summary_progress_bar = ctk.CTkProgressBar(
            self.summary_header,
            height=6,
            corner_radius=3,
            mode="indeterminate",
            progress_color=COLORS["blue"],
            fg_color=COLORS["line"],
        )
        self.summary_progress_bar.grid(
            row=4,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(0, 2),
        )
        self.summary_progress_label.grid_remove()
        self.summary_progress_bar.grid_remove()
        self.summary_box = ctk.CTkTextbox(
            self,
            height=210,
            wrap="word",
            fg_color=COLORS["surface"],
            text_color=COLORS["summary_text"],
            border_width=1,
            border_color=COLORS["box_border"],
            corner_radius=8,
            font=ctk.CTkFont(FONT, 10),
            spacing2=wrapped_line_spacing(7),
            spacing3=7,
            padx=14,
            pady=12,
            cursor="xterm",
        )
        self.summary_box.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 18))
        self.chat_label = ctk.CTkLabel(
            self,
            text="文字起こしについて質問する",
            font=ctk.CTkFont(FONT, 12, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        )
        self.chat_label.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 8))
        self.chat_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=COLORS["surface"],
            corner_radius=8,
            border_width=1,
            border_color=COLORS["box_border"],
            scrollbar_button_color="#c2c9d0",
            scrollbar_button_hover_color="#aab3bc",
        )
        self.chat_frame.grid(row=4, column=0, sticky="nsew", padx=20, pady=(0, 10))
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self._add_chat_message(
            "AIチューター",
            "文字起こしについて、分からないところを質問できます。",
            is_user=False,
        )
        self.entry_row = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.entry_row.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 18))
        self.entry_row.grid_columnconfigure(0, weight=1)
        self.chat_entry = ctk.CTkEntry(
            self.entry_row,
            placeholder_text="分からないところを入力",
            height=40,
            corner_radius=8,
            border_color=COLORS["input_border"],
            font=ctk.CTkFont(FONT, 10),
        )
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.chat_entry.bind("<Return>", self._request_chat_from_event)
        self._send_btn = ctk.CTkButton(
            self.entry_row,
            text="送信",
            width=68,
            height=40,
            corner_radius=8,
            command=self._request_chat,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            font=ctk.CTkFont(FONT, 10, "bold"),
        )
        self._send_btn.grid(row=0, column=1)

    def show_note(self, note: dict[str, str]) -> None:
        self.title_label.configure(text="要約とAIチューター")
        self.summary_label.configure(text="全体の要約")
        self._summary_text = str(note["summary"])
        replace_read_only_text(self.summary_box, note["summary"])
        self._note_controls_visible = True
        self.speak_summary_button.grid()
        self.focus_button.grid()
        self._apply_summary_controls()
        self.summarize_button.configure(text="要約を作成")
        self._set_chat_visible(True)

    def set_empty_note(self) -> None:
        self.title_label.configure(text="要約とAIチューター")
        self.summary_label.configure(text="全体の要約")
        replace_read_only_text(
            self.summary_box,
            "ノートを選択すると、要約と質問機能を利用できます。",
        )
        self._summary_text = ""
        self._note_controls_visible = False
        self.speak_summary_button.grid_remove()
        self.summarize_button.grid_remove()
        self.template_menu.grid_remove()
        self.manage_templates_button.grid_remove()
        self.summary_unavailable_label.grid_remove()
        self._set_chat_visible(False)

    def show_route(self, route: str, context: str = "") -> None:
        content = {
            "record": (
                "録音サポート",
                "録音設定",
                "入力デバイスと録音状態をここに表示します。",
            ),
            "import": (
                "取り込みサポート",
                "ファイル情報",
                "選択した音声ファイルの長さと形式をここに表示します。",
            ),
            "search": (
                "検索のヒント",
                "検索対象",
                "タイトル、キーワード、文字起こし本文から検索します。",
            ),
            "dictionary": (
                "辞書",
                context or "用語の説明",
                "選択した言葉の読み方と、学習者向けの短い説明を表示します。",
            ),
            "settings": (
                "設定ガイド",
                context or "設定内容",
                "設定はこの端末だけに保存され、外部へ送信されません。",
            ),
        }[route]
        self.title_label.configure(text=content[0])
        self.summary_label.configure(text=content[1])
        replace_read_only_text(self.summary_box, content[2])
        self._summary_text = ""
        self._note_controls_visible = False
        self.speak_summary_button.grid_remove()
        self.summarize_button.grid_remove()
        self.template_menu.grid_remove()
        self.manage_templates_button.grid_remove()
        self.summary_unavailable_label.grid_remove()
        self.focus_button.grid_remove()
        self._set_chat_visible(False)

    def set_summarize_available(self, available: bool) -> None:
        """AI要約（生成）の可用性を反映する。

        使えない環境では「要約を作成」「テンプレート選択」「テンプレート
        管理」を隠し、平易な案内文を表示する。キャッシュ済み要約の表示・
        読み上げとチャットはそのまま使える。"""
        self._summarize_available = bool(available)
        if self._note_controls_visible:
            self._apply_summary_controls()

    def _apply_summary_controls(self) -> None:
        """ノート表示中の要約生成系UIを可用性マップに従って出し分ける。"""
        visibility = summary_controls_visibility(self._summarize_available)
        widgets = {
            "summarize_button": self.summarize_button,
            "template_menu": self.template_menu,
            "manage_templates_button": self.manage_templates_button,
            "unavailable_notice": self.summary_unavailable_label,
        }
        for key, widget in widgets.items():
            if visibility[key]:
                widget.grid()
            else:
                widget.grid_remove()

    def _request_summarize(self) -> None:
        if not self._summarize_available:
            # 非表示のはずのボタン経由で万一呼ばれても何もしない防御。
            return
        if self._summarizing:
            if self._on_cancel_summarize is not None:
                self._on_cancel_summarize()
                self.summarize_button.configure(
                    state="disabled",
                    text="キャンセル中…",
                )
            return
        if self._on_summarize is not None:
            selected_name = self.template_menu.get()
            self._on_summarize(
                self._template_ids_by_name.get(selected_name, "lesson_record")
            )

    def _template_selected(self, selected_name: str) -> None:
        if self._on_template_selected is not None:
            self._on_template_selected(
                self._template_ids_by_name.get(
                    selected_name,
                    "lesson_record",
                )
            )

    def set_templates(
        self,
        templates: list[dict[str, Any]],
        selected_id: str = "",
        statuses: dict[str, str] | None = None,
    ) -> None:
        statuses = statuses or {}
        markers = {
            "generated": "✅",
            "stale": "⚠",
            "missing": "○",
        }
        self._template_ids_by_name = {
            f"{markers.get(statuses.get(str(value['id']), ''), '○')} "
            f"{value['name']}": str(value["id"])
            for value in templates
        }
        names = list(self._template_ids_by_name) or ["授業の要点"]
        self.template_menu.configure(values=names)
        selected_name = next(
            (
                name
                for name, template_id in self._template_ids_by_name.items()
                if template_id == selected_id
            ),
            names[0],
        )
        self.template_menu.set(selected_name)

    def set_focus_mode(self, focused: bool) -> None:
        self.focus_button.configure(
            text="元に戻す" if focused else "広く表示"
        )

    def set_summarizing(self, summarizing: bool) -> None:
        self._summarizing = summarizing
        if summarizing:
            cancellable = self._on_cancel_summarize is not None
            self.summarize_button.configure(
                state="normal" if cancellable else "disabled",
                text="キャンセル" if cancellable else "要約中…",
            )
            self.summary_progress_label.configure(text="要約を作成中…")
            self.summary_progress_label.grid()
            self.summary_progress_bar.grid()
            self.summary_progress_bar.start()
        else:
            # 完了・エラー・キャンセルのどの経路でも必ずここを通って
            # 進捗表示を閉じる（llm_summary_done / llm_error / llm_cancelled）。
            self.summary_progress_bar.stop()
            self.summary_progress_bar.grid_remove()
            self.summary_progress_label.grid_remove()
            self.summarize_button.configure(
                state="normal",
                text="要約を作成",
            )

    def set_summary_progress(self, text: str) -> None:
        """summary_progress イベントの表示文字列を進捗ラベルへ反映する。"""
        if not self._summarizing:
            # 完了・キャンセル後に遅れて届いた進捗は無視する。
            return
        self.summary_progress_label.configure(text=str(text))

    def _set_chat_visible(self, visible: bool) -> None:
        widgets = (self.chat_label, self.chat_frame, self.entry_row)
        for widget in widgets:
            if visible:
                widget.grid()
            else:
                widget.grid_remove()

    def _request_chat_from_event(self, _event=None) -> str:
        self._request_chat()
        return "break"

    def _request_chat(self) -> None:
        question = self.chat_entry.get().strip()
        if not question or self._thinking:
            return
        self.chat_entry.delete(0, "end")
        self._add_chat_message("あなた", question, is_user=True)
        if self._on_chat is not None:
            self.set_thinking(True)
            self._on_chat(question)
        else:
            def worker() -> str:
                time.sleep(0.2)
                return "Step 1のダミー回答です。バックエンド接続後は文字起こしを参照します。"
            self.run_background(worker, self.append_answer)

    def clear_chat(self) -> None:
        """Remove all chat messages and reset the thinking state."""
        self.set_thinking(False)
        self._stream_label = None
        self._stream_text = ""
        for child in self.chat_frame.winfo_children():
            child.destroy()
        self._chat_row = 0
        self._add_chat_message(
            "AIチューター",
            "文字起こしについて、分からないところを質問できます。",
            is_user=False,
        )

    def set_thinking(self, thinking: bool) -> None:
        self._thinking = thinking
        state = "disabled" if thinking else "normal"
        self.chat_entry.configure(state=state)
        self._send_btn.configure(
            state=state,
            text="…" if thinking else "送信",
        )

    def append_answer(self, text: str) -> None:
        self.set_thinking(False)
        if self._stream_label is not None:
            # ストリーミング中の吹き出しを最終回答の全文で置き換える
            # （途中経過との重複を防ぐ）。
            label = self._stream_label
            self._stream_label = None
            self._stream_text = ""
            label.configure(text=text)
            self._scroll_chat_to_bottom()
            return
        self._add_chat_message("AIチューター", text, is_user=False)

    def append_answer_chunk(self, text: str) -> None:
        """ストリーミング中のAI回答の吹き出しへ差分テキストを追記する。"""
        if not self._thinking:
            # 質問中でなければ（ノート切替後の遅延チャンク等）無視する。
            return
        if self._stream_label is None:
            self._stream_text = ""
            self._stream_label = self._add_chat_message(
                "AIチューター", "", is_user=False
            )
        self._stream_text += str(text)
        self._stream_label.configure(text=self._stream_text)
        self._scroll_chat_to_bottom()

    def _scroll_chat_to_bottom(self) -> None:
        self.after_idle(
            lambda: self.chat_frame._parent_canvas.yview_moveto(1.0)
        )

    def apply_text_layout(self, *, spacing: int) -> None:
        """Apply the line-spacing preference to the summary text."""
        self.summary_box._textbox.configure(
            spacing2=wrapped_line_spacing(spacing),
            spacing3=spacing,
        )

    def _request_speak_summary(self) -> None:
        if self._on_speak_summary is not None:
            self._on_speak_summary(self._summary_text)

    def set_speaking_summary(self, speaking: bool) -> None:
        self.speak_summary_button.configure(
            text="⏹ 停止" if speaking else "🔊 読み上げ"
        )

    def set_summary(self, text: str, status: str = "") -> None:
        self._summary_text = str(text)
        replace_read_only_text(self.summary_box, text)
        if status == "generated":
            self.summary_label.configure(text="保存済みの要約")
            self.summarize_button.configure(text="再生成")
        elif status == "stale":
            self.summary_label.configure(text="以前の要約（更新あり）")
            self.summarize_button.configure(text="更新する")
        else:
            self.summary_label.configure(text="要約")
            self.summarize_button.configure(text="要約を作成")

    def _add_chat_message(
        self, author: str, text: str, is_user: bool
    ) -> ctk.CTkLabel:
        message = ctk.CTkFrame(
            self.chat_frame,
            fg_color=COLORS["chat_user"] if is_user else COLORS["chat_ai"],
            corner_radius=8,
        )
        message.grid(
            row=self._chat_row,
            column=0,
            sticky="ew",
            padx=(28, 6) if is_user else (6, 28),
            pady=5,
        )
        message.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            message,
            text=author,
            font=ctk.CTkFont(FONT, 9, "bold"),
            text_color=COLORS["link_text"] if is_user else COLORS["green"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        body = ctk.CTkLabel(
            message,
            text=text,
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["text"],
            justify="left",
            anchor="w",
            wraplength=220,
        )
        body.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        self._chat_row += 1
        self.after_idle(
            lambda: self.chat_frame._parent_canvas.yview_moveto(1.0)
        )
        return body
