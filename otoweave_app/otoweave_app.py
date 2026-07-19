from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter.font as tkfont
from collections.abc import Callable, Mapping
from datetime import date as date_cls
from pathlib import Path
from tkinter import Menu, filedialog, messagebox
from typing import Any

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD

from otoweave_app.app_logging import (
    default_log_dir,
    get_logger,
    log_exception,
    setup_logging,
)
from otoweave_app.controller import LearningAccessController
from otoweave_app.diarization import diarization_available
from otoweave_app.display_settings import (
    DisplaySettings,
    available_reading_fonts,
    load_display_settings,
    save_display_settings,
)
from otoweave_app.customtkinter_mock_data import NOTE_BY_ID, NOTES
from otoweave_app.customtkinter_views import (
    COLORS,
    FONT,
    BASE_FONT_SIZE,
    SUMMARY_UNAVAILABLE_MESSAGE,
    ActivityBar,
    DetailPane,
    MainPane,
    RightPane,
)
from otoweave_app.models import LessonRecord
from otoweave_app.model_catalog import model_disclosures
from otoweave_app.segment_editing import rename_speaker
from otoweave_app.storage import LessonStore
from otoweave_app.summary_templates import (
    load_templates,
    normalize_template,
    save_custom_templates,
    template_by_id,
    templates_path,
)
from otoweave_app.summary_cache import (
    activate_cached_summary,
    inspect_cached_summary,
)
from otoweave_app.user_dictionary import (
    CATEGORIES as DICTIONARY_CATEGORIES,
    dictionary_path,
    glossary_prompt,
    load_dictionary,
    normalize_entry,
    save_dictionary,
)
from otoweave_app.tts import (
    WindowsTts,
    cleanup_stale_tts_files,
    tts_temp_dir,
)
from otoweave_app.windows_process import decode_windows_process_output


ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


DEFAULT_WINDOW_SIZE = (1440, 860)
DEFAULT_MIN_WINDOW_SIZE = (1120, 680)

CLOUD_SYNC_ENV_VARS = ("OneDrive", "OneDriveCommercial", "OneDriveConsumer")
CLOUD_SYNC_KEYWORDS = (
    "onedrive",
    "googledrive",
    # Google Drive for Desktop の標準マウント名（C:\Users\x\Google Drive、
    # G:\My Drive）も検知できるよう、スペース入り表記を含める。
    "google drive",
    "google ドライブ",
    "my drive",
    "マイドライブ",
    "dropbox",
)


def initial_window_geometry(
    screen_width: int,
    screen_height: int,
) -> tuple[int, int, int, int]:
    """初期ウィンドウサイズを画面内に収める純関数。

    1366x768 の GIGA 端末でも下部ステータス行が隠れないよう、既定の
    1440x860 を画面サイズでクランプする。戻り値は
    (width, height, min_width, min_height)。
    """
    width = max(640, min(DEFAULT_WINDOW_SIZE[0], screen_width - 40))
    height = max(480, min(DEFAULT_WINDOW_SIZE[1], screen_height - 100))
    min_width = min(DEFAULT_MIN_WINDOW_SIZE[0], width)
    min_height = min(DEFAULT_MIN_WINDOW_SIZE[1], height)
    return width, height, min_width, min_height


def _path_is_under(path: str, root: str) -> bool:
    """path が root 配下（root 自身を含む）かどうかを大文字小文字を無視して判定する。"""
    try:
        norm_path = os.path.normcase(os.path.normpath(str(path)))
        norm_root = os.path.normcase(os.path.normpath(str(root)))
    except (TypeError, ValueError):
        return False
    if not norm_root or norm_root == ".":
        return False
    if norm_path == norm_root:
        return True
    return norm_path.startswith(norm_root.rstrip("\\/") + os.sep)


def is_cloud_synced_path(
    path: Path | str,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """保存先パスがクラウド同期フォルダ配下と思われるかを判定する純関数。

    OneDrive の既知フォルダリダイレクト（環境変数 OneDrive /
    OneDriveCommercial / OneDriveConsumer のパス配下）と、パス文字列に
    含まれる同期サービス名から推定する。
    """
    env = os.environ if environ is None else environ
    text = str(path)
    for name in CLOUD_SYNC_ENV_VARS:
        root = str(env.get(name, "") or "").strip()
        if root and _path_is_under(text, root):
            return True
    lowered = text.casefold()
    return any(keyword in lowered for keyword in CLOUD_SYNC_KEYWORDS)


def cloud_sync_notice_path(data_root: Path | str) -> Path:
    """「クラウド同期の警告を表示済み」フラグの保存先。"""
    return Path(data_root) / "cloud_sync_notice.json"


def load_cloud_sync_notice_shown(path: Path | str) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("warned") is True


def save_cloud_sync_notice_shown(path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"warned": True}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


# 要約サブプロセスが報告する進捗ステージ → 生徒向けの表示文字列。
# 未知のステージが増えても壊れないよう、表示側は必ずフォールバックを使う。
SUMMARY_STAGE_LABELS = {
    "load": "AIを準備しています…",
    "clean": "文字起こしを整えています…",
    "part": "要約を作っています…",
    "compact": "要約を短くまとめています…",
    "merge": "要約を仕上げています…",
}
SUMMARY_PROGRESS_FALLBACK = "要約を処理中…"

# アプリ終了時にバックグラウンド保存を待つとき、この秒数を超えたら
# 「まだ動いている」ことが分かる文言に切り替える。
CLOSE_WAIT_NOTICE_SECONDS = 15.0

# 技術的な詳細をログへ逃がしたときに、ダイアログへ添える案内文。
FRIENDLY_ERROR_LOG_NOTE = "くわしい記録はログに保存されています。"


def summary_progress_text(progress: Any) -> str:
    """summary_progress イベントの progress dict を表示文字列へ変換する純関数。

    {"stage": "part", "current": 2, "total": 5} → 「要約を作っています… (2/5)」。
    未知の stage・欠けたキー・型違いでも例外を出さずフォールバックする。
    """
    if not isinstance(progress, Mapping):
        return SUMMARY_PROGRESS_FALLBACK
    stage = str(progress.get("stage", ""))
    label = SUMMARY_STAGE_LABELS.get(stage, SUMMARY_PROGRESS_FALLBACK)
    try:
        current = int(progress.get("current"))
        total = int(progress.get("total"))
    except (TypeError, ValueError):
        return label
    if current > 0 and total > 0:
        return f"{label} ({current}/{total})"
    return label


def summary_error_status(message: Any) -> str:
    """要約失敗メッセージを、生徒に見せられるステータス文へ変換する純関数。

    自前で組み立てた平易な日本語（タイムアウト文言）はそのまま通し、
    スタックトレース等を含み得る技術的なメッセージは固定の平易文に
    置き換える（元の文字列はログに残す前提）。
    """
    text = str(message).strip()
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.startswith("要約がタイムアウトしました"):
        return f"⚠ {first_line}"
    return "⚠ 要約を作れませんでした。もう一度試してください。"


def summary_display_text(
    status: str,
    text: str,
    summarize_available: bool = True,
) -> str:
    """キャッシュ済み要約の状態 → 要約欄に表示する本文（純関数）。

    AI要約の生成が使えない環境（Lite版・メモリの少ない端末）でも、
    キャッシュ済みの要約はそのまま閲覧できる。生成ボタンが無い環境で
    「押してください」と案内しないよう、可用性で文言を切り替える。
    """
    if status == "missing":
        if not summarize_available:
            return "（" + SUMMARY_UNAVAILABLE_MESSAGE.replace("\n", "") + "）"
        return (
            "（このテンプレートの要約はまだありません。"
            "「要約を作成」を押してください。）"
        )
    if status == "stale":
        notice = (
            "⚠ この要約は以前の文字起こし・テンプレート・"
            "辞書から生成されています。\n"
        )
        if summarize_available:
            notice += "必要に応じて「更新する」を押してください。\n"
        return notice + "\n" + text
    return text


def close_wait_status(waited_seconds: float) -> str:
    """終了待ち中に表示するステータス文言（純関数）。"""
    if waited_seconds >= CLOSE_WAIT_NOTICE_SECONDS:
        return "保存を続けています。もう少しお待ちください…"
    return "保存中です。しばらくお待ちください…"


def build_friendly_error(
    title: str,
    message: str,
    exc: BaseException | None = None,
    detail: str = "",
) -> str:
    """技術情報をログにだけ記録し、生徒向けダイアログ本文を組み立てる。

    例外メッセージ・スタックトレース・英語のエラー文は画面に出さない。
    ログへ記録できた場合のみ「くわしい記録はログに…」の案内を添える。
    """
    logged = False
    if exc is not None:
        log_exception(f"操作に失敗しました（{title}）", exc)
        logged = True
    elif detail:
        try:
            get_logger().error("操作に失敗しました（%s）: %s", title, detail)
            logged = True
        except Exception:
            pass
    if logged:
        return f"{message}\n\n{FRIENDLY_ERROR_LOG_NOTE}"
    return message


def _note_label_and_date(title: str, date_value: str) -> tuple[str, str]:
    try:
        d = date_cls.fromisoformat(date_value)
        return f"{d.month}/{d.day}  {title}", f"{d.year}年{d.month}月{d.day}日"
    except (ValueError, AttributeError):
        return title, date_value


def _format_duration(seconds: float) -> str:
    if seconds <= 0:
        return ""
    secs = int(seconds)
    h, m = divmod(secs, 3600)
    m, s = divmod(m, 60)
    return "  |  " + (f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}")


def _lesson_to_note(folder: Path, lesson: LessonRecord) -> dict:
    """Convert a fully loaded LessonRecord to the note dict used by views."""
    title = lesson.suggested_title or lesson.title
    label, date_str = _note_label_and_date(title, lesson.date)
    duration = _format_duration(
        max((seg.end for seg in lesson.segments), default=0.0)
    )

    tags: list[str] = []
    if any(seg.important for seg in lesson.segments):
        tags.append("★重要")
    if any(seg.question for seg in lesson.segments):
        tags.append("？質問")
    keywords = "  ".join(tags) if tags else ("文字起こし済み" if lesson.segments else "文字起こしなし")
    if lesson.status == "needs_repair":
        keywords = "⚠ 要修復（記録の一部を読み取れません）"

    lines: list[str] = []
    for seg in lesson.segments:
        m2, s2 = divmod(int(seg.start), 60)
        speaker = f"{seg.speaker}: " if seg.speaker else ""
        lines.append(f"{m2:02d}:{s2:02d}  {speaker}{seg.text}")
    transcript = "\n\n".join(lines) if lines else "（文字起こしデータがありません）"

    return {
        "id": lesson.lesson_id,
        "label": label,
        "title": title,
        "meta": (
            date_str
            + duration
            + (
                f"\n元ファイル: {lesson.source_audio_name}"
                if lesson.source_audio_name
                else ""
            )
        ),
        "keywords": keywords,
        "transcript": transcript,
        "editable_transcript": "\n\n".join(
            segment.text for segment in lesson.segments
        ),
        "summary": "（読み込み中…）",
        "source_audio_name": lesson.source_audio_name,
        "has_transcript": bool(lesson.segments),
        # One entry per segment, in display order, "" when no speaker
        # label — used to color-tag the transcript's speaker prefixes
        # without re-parsing the rendered text (see MainPane._tag_speaker_colors).
        "speaker_lines": [seg.speaker for seg in lesson.segments],
        "has_speakers": any(seg.speaker for seg in lesson.segments),
        "has_audio": bool(
            lesson.audio_file
            and (folder / lesson.audio_file).is_file()
        ),
        "duration_seconds": max(
            (seg.end for seg in lesson.segments), default=0.0
        ),
        "_folder": str(folder),
        "_loaded": True,
    }


def _metadata_to_note(folder: Path, metadata: dict) -> dict:
    """Build a list-view note from metadata.json alone.

    The transcript body is loaded lazily when the note is opened, so the
    lesson list stays fast even with hundreds of recordings."""
    title = str(metadata.get("suggested_title") or metadata.get("title") or "授業")
    date_value = str(metadata.get("date", ""))
    label, date_str = _note_label_and_date(title, date_value)
    duration = _format_duration(float(metadata.get("duration_seconds", 0.0) or 0.0))
    segment_count = int(metadata.get("segment_count", 0) or 0)
    source_audio_name = str(metadata.get("source_audio_name", ""))
    audio_file = str(metadata.get("audio_file", ""))

    tags: list[str] = []
    if metadata.get("has_important"):
        tags.append("★重要")
    if metadata.get("has_question"):
        tags.append("？質問")
    keywords = "  ".join(tags) if tags else ("文字起こし済み" if segment_count else "文字起こしなし")
    if str(metadata.get("status", "")) == "needs_repair":
        keywords = "⚠ 要修復（記録の一部を読み取れません）"

    return {
        "id": str(metadata.get("lesson_id", folder.name)),
        "label": label,
        "title": title,
        "meta": (
            date_str
            + duration
            + (f"\n元ファイル: {source_audio_name}" if source_audio_name else "")
        ),
        "keywords": keywords,
        "transcript": "（本文を読み込み中…）",
        "editable_transcript": "",
        "summary": "（読み込み中…）",
        "source_audio_name": source_audio_name,
        # Editing/transcription stay disabled until the body is loaded.
        "has_transcript": False,
        "has_audio": bool(audio_file and (folder / audio_file).is_file()),
        "duration_seconds": float(metadata.get("duration_seconds", 0.0) or 0.0),
        "_folder": str(folder),
        "_loaded": False,
    }


class _ModeDialog(ctk.CTkToplevel):
    """Modal dialog for selecting recording/transcription language mode."""

    MODES = [
        ("japanese", "日本語（ReazonSpeech）"),
        ("english", "English"),
        ("record_only", "録音のみ（文字起こしなし）"),
    ]

    # Subclasses that offer a speaker-diarization option set this True.
    SHOW_DIARIZATION_OPTION = False

    DIARIZATION_CHOICES = [
        ("0", "なし（デフォルト）"),
        ("2", "2名"),
        ("3", "3名"),
        ("4", "4名"),
    ]

    def __init__(
        self,
        parent: ctk.CTk,
        title: str,
        detail: str = "",
        diarization_available: bool = False,
    ) -> None:
        super().__init__(parent)
        self.withdraw()
        self.title(title)
        self.result: str | None = None
        self.diarization_speakers: int | None = None
        show_diarization = self.SHOW_DIARIZATION_OPTION and diarization_available
        dialog_height = 300 if detail else 240
        if show_diarization:
            dialog_height += 150
        self.geometry(f"320x{dialog_height}")
        self.resizable(False, False)
        self.transient(parent)

        ctk.CTkLabel(
            self, text=title, font=ctk.CTkFont(FONT, 14, "bold"), anchor="w",
        ).pack(fill="x", padx=20, pady=(18, 10))
        if detail:
            ctk.CTkLabel(
                self,
                text=f"選択したファイル:\n{detail}",
                font=ctk.CTkFont(FONT, 10),
                text_color=COLORS["muted"],
                anchor="w",
                justify="left",
                wraplength=280,
            ).pack(fill="x", padx=20, pady=(0, 8))

        self._var = ctk.StringVar(value="japanese")
        first_radio: ctk.CTkRadioButton | None = None
        for value, label in self.MODES:
            radio = ctk.CTkRadioButton(
                self, text=label, variable=self._var, value=value,
                font=ctk.CTkFont(FONT, 12),
            )
            radio.pack(anchor="w", padx=24, pady=4)
            if first_radio is None:
                first_radio = radio
        self._first_radio = first_radio

        self._diarization_var: ctk.StringVar | None = None
        if show_diarization:
            ctk.CTkLabel(
                self,
                text="話者分離（任意）",
                font=ctk.CTkFont(FONT, 12, "bold"),
                anchor="w",
            ).pack(fill="x", padx=20, pady=(12, 2))
            self._diarization_var = ctk.StringVar(value="0")
            for value, label in self.DIARIZATION_CHOICES:
                ctk.CTkRadioButton(
                    self, text=label, variable=self._diarization_var, value=value,
                    font=ctk.CTkFont(FONT, 12),
                ).pack(anchor="w", padx=24, pady=2)
            ctk.CTkLabel(
                self,
                text="話者分離は処理時間が長くなります（音声10分あたり約2分）",
                font=ctk.CTkFont(FONT, 10),
                text_color=COLORS["muted"],
                anchor="w",
                justify="left",
                wraplength=280,
            ).pack(fill="x", padx=20, pady=(2, 4))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(14, 18))
        ctk.CTkButton(
            btn_row, text="キャンセル", width=100, fg_color="transparent",
            border_width=1, text_color=COLORS["text"],
            command=self.destroy,
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btn_row, text="開始", width=100,
            fg_color=COLORS["blue"], hover_color=COLORS["blue_hover"],
            command=self._ok,
        ).pack(side="right")

        self.update_idletasks()
        parent.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - 320) // 2)
        y = parent.winfo_rooty() + max(
            0,
            (parent.winfo_height() - dialog_height) // 2,
        )
        self.geometry(f"320x{dialog_height}+{x}+{y}")
        # Keyboard access: Enter = start, Escape = cancel.
        self.bind("<Return>", lambda _event: self._ok())
        self.bind("<Escape>", lambda _event: self.destroy())
        self.deiconify()
        self.lift()
        self.attributes("-topmost", True)
        self.after(250, self._release_topmost)
        self.wait_visibility()
        self.grab_set()
        self.focus_force()
        if self._first_radio is not None:
            self._first_radio.focus_set()
        self.wait_window()

    def _release_topmost(self) -> None:
        if self.winfo_exists():
            self.attributes("-topmost", False)

    def _ok(self) -> None:
        self.result = self._var.get()
        if self._diarization_var is not None:
            value = int(self._diarization_var.get())
            self.diarization_speakers = value if value >= 1 else None
        self.destroy()


class _TranscriptionModeDialog(_ModeDialog):
    """Low-memory choices for transcribing an already saved recording."""

    MODES = [
        ("japanese", "日本語（ReazonSpeech）"),
        ("mixed", "日英混在（標準・低メモリ）"),
        ("english", "English（Parakeet）"),
    ]

    # Re-transcription re-runs ASR, so this is the one dialog that can also
    # offer speaker diarization on the result (see diarization_available()).
    SHOW_DIARIZATION_OPTION = True


class _TemplateManagerDialog(ctk.CTkToplevel):
    def __init__(
        self,
        parent: ctk.CTk,
        templates: list[dict[str, Any]],
        save_changes: Callable[[list[dict[str, Any]]], None],
    ) -> None:
        super().__init__(parent)
        self.title("要約テンプレート管理")
        self.geometry("860x620")
        self.minsize(760, 540)
        self.transient(parent)
        self._templates = [dict(value) for value in templates]
        self._save_changes = save_changes
        self._selected_id = ""
        self._buttons: dict[str, ctk.CTkButton] = {}
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            self,
            text="要約テンプレート管理",
            font=ctk.CTkFont(FONT, 20, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=22,
            pady=(20, 14),
        )

        left = ctk.CTkFrame(
            self,
            width=250,
            fg_color=COLORS["detail"],
            corner_radius=12,
        )
        left.grid(row=1, column=0, sticky="nsew", padx=(22, 10), pady=(0, 20))
        left.grid_propagate(False)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)
        self.list_frame = ctk.CTkScrollableFrame(
            left,
            fg_color="transparent",
            corner_radius=0,
        )
        self.list_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.list_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(
            left,
            text="＋ 新規テンプレート",
            command=self._new_template,
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 10, "bold"),
        ).grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))

        editor = ctk.CTkFrame(
            self,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["line"],
        )
        editor.grid(row=1, column=1, sticky="nsew", padx=(10, 22), pady=(0, 20))
        editor.grid_columnconfigure(0, weight=1)
        editor.grid_rowconfigure(5, weight=1)
        ctk.CTkLabel(
            editor,
            text="テンプレート名",
            font=ctk.CTkFont(FONT, 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 5))
        self.name_entry = ctk.CTkEntry(
            editor,
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 10),
        )
        self.name_entry.grid(row=1, column=0, sticky="ew", padx=18)
        ctk.CTkLabel(
            editor,
            text="AIへの指示",
            font=ctk.CTkFont(FONT, 10, "bold"),
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=18, pady=(14, 5))
        self.instruction_box = ctk.CTkTextbox(
            editor,
            height=150,
            corner_radius=8,
            border_width=1,
            border_color=COLORS["line"],
            font=ctk.CTkFont(FONT, 10),
            wrap="word",
        )
        self.instruction_box.grid(row=3, column=0, sticky="ew", padx=18)
        ctk.CTkLabel(
            editor,
            text="出力見出し（1行に1つ）",
            font=ctk.CTkFont(FONT, 10, "bold"),
            anchor="w",
        ).grid(row=4, column=0, sticky="new", padx=18, pady=(14, 5))
        self.sections_box = ctk.CTkTextbox(
            editor,
            height=120,
            corner_radius=8,
            border_width=1,
            border_color=COLORS["line"],
            font=ctk.CTkFont(FONT, 10),
            wrap="word",
        )
        self.sections_box.grid(
            row=5,
            column=0,
            sticky="nsew",
            padx=18,
        )
        self.help_label = ctk.CTkLabel(
            editor,
            text="標準テンプレートは保護されています。複製すると編集できます。",
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
            anchor="w",
        )
        self.help_label.grid(row=6, column=0, sticky="ew", padx=18, pady=(8, 6))
        actions = ctk.CTkFrame(editor, fg_color="transparent")
        actions.grid(row=7, column=0, sticky="ew", padx=18, pady=(4, 18))
        actions.grid_columnconfigure(0, weight=1)
        self.delete_button = ctk.CTkButton(
            actions,
            text="削除",
            command=self._delete_selected,
            width=74,
            fg_color="transparent",
            hover_color="#fde9e6",
            text_color="#a13d32",
            border_width=1,
            border_color="#e4bbb7",
        )
        self.delete_button.grid(row=0, column=1, padx=(8, 0))
        self.duplicate_button = ctk.CTkButton(
            actions,
            text="複製して編集",
            command=self._duplicate_selected,
            width=110,
            fg_color="transparent",
            hover_color="#e5e9ed",
            text_color=COLORS["text"],
            border_width=1,
            border_color=COLORS["line"],
        )
        self.duplicate_button.grid(row=0, column=2, padx=(8, 0))
        self.save_button = ctk.CTkButton(
            actions,
            text="保存",
            command=self._save_selected,
            width=84,
        )
        self.save_button.grid(row=0, column=3, padx=(8, 0))
        ctk.CTkButton(
            actions,
            text="閉じる",
            command=self.destroy,
            width=84,
            fg_color="transparent",
            hover_color="#e5e9ed",
            text_color=COLORS["text"],
            border_width=1,
            border_color=COLORS["line"],
        ).grid(row=0, column=4, padx=(8, 0))

        self._render_list()
        if self._templates:
            self._select(self._templates[0]["id"])
        self.grab_set()
        self.focus_force()

    def _render_list(self) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()
        self._buttons.clear()
        for row, value in enumerate(self._templates):
            button = ctk.CTkButton(
                self.list_frame,
                text=("標準  " if value.get("builtin") else "自作  ")
                + str(value["name"]),
                command=lambda template_id=value["id"]: self._select(
                    template_id
                ),
                height=38,
                corner_radius=8,
                fg_color="transparent",
                hover_color="#e5e9ed",
                text_color=COLORS["text"],
                anchor="w",
                font=ctk.CTkFont(FONT, 9),
            )
            button.grid(row=row, column=0, sticky="ew", pady=2)
            self._buttons[str(value["id"])] = button

    def _selected(self) -> dict[str, Any] | None:
        return next(
            (
                value
                for value in self._templates
                if value["id"] == self._selected_id
            ),
            None,
        )

    def _select(self, template_id: str) -> None:
        self._selected_id = template_id
        value = self._selected()
        if value is None:
            return
        for key, button in self._buttons.items():
            button.configure(
                fg_color=COLORS["blue_soft"]
                if key == template_id
                else "transparent"
            )
        self._set_editor_text(value)
        builtin = bool(value.get("builtin"))
        state = "disabled" if builtin else "normal"
        self.name_entry.configure(state=state)
        self.instruction_box.configure(state=state)
        self.sections_box.configure(state=state)
        self.save_button.configure(state=state)
        self.delete_button.configure(state=state)
        self.duplicate_button.configure(state="normal")

    def _set_editor_text(self, value: dict[str, Any]) -> None:
        for widget in (
            self.name_entry,
            self.instruction_box,
            self.sections_box,
        ):
            widget.configure(state="normal")
        self.name_entry.delete(0, "end")
        self.name_entry.insert(0, str(value["name"]))
        self.instruction_box.delete("1.0", "end")
        self.instruction_box.insert("1.0", str(value["instruction"]))
        self.sections_box.delete("1.0", "end")
        self.sections_box.insert("1.0", "\n".join(value["sections"]))

    def _new_template(self) -> None:
        value = {
            "id": f"custom_{int(time.time() * 1000)}",
            "name": "新しいテンプレート",
            "instruction": "文字起こしの内容を分かりやすく整理してください。",
            "sections": ["要約"],
            "builtin": False,
        }
        self._templates.append(value)
        self._render_list()
        self._select(value["id"])

    def _duplicate_selected(self) -> None:
        source = self._selected()
        if source is None:
            return
        value = {
            **source,
            "id": f"custom_{int(time.time() * 1000)}",
            "name": f"{source['name']} のコピー",
            "builtin": False,
        }
        self._templates.append(value)
        self._render_list()
        self._select(value["id"])

    def _save_selected(self) -> None:
        value = self._selected()
        if value is None or value.get("builtin"):
            return
        try:
            normalized = normalize_template(
                {
                    "id": value["id"],
                    "name": self.name_entry.get(),
                    "instruction": self.instruction_box.get(
                        "1.0",
                        "end-1c",
                    ),
                    "sections": self.sections_box.get(
                        "1.0",
                        "end-1c",
                    ).splitlines(),
                },
                builtin=False,
            )
        except ValueError as exc:
            messagebox.showerror(
                "テンプレート",
                str(exc),
                parent=self,
            )
            return
        value.update(normalized)
        self._persist()
        self._render_list()
        self._select(value["id"])

    def _delete_selected(self) -> None:
        value = self._selected()
        if value is None or value.get("builtin"):
            return
        if not messagebox.askyesno(
            "テンプレートを削除",
            f"「{value['name']}」を削除しますか？",
            parent=self,
        ):
            return
        self._templates.remove(value)
        self._persist()
        self._render_list()
        if self._templates:
            self._select(self._templates[0]["id"])

    def _persist(self) -> None:
        self._save_changes(self._templates)


class _DictionaryManagerDialog(ctk.CTkToplevel):
    def __init__(
        self,
        parent: ctk.CTk,
        entries: list[dict[str, Any]],
        save_changes: Callable[[list[dict[str, Any]]], None],
    ) -> None:
        super().__init__(parent)
        self.title("補正辞書")
        self.geometry("820x600")
        self.minsize(740, 540)
        self.transient(parent)
        self._entries = [dict(value) for value in entries]
        self._save_changes = save_changes
        self._selected_id = ""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text="文字起こし・要約用の補正辞書",
            font=ctk.CTkFont(FONT, 20, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=22,
            pady=(20, 5),
        )
        ctk.CTkLabel(
            self,
            text=(
                "誤認識候補は文字起こし後に正式表記へ補正され、"
                "読みと説明は要約AIの参考語彙になります。"
            ),
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
            anchor="w",
        ).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="new",
            padx=22,
        )

        left = ctk.CTkFrame(
            self,
            width=250,
            fg_color=COLORS["detail"],
            corner_radius=12,
        )
        left.grid(row=2, column=0, sticky="nsew", padx=(22, 10), pady=(14, 20))
        left.grid_propagate(False)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)
        self.list_frame = ctk.CTkScrollableFrame(
            left,
            fg_color="transparent",
            corner_radius=0,
        )
        self.list_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.list_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(
            left,
            text="＋ 新しい用語",
            command=self._new_entry,
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 10, "bold"),
        ).grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))

        editor = ctk.CTkFrame(
            self,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["line"],
        )
        editor.grid(row=2, column=1, sticky="nsew", padx=(10, 22), pady=(14, 20))
        editor.grid_columnconfigure((0, 1), weight=1)
        editor.grid_rowconfigure(7, weight=1)
        self.term_entry = self._labeled_entry(
            editor,
            "正式表記（必須）",
            0,
            0,
            "例：ディスレクシア",
        )
        self.reading_entry = self._labeled_entry(
            editor,
            "読み",
            0,
            1,
            "例：でぃすれくしあ",
        )
        ctk.CTkLabel(
            editor,
            text="分類",
            font=ctk.CTkFont(FONT, 10, "bold"),
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=(18, 8), pady=(14, 5))
        self.category_menu = ctk.CTkOptionMenu(
            editor,
            values=list(DICTIONARY_CATEGORIES),
            height=38,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 10),
        )
        self.category_menu.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=(18, 8),
        )
        ctk.CTkLabel(
            editor,
            text="誤認識候補（カンマ区切り）",
            font=ctk.CTkFont(FONT, 10, "bold"),
            anchor="w",
        ).grid(row=2, column=1, sticky="ew", padx=(8, 18), pady=(14, 5))
        self.aliases_entry = ctk.CTkEntry(
            editor,
            placeholder_text="例：ディスレキシア, ディスレクシヤ",
            height=38,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 9),
        )
        self.aliases_entry.grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(8, 18),
        )
        ctk.CTkLabel(
            editor,
            text="意味・説明",
            font=ctk.CTkFont(FONT, 10, "bold"),
            anchor="w",
        ).grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=18,
            pady=(14, 5),
        )
        self.description_box = ctk.CTkTextbox(
            editor,
            height=170,
            corner_radius=8,
            border_width=1,
            border_color=COLORS["line"],
            wrap="word",
            font=ctk.CTkFont(FONT, 10),
        )
        self.description_box.grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="nsew",
            padx=18,
        )
        ctk.CTkLabel(
            editor,
            text=(
                "注意：誤認識候補は一致した文字だけを置換します。"
                "短すぎる一般語は登録しないでください。"
            ),
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["muted"],
            anchor="w",
        ).grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=18,
            pady=(8, 4),
        )
        actions = ctk.CTkFrame(editor, fg_color="transparent")
        actions.grid(
            row=8,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=18,
            pady=(10, 18),
        )
        actions.grid_columnconfigure(0, weight=1)
        self.delete_button = ctk.CTkButton(
            actions,
            text="削除",
            command=self._delete_selected,
            width=76,
            fg_color="transparent",
            hover_color="#fde9e6",
            text_color="#a13d32",
            border_width=1,
            border_color="#e4bbb7",
        )
        self.delete_button.grid(row=0, column=1, padx=(8, 0))
        ctk.CTkButton(
            actions,
            text="保存",
            command=self._save_selected,
            width=84,
        ).grid(row=0, column=2, padx=(8, 0))
        ctk.CTkButton(
            actions,
            text="閉じる",
            command=self.destroy,
            width=84,
            fg_color="transparent",
            hover_color="#e5e9ed",
            text_color=COLORS["text"],
            border_width=1,
            border_color=COLORS["line"],
        ).grid(row=0, column=3, padx=(8, 0))

        self._render_list()
        if self._entries:
            self._select(self._entries[0]["id"])
        else:
            self._new_entry()
        self.grab_set()
        self.focus_force()

    @staticmethod
    def _labeled_entry(
        parent,
        label: str,
        row: int,
        column: int,
        placeholder: str,
    ) -> ctk.CTkEntry:
        padx = (18, 8) if column == 0 else (8, 18)
        ctk.CTkLabel(
            parent,
            text=label,
            font=ctk.CTkFont(FONT, 10, "bold"),
            anchor="w",
        ).grid(row=row, column=column, sticky="ew", padx=padx, pady=(18, 5))
        entry = ctk.CTkEntry(
            parent,
            placeholder_text=placeholder,
            height=38,
            corner_radius=8,
            font=ctk.CTkFont(FONT, 10),
        )
        entry.grid(row=row + 1, column=column, sticky="ew", padx=padx)
        return entry

    def _render_list(self) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()
        self._buttons: dict[str, ctk.CTkButton] = {}
        for row, entry in enumerate(self._entries):
            button = ctk.CTkButton(
                self.list_frame,
                text=f"{entry['term']}"
                + (f"\n{entry['reading']}" if entry.get("reading") else ""),
                command=lambda entry_id=entry["id"]: self._select(entry_id),
                height=46,
                corner_radius=8,
                fg_color="transparent",
                hover_color="#e5e9ed",
                text_color=COLORS["text"],
                anchor="w",
                font=ctk.CTkFont(FONT, 9),
            )
            button.grid(row=row, column=0, sticky="ew", pady=2)
            self._buttons[str(entry["id"])] = button

    def _selected(self) -> dict[str, Any] | None:
        return next(
            (
                entry
                for entry in self._entries
                if entry["id"] == self._selected_id
            ),
            None,
        )

    def _select(self, entry_id: str) -> None:
        self._selected_id = entry_id
        entry = self._selected()
        if entry is None:
            return
        for key, button in self._buttons.items():
            button.configure(
                fg_color=COLORS["blue_soft"]
                if key == entry_id
                else "transparent"
            )
        self.term_entry.delete(0, "end")
        self.term_entry.insert(0, entry["term"])
        self.reading_entry.delete(0, "end")
        self.reading_entry.insert(0, entry.get("reading", ""))
        self.aliases_entry.delete(0, "end")
        self.aliases_entry.insert(0, ", ".join(entry.get("aliases", [])))
        self.category_menu.set(entry.get("category", "学習用語"))
        self.description_box.delete("1.0", "end")
        self.description_box.insert("1.0", entry.get("description", ""))

    def _new_entry(self) -> None:
        entry = normalize_entry(
            {
                "term": "新しい用語",
                "category": "学習用語",
            }
        )
        self._entries.append(entry)
        self._render_list()
        self._select(entry["id"])
        self.term_entry.select_range(0, "end")
        self.term_entry.focus_set()

    def _save_selected(self) -> None:
        entry = self._selected()
        if entry is None:
            return
        try:
            value = normalize_entry(
                {
                    "id": entry["id"],
                    "term": self.term_entry.get(),
                    "reading": self.reading_entry.get(),
                    "aliases": self.aliases_entry.get(),
                    "description": self.description_box.get(
                        "1.0",
                        "end-1c",
                    ),
                    "category": self.category_menu.get(),
                }
            )
        except ValueError as exc:
            messagebox.showerror("補正辞書", str(exc), parent=self)
            return
        entry.update(value)
        self._persist()
        self._render_list()
        self._select(entry["id"])

    def _delete_selected(self) -> None:
        entry = self._selected()
        if entry is None:
            return
        if not messagebox.askyesno(
            "用語を削除",
            f"「{entry['term']}」を辞書から削除しますか？",
            parent=self,
        ):
            return
        self._entries.remove(entry)
        self._persist()
        self._render_list()
        if self._entries:
            self._select(self._entries[0]["id"])
        else:
            self._new_entry()

    def _persist(self) -> None:
        self._save_changes(self._entries)


# TranscriptSegment.speaker == "teacher" is silently normalized to "" on
# load (see models.TranscriptSegment.from_dict), so a segment renamed to
# that exact word would appear to lose its speaker after the app restarts.
RESERVED_SPEAKER_NAMES = {"teacher"}


def _build_speaker_rename_mapping(
    entries: dict[str, str],
) -> tuple[dict[str, str], str]:
    """Sanitize raw speaker-rename dialog entries into an apply mapping.

    `entries` maps each current speaker name to whatever the user typed
    for its replacement. Returns (mapping, error_message):
    - Blank input, or input equal to the current name, means "don't
      change this speaker" and is skipped (not an error).
    - A new name of "teacher" (any case) is rejected outright: when
      `error_message` is non-empty, `mapping` is always `{}` and the
      caller should show the message instead of applying anything.
    Pulled out of the dialog class as a pure function so the validation
    rules can be unit-tested without building a real Tk window.
    """
    mapping: dict[str, str] = {}
    for old_name, raw_new_name in entries.items():
        new_name = " ".join(raw_new_name.split()).strip()
        if not new_name or new_name == old_name:
            continue
        if new_name.lower() in RESERVED_SPEAKER_NAMES:
            return {}, (
                f"「{new_name}」という名前は使えません。"
                "別の名前を入力してください。"
            )
        mapping[old_name] = new_name
    return mapping, ""


class _SpeakerRenameDialog(ctk.CTkToplevel):
    """Modal dialog to rename one or more distinct speaker labels.

    Shows the current, distinct speaker names (in order of first
    appearance) each with an entry for its new name. A blank entry means
    "don't change this speaker". On success, `self.result` holds the
    {old_name: new_name} mapping to apply; it stays None if the dialog
    was cancelled or nothing was actually changed.
    """

    def __init__(self, parent: ctk.CTk, speakers: list[str]) -> None:
        super().__init__(parent)
        self.withdraw()
        self.title("話者名を変更")
        self.result: dict[str, str] | None = None
        self._entries: dict[str, ctk.CTkEntry] = {}
        self.resizable(False, False)
        self.transient(parent)

        ctk.CTkLabel(
            self,
            text="話者名を変更",
            font=ctk.CTkFont(FONT, 14, "bold"),
            anchor="w",
        ).pack(fill="x", padx=20, pady=(18, 4))
        ctk.CTkLabel(
            self,
            text=(
                "新しい名前を入力してください。空欄のままにすると、"
                "その話者名は変更しません。"
            ),
            font=ctk.CTkFont(FONT, 10),
            text_color=COLORS["muted"],
            anchor="w",
            justify="left",
            wraplength=360,
        ).pack(fill="x", padx=20, pady=(0, 10))

        rows = ctk.CTkFrame(self, fg_color="transparent")
        rows.pack(fill="x", padx=20)
        rows.grid_columnconfigure(1, weight=1)
        first_entry: ctk.CTkEntry | None = None
        for row, name in enumerate(speakers):
            ctk.CTkLabel(
                rows,
                text=name,
                font=ctk.CTkFont(FONT, 11, "bold"),
                anchor="w",
                width=110,
            ).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
            entry = ctk.CTkEntry(
                rows,
                placeholder_text=f"例：先生／{name}のまま",
                width=200,
                height=32,
                corner_radius=8,
                font=ctk.CTkFont(FONT, 10),
            )
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            self._entries[name] = entry
            if first_entry is None:
                first_entry = entry

        self._error_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(FONT, 9),
            text_color=COLORS["danger_text"],
            anchor="w",
            justify="left",
            wraplength=360,
        )
        self._error_label.pack(fill="x", padx=20, pady=(8, 0))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(10, 18))
        ctk.CTkButton(
            btn_row, text="キャンセル", width=100, fg_color="transparent",
            border_width=1, text_color=COLORS["text"],
            command=self.destroy,
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btn_row, text="変更する", width=100,
            fg_color=COLORS["blue"], hover_color=COLORS["blue_hover"],
            command=self._ok,
        ).pack(side="right")

        self.update_idletasks()
        width = 420
        height = self.winfo_reqheight()
        parent.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - width) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.bind("<Return>", lambda _event: self._ok())
        self.bind("<Escape>", lambda _event: self.destroy())
        self.deiconify()
        self.lift()
        self.attributes("-topmost", True)
        self.after(250, self._release_topmost)
        self.wait_visibility()
        self.grab_set()
        self.focus_force()
        if first_entry is not None:
            first_entry.focus_set()
        self.wait_window()

    def _release_topmost(self) -> None:
        if self.winfo_exists():
            self.attributes("-topmost", False)

    def _ok(self) -> None:
        raw_entries = {
            old_name: entry.get() for old_name, entry in self._entries.items()
        }
        mapping, error = _build_speaker_rename_mapping(raw_entries)
        if error:
            self._error_label.configure(text=error)
            return
        self.result = mapping
        self.destroy()


class _DiarizeSpeakerCountDialog(ctk.CTkToplevel):
    """Modal dialog to pick how many distinct speakers to diarize for.

    Deliberately narrower than _ModeDialog's DIARIZATION_CHOICES (which
    also offers "なし"): reaching this dialog already means the user
    wants to diarize, so "none" isn't offered here.
    """

    CHOICES = [
        ("2", "2名"),
        ("3", "3名"),
        ("4", "4名"),
    ]

    def __init__(self, parent: ctk.CTk) -> None:
        super().__init__(parent)
        self.withdraw()
        self.title("話者を推定")
        self.result: int | None = None
        self.resizable(False, False)
        self.transient(parent)

        ctk.CTkLabel(
            self,
            text="話者の人数を選んでください",
            font=ctk.CTkFont(FONT, 14, "bold"),
            anchor="w",
        ).pack(fill="x", padx=20, pady=(18, 10))

        self._var = ctk.StringVar(value="2")
        first_radio: ctk.CTkRadioButton | None = None
        for value, label in self.CHOICES:
            radio = ctk.CTkRadioButton(
                self, text=label, variable=self._var, value=value,
                font=ctk.CTkFont(FONT, 12),
            )
            radio.pack(anchor="w", padx=24, pady=4)
            if first_radio is None:
                first_radio = radio
        self._first_radio = first_radio

        ctk.CTkLabel(
            self,
            text="話者分離は処理時間が長くなります（音声10分あたり約2分）",
            font=ctk.CTkFont(FONT, 10),
            text_color=COLORS["muted"],
            anchor="w",
            justify="left",
            wraplength=280,
        ).pack(fill="x", padx=20, pady=(8, 4))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(14, 18))
        ctk.CTkButton(
            btn_row, text="キャンセル", width=100, fg_color="transparent",
            border_width=1, text_color=COLORS["text"],
            command=self.destroy,
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btn_row, text="開始", width=100,
            fg_color=COLORS["blue"], hover_color=COLORS["blue_hover"],
            command=self._ok,
        ).pack(side="right")

        self.update_idletasks()
        width = 320
        height = self.winfo_reqheight()
        parent.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - width) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.bind("<Return>", lambda _event: self._ok())
        self.bind("<Escape>", lambda _event: self.destroy())
        self.deiconify()
        self.lift()
        self.attributes("-topmost", True)
        self.after(250, self._release_topmost)
        self.wait_visibility()
        self.grab_set()
        self.focus_force()
        if first_radio is not None:
            first_radio.focus_set()
        self.wait_window()

    def _release_topmost(self) -> None:
        if self.winfo_exists():
            self.attributes("-topmost", False)

    def _ok(self) -> None:
        self.result = int(self._var.get())
        self.destroy()


class OtoWeaveApp(ctk.CTk):
    """Four-pane CustomTkinter shell with VS Code-style smart routing."""

    ACTION_ROUTES = {"record", "import"}
    VIEW_ROUTES = {"notes", "dictionary", "settings"}

    def __init__(self, controller: LearningAccessController | None = None) -> None:
        super().__init__()
        # report_callback_exception は __init__ 中の Tk コールバックからも
        # 呼ばれ得るため、参照する状態を最初に用意しておく。
        self._last_error_dialog_at = 0.0
        self._close_wait_started = 0.0
        self.title("OtoWeave")
        # ウィンドウ/タスクバーのアイコン。アセット欠如や非Windows環境でも
        # 起動を妨げないよう失敗は無視する。
        try:
            icon_path = Path(__file__).resolve().parent / "assets" / "icon.ico"
            if os.name == "nt" and icon_path.is_file():
                self.iconbitmap(default=str(icon_path))
        except Exception:
            pass
        width, height, min_width, min_height = initial_window_geometry(
            self.winfo_screenwidth(),
            self.winfo_screenheight(),
        )
        self.geometry(f"{width}x{height}")
        self.minsize(min_width, min_height)
        self.configure(fg_color=COLORS["page"])

        self.controller = controller
        self.active_note_id: str = ""
        self.current_route = "notes"
        self.right_visible = True
        self._active_folder: Path | None = None
        self._note_map: dict[str, dict] = {}
        self._elapsed_start: float = 0.0
        self._elapsed_after_id: str = ""
        self._player_duration: float = 0.0
        self._tts_target: str = ""
        # 読み上げテキスト（文字起こし本文を含む）は共有 %TEMP% ではなく
        # データルート配下に置く。前回強制終了で残ったファイルも掃除する。
        tts_dir = (
            tts_temp_dir(controller.store.root)
            if controller is not None
            else None
        )
        if tts_dir is not None:
            cleanup_stale_tts_files(tts_dir)
        self._tts = WindowsTts(
            on_finished=self._notify_tts_finished,
            on_error=self._notify_tts_error,
            temp_dir=tts_dir,
        )
        self._file_dialog_active = False
        self._file_dialog_process: subprocess.Popen[bytes] | None = None
        self._detail_width = 280
        self._right_width = 310
        self._resize_state: tuple[int, int, int, int, int] | None = None
        self._font_baselines: dict[int, tuple[ctk.CTkFont, int]] = {}
        self._ai_focus_mode = False
        self._selected_summary_template_id = "lesson_record"
        self._summary_templates_path = (
            templates_path(controller.store.root)
            if controller is not None
            else None
        )
        self._summary_templates = (
            load_templates(self._summary_templates_path)
            if self._summary_templates_path is not None
            else load_templates(Path(""))
        )
        self._dictionary_path = (
            dictionary_path(controller.store.root)
            if controller is not None
            else None
        )
        self._dictionary_entries = (
            load_dictionary(self._dictionary_path)
            if self._dictionary_path is not None
            else []
        )
        self._display_settings_path = (
            controller.store.root / "display_settings.json"
            if controller is not None
            else None
        )
        self._display_settings = (
            load_display_settings(self._display_settings_path)
            if self._display_settings_path is not None
            else DisplaySettings(font_family=FONT)
        )
        self._available_fonts = list(available_reading_fonts(tkfont.families(self)))
        if self._display_settings.font_family not in self._available_fonts:
            self._available_fonts.insert(0, self._display_settings.font_family)
        self._ui_results: queue.SimpleQueue[
            tuple[Callable[[Any], None] | None, Any, Exception | None]
        ] = queue.SimpleQueue()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=72)
        self.grid_columnconfigure(1, weight=0, minsize=self._detail_width)
        self.grid_columnconfigure(2, weight=0, minsize=6)
        self.grid_columnconfigure(3, weight=1, minsize=480)
        self.grid_columnconfigure(4, weight=0, minsize=6)
        self.grid_columnconfigure(5, weight=0, minsize=self._right_width)

        ctk.set_appearance_mode(self._display_settings.color_mode)
        self._build_panes()
        self._enable_drag_and_drop()
        self._apply_display_preferences()
        if sys.platform == "darwin":
            # macOSだけ画面上部のネイティブメニューバーを設定する。Windowsは
            # 従来どおりメニューバー無し。
            # TODO(platform_support): platform_support.py 導入後は共通の
            # is_macos() 判定に置き換える。
            self._setup_mac_menu_bar()

        if controller:
            self._load_lessons()
            self.after(100, self._poll_events)
            self.after(400, self._maybe_warn_cloud_sync)
        else:
            # Controller-less demo shell (UI review / tests) uses mock data.
            self.detail_pane.populate_notes(
                {group: list(notes) for group, notes in NOTES.items()}
            )
            self.active_note_id = "kokoro"
            self._apply_note(NOTE_BY_ID[self.active_note_id])
            self.route_to("notes")

        self.after(40, self._drain_ui_results)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def report_callback_exception(self, exc_type, exc_value, exc_traceback) -> None:
        """Tk コールバック中の未捕捉例外をログに残し、平易な言葉で知らせる。

        技術的な詳細（例外メッセージ・スタックトレース）はログにのみ
        出力し、生徒にはそのまま見せない。
        """
        log_exception(
            "画面の処理中に未捕捉の例外が発生しました",
            (exc_type, exc_value, exc_traceback),
        )
        now = time.monotonic()
        # __init__ 完了前に呼ばれても落ちないよう getattr で防御する。
        if now - getattr(self, "_last_error_dialog_at", 0.0) < 10.0:
            return
        self._last_error_dialog_at = now
        try:
            messagebox.showerror(
                "エラー",
                "うまく処理できませんでした。\n"
                "もう一度試してみてください。\n"
                "何度も出るときは、アプリを閉じて開き直してください。",
                parent=self,
            )
        except Exception:
            pass

    def _show_friendly_error(
        self,
        title: str,
        message: str,
        exc: BaseException | None = None,
        detail: str = "",
    ) -> None:
        """例外はログへ記録し、ダイアログには平易な日本語だけを表示する。"""
        messagebox.showerror(
            title,
            build_friendly_error(title, message, exc, detail),
            parent=self,
        )

    def _maybe_warn_cloud_sync(self) -> None:
        """保存先がクラウド同期フォルダ配下なら、一度だけ注意を表示する。"""
        if self.controller is None:
            return
        data_root = self.controller.store.root
        notice_path = cloud_sync_notice_path(data_root)
        if load_cloud_sync_notice_shown(notice_path):
            return
        if not is_cloud_synced_path(data_root):
            return
        messagebox.showwarning(
            "保存先についてのお知らせ",
            "録音とノートの保存先が、OneDrive などのインターネットと"
            "同期されるフォルダの中にあります。\n"
            "このままだと、録音した音声が自動でインターネット上にも"
            "保存されることがあります。\n\n"
            "この端末の中だけに保存したいときは、録音画面の「保存先」の"
            "「変更」ボタンから、同期されないフォルダを選んでください。\n"
            "（このお知らせは一度だけ表示されます）",
            parent=self,
        )
        # 表示できる前に落ちた場合でも次回また警告できるよう、
        # 「表示済み」フラグはダイアログを出した後に保存する。
        try:
            save_cloud_sync_notice_shown(notice_path)
        except OSError:
            pass

    def _build_panes(self) -> None:
        self.activity_bar = ActivityBar(self, self.route_to)
        self.activity_bar.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)

        self.detail_pane = DetailPane(
            self,
            run_background=self.run_background,
            request_note=self._request_note,
            request_context=self._show_context,
            font_families=self._available_fonts,
            display_settings=self._display_settings,
            selected_font_size=self._text_size_points(
                self._display_settings.text_size
            ),
            request_display_settings=self._change_display_preferences,
            request_dictionary_manage=self._manage_dictionary,
            request_content_search=(
                self._request_content_search if self.controller else None
            ),
        )
        self.detail_pane.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=16)

        self.detail_resizer = self._build_column_resizer(
            target_column=1,
            minimum=220,
            maximum=480,
            direction=1,
        )
        self.detail_resizer.grid(row=0, column=2, sticky="ns", pady=16)

        self.main_pane = MainPane(
            self,
            run_background=self.run_background,
            request_right_toggle=self._toggle_right_pane,
            request_transcribe=self._on_transcribe_recording,
            request_rename=self._on_rename_note,
            request_delete=self._on_delete_note,
            request_record_start=self._start_record,
            request_audio_test=self._test_audio_input,
            request_save_location=self._change_recording_save_location,
            request_transcript_save=self._save_transcript_text,
            request_play_toggle=self._on_play_toggle,
            request_segment_play=self._on_segment_play,
            request_speak_transcript=self._on_speak_transcript,
            request_cancel_processing=self._on_cancel_processing,
            request_speaker_rename=self._on_rename_speakers,
            request_diarize=self._on_diarize_lesson,
        )
        self.main_pane.grid(row=0, column=3, sticky="nsew", padx=(0, 8), pady=16)

        self.right_resizer = self._build_column_resizer(
            target_column=5,
            minimum=260,
            maximum=520,
            direction=-1,
        )
        self.right_resizer.grid(row=0, column=4, sticky="ns", pady=16)

        on_chat = self._on_chat_question if self.controller else None
        on_summarize = self._on_summarize if self.controller else None
        on_cancel_summarize = self._on_cancel_summary if self.controller else None
        self.right_pane = RightPane(
            self,
            run_background=self.run_background,
            on_chat=on_chat,
            on_summarize=on_summarize,
            on_cancel_summarize=on_cancel_summarize,
            on_manage_templates=self._manage_summary_templates,
            on_focus_toggle=self._toggle_ai_focus,
            on_template_selected=self._on_summary_template_selected,
            on_speak_summary=self._on_speak_summary,
        )
        self.right_pane.grid(row=0, column=5, sticky="nsew", padx=(0, 16), pady=16)
        self.right_pane.set_templates(
            self._summary_templates,
            self._selected_summary_template_id,
        )
        # AI要約は4Bモデルが使える環境（ファイルあり・低メモリ機でない）
        # のみ有効。使えない環境では生成UIを隠して案内を出す。
        # チャット（2B）には影響しない。
        self._summarize_available = True
        if self.controller is not None:
            from otoweave_app import llm_chat

            self._summarize_available, _reason = llm_chat.summarize_availability(
                self.controller.project_root
            )
        self.right_pane.set_summarize_available(self._summarize_available)

        # 話者分離は該当モデルが同梱されている環境でのみ有効（テキスト
        # 起こしと同じくローカル限定・追加ダウンロードなし）。
        self._diarization_available = bool(
            self.controller is not None
            and diarization_available(self.controller.project_root)
        )
        self.main_pane.set_diarization_available(self._diarization_available)

        self.main_pane.stop_button.configure(command=self._on_stop_recording)
        self.main_pane.pause_button.configure(command=self._on_pause_recording)

    def _build_column_resizer(
        self,
        *,
        target_column: int,
        minimum: int,
        maximum: int,
        direction: int,
    ) -> ctk.CTkFrame:
        handle = ctk.CTkFrame(
            self,
            width=6,
            fg_color=COLORS["line"],
            corner_radius=3,
            cursor="sb_h_double_arrow",
        )
        handle.grid_propagate(False)
        handle.bind(
            "<ButtonPress-1>",
            lambda event: self._begin_column_resize(
                event,
                target_column,
                minimum,
                maximum,
                direction,
            ),
        )
        handle.bind("<B1-Motion>", self._drag_column_resize)
        handle.bind("<ButtonRelease-1>", self._end_column_resize)
        handle.bind(
            "<Enter>",
            lambda _event: handle.configure(fg_color="#9ec5ef"),
        )
        handle.bind(
            "<Leave>",
            lambda _event: (
                handle.configure(fg_color=COLORS["line"])
                if self._resize_state is None
                else None
            ),
        )
        return handle

    def _begin_column_resize(
        self,
        event,
        target_column: int,
        minimum: int,
        maximum: int,
        direction: int,
    ) -> None:
        current = int(self.grid_columnconfigure(target_column)["minsize"])
        self._resize_state = (
            int(event.x_root),
            current,
            target_column,
            minimum,
            maximum if direction > 0 else -maximum,
        )

    def _drag_column_resize(self, event) -> None:
        if self._resize_state is None:
            return
        start_x, start_width, column, minimum, encoded_maximum = self._resize_state
        direction = 1 if encoded_maximum > 0 else -1
        maximum = abs(encoded_maximum)
        width = self._resized_width(
            start_width,
            int(event.x_root) - start_x,
            direction,
            minimum,
            maximum,
        )
        self.grid_columnconfigure(column, weight=0, minsize=width)
        if column == 1:
            self._detail_width = width
            self.detail_pane.configure(width=width)
        else:
            self._right_width = width
            self.right_pane.configure(width=width)

    @staticmethod
    def _resized_width(
        start_width: int,
        pointer_delta: int,
        direction: int,
        minimum: int,
        maximum: int,
    ) -> int:
        width = start_width + pointer_delta * direction
        return max(minimum, min(maximum, width))

    def _end_column_resize(self, _event=None) -> None:
        self._resize_state = None
        self.detail_resizer.configure(fg_color=COLORS["line"])
        self.right_resizer.configure(fg_color=COLORS["line"])

    # ------------------------------------------------------------------
    # Lesson loading
    # ------------------------------------------------------------------

    def _load_lessons(self) -> None:
        assert self.controller is not None
        self.run_background(
            self.controller.store.list_lesson_metadata,
            self._apply_lessons,
        )

    def _apply_lessons(
        self,
        lessons: list[tuple[Path, LessonRecord | dict]],
    ) -> None:
        selected_id = self.active_note_id
        self._note_map = {}
        groups: dict[str, list[dict]] = {}
        for folder, record in lessons:
            if isinstance(record, dict):
                metadata = record
                note = _metadata_to_note(folder, metadata)
            else:
                metadata = record.metadata_dict()
                note = _lesson_to_note(folder, record)
            self._note_map[note["id"]] = note
            try:
                d = date_cls.fromisoformat(str(metadata.get("date", "")))
                group_key = f"{d.year}年 {d.month}月"
            except Exception:
                group_key = "その他"
            groups.setdefault(group_key, []).append(note)
        self.detail_pane.populate_notes(groups)
        if groups:
            selected_note = self._note_map.get(selected_id)
            if selected_note is None:
                selected_note = next(iter(next(iter(groups.values()))))
            self._apply_note(selected_note)
        else:
            self.active_note_id = ""
            self._active_folder = None
            self.main_pane.set_empty_note()
            self.right_pane.set_empty_note()
        self.route_to("notes")
        self._apply_display_preferences()

    # ------------------------------------------------------------------
    # Chat backend connection
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Recording / Import actions
    # ------------------------------------------------------------------

    def _start_record(self, options: dict[str, Any]) -> None:
        assert self.controller is not None
        if self.controller.busy:
            messagebox.showinfo("録音", "現在処理中です。完了後にお試しください。", parent=self)
            return
        dialog = _ModeDialog(self, "録音モードを選択")
        if dialog.result is None:
            return
        # The microphone would pick up the synthesized voice.
        self._tts.stop()
        source = options.get("source")
        if source is None:
            messagebox.showerror("音声入力", "音声入力デバイスが見つかりません。", parent=self)
            return
        from otoweave_app.audio import AudioProcessingOptions

        processing = AudioProcessingOptions(
            noise_reduction=bool(options.get("noise_reduction")),
            sensitivity=float(options.get("sensitivity", 1.0)),
            automatic_gain_control=bool(
                options.get("automatic_gain_control")
            ),
        )
        try:
            self.controller.start_lesson(
                dialog.result,
                source,
                processing=processing,
                speaker_label=str(options.get("speaker_label", "")),
            )
        except RuntimeError as exc:
            # 自前の平易な日本語メッセージ（処理中など）はそのまま見せる。
            messagebox.showerror("録音", str(exc), parent=self)
        except Exception as exc:
            self._show_friendly_error(
                "録音",
                "録音を始められませんでした。\n"
                "マイクの接続を確認して、もう一度試してください。",
                exc,
            )

    def _show_recording_setup(self) -> None:
        if self.controller is None:
            self.main_pane.configure_recording_sources([], "")
            self.main_pane.show_route("record")
            self.right_pane.show_route("record")
            return
        from otoweave_app.audio import available_audio_sources

        sources = available_audio_sources()
        self.main_pane.configure_recording_sources(
            sources,
            str(self.controller.store.root),
        )
        self.main_pane.show_route("record")
        self.right_pane.show_route("record")

    def _test_audio_input(self, options: dict[str, Any]) -> None:
        source = options.get("source")
        if source is None:
            self.main_pane.show_audio_test_result(
                None,
                "利用できる入力を選んでください。",
            )
            return
        if self.controller is not None and self.controller.busy:
            self.main_pane.show_audio_test_result(
                None,
                "録音・文字起こし・要約の処理中です。",
            )
            return
        from otoweave_app.audio import measure_audio_input

        def worker() -> tuple[Any, str]:
            try:
                return (
                    measure_audio_input(source, duration_seconds=3.0),
                    "",
                )
            except Exception as exc:
                # 技術的な例外文はログに残し、画面には平易な文だけ出す。
                log_exception("マイクの音量テストに失敗しました", exc)
                return None, (
                    "マイクの音を測定できませんでした。"
                    "接続を確認して、もう一度試してください。"
                )

        def done(result: tuple[Any, str]) -> None:
            if self.current_route == "record":
                measurement, error = result
                self.main_pane.show_audio_test_result(measurement, error)

        self.run_background(worker, done)

    def _change_recording_save_location(self) -> None:
        if self.controller is None:
            return
        if self.controller.busy:
            messagebox.showinfo(
                "保存先",
                "処理中は保存先を変更できません。",
                parent=self,
            )
            return
        # パスの手入力は生徒には難しいので、フォルダ選択ダイアログで選ぶ。
        while True:
            value = filedialog.askdirectory(
                parent=self,
                initialdir=str(self.controller.store.root),
                title="録音とノートの保存先フォルダを選んでください",
            )
            if not value:
                return
            root = Path(value).expanduser().resolve()
            if self._confirm_save_location(root):
                break
            # 「いいえ」＝選び直し。ダイアログをもう一度開く。
        try:
            root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise NotADirectoryError(root)
        except OSError as exc:
            self._show_friendly_error(
                "保存先",
                "このフォルダは保存先に使えませんでした。\n"
                "別のフォルダを選んでください。",
                exc,
            )
            return
        self.controller.store = LessonStore(root)
        # 読み上げ用一時ファイルの置き場も新しい保存先に追従させる。
        self._tts.temp_dir = tts_temp_dir(root)
        self._display_settings_path = root / "display_settings.json"
        self._summary_templates_path = templates_path(root)
        self._summary_templates = load_templates(
            self._summary_templates_path
        )
        self.right_pane.set_templates(self._summary_templates)
        self._dictionary_path = dictionary_path(root)
        self._dictionary_entries = load_dictionary(
            self._dictionary_path
        )
        self.controller.reload_dictionary()
        save_display_settings(
            self._display_settings_path,
            self._display_settings,
        )
        self._load_lessons()
        self.route_to("record")

    def _confirm_save_location(self, root: Path) -> bool:
        """選んだ保存先を使ってよいか確認する。True なら決定、False なら選び直し。

        クラウド同期フォルダらしき場所のときだけ、平易な言葉で注意を出す。"""
        if not is_cloud_synced_path(root):
            return True
        return bool(
            messagebox.askyesno(
                "保存先の確認",
                "この場所は、インターネットに自動保存される可能性があります。\n"
                "（OneDrive や Google ドライブなどと同期されるフォルダのようです）\n\n"
                "録音した音声をここに保存してよいか、学校の先生と確認してください。\n\n"
                "このフォルダをこのまま使いますか？\n"
                "「いいえ」を選ぶと、フォルダを選び直せます。",
                parent=self,
            )
        )

    def _start_import(self) -> None:
        assert self.controller is not None
        if self._file_dialog_active:
            self._cancel_audio_file_dialog()
            return
        if self.controller.busy:
            messagebox.showinfo("取り込み", "現在処理中です。完了後にお試しください。", parent=self)
            self.route_to("notes")
            return
        self._open_audio_file_dialog()

    def _open_audio_file_dialog(self) -> None:
        assert self.controller is not None
        if os.name != "nt":
            # macOS/Linux: Windows専用のPowerShellヘルパーは使えないため、
            # tkinterのネイティブファイル選択をメインスレッドでモーダル表示する。
            # TODO(platform_support): platform_support.py 導入後は共通の
            # is_windows() 判定に置き換える。
            self._open_audio_file_dialog_native()
            return
        helper = self.controller.project_root / "scripts" / "production" / "windows_audio_file_dialog.ps1"
        initial = Path.home() / "Documents"
        if not initial.is_dir():
            initial = Path.home()
        self._file_dialog_active = True
        self.main_pane.update_status(
            "Windowsのファイル選択を開いています。もう一度「取込」を押すと閉じます。"
        )

        def worker() -> tuple[str, str]:
            try:
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                process = subprocess.Popen(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-STA",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(helper),
                        "-InitialDirectory",
                        str(initial),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=creationflags,
                )
                self._file_dialog_process = process
                if not self._file_dialog_active:
                    process.terminate()
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    return (
                        "error",
                        decode_windows_process_output(stderr)
                        or "ファイル選択を開けませんでした。",
                    )
                return "selected", decode_windows_process_output(stdout)
            except Exception as exc:
                return "error", str(exc)
            finally:
                self._file_dialog_process = None

        self.run_background(worker, self._finish_audio_file_dialog)

    def _open_audio_file_dialog_native(self) -> None:
        """macOS/Linux 向けのファイル選択（tkinter.filedialog）。

        外部プロセスを起動する必要が無く、メインスレッドでモーダル表示して
        戻り値をそのまま使えるため、Windows版のような別プロセス起動・
        トグルで閉じる仕組みは不要。拡張子フィルタはWindows版と同じ8形式。
        """
        assert self.controller is not None
        initial = Path.home() / "Documents"
        if not initial.is_dir():
            initial = Path.home()
        self._file_dialog_active = True
        pattern = " ".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS))
        try:
            selected = filedialog.askopenfilename(
                parent=self,
                title="音声ファイルを選択",
                initialdir=str(initial),
                filetypes=[
                    ("音声ファイル", pattern),
                    ("すべてのファイル", "*.*"),
                ],
            )
        except Exception as exc:
            self._file_dialog_active = False
            self._show_friendly_error(
                "取り込み",
                "ファイル選択の画面を開けませんでした。\n"
                "もう一度「取込」を押して試してください。",
                exc,
            )
            self.route_to("notes")
            return
        self._file_dialog_active = False
        if not selected:
            self.main_pane.update_status("ファイル選択を閉じました")
            self.route_to("notes")
            return
        path = Path(selected)
        self.main_pane.update_status(
            f"選択しました: {path.name}　文字起こしモードを選んでください"
        )
        self.after(150, lambda chosen=path: self._start_selected_audio_import(chosen))

    def _finish_audio_file_dialog(self, result: tuple[str, str]) -> None:
        if not self._file_dialog_active:
            return
        self._file_dialog_active = False
        status, value = result
        if status == "error":
            self._show_friendly_error(
                "取り込み",
                "ファイル選択の画面を開けませんでした。\n"
                "もう一度「取込」を押して試してください。",
                detail=value,
            )
            self.route_to("notes")
            return
        if not value:
            self.main_pane.update_status("ファイル選択を閉じました")
            self.route_to("notes")
            return
        path = Path(value)
        self.main_pane.update_status(
            f"選択しました: {path.name}　文字起こしモードを選んでください"
        )
        # Let the external Windows dialog finish restoring focus before
        # creating our modal CTk window. Otherwise it can open behind the app.
        self.after(150, lambda selected=path: self._start_selected_audio_import(selected))

    def _start_selected_audio_import(self, path: Path) -> None:
        try:
            self._accept_dropped_paths([path])
        except (RuntimeError, ValueError) as exc:
            # 自前の平易な日本語メッセージ（形式違いなど）はそのまま見せる。
            messagebox.showerror("音声ファイル", str(exc), parent=self)
            self.route_to("notes")
        except Exception as exc:
            self._show_friendly_error(
                "音声ファイル",
                "このファイルを取り込めませんでした。\n"
                "もう一度試してください。",
                exc,
            )
            self.route_to("notes")

    def _cancel_audio_file_dialog(self) -> None:
        self._file_dialog_active = False
        process = self._file_dialog_process
        self._file_dialog_process = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        self.main_pane.update_status("ファイル選択を終了しました")
        self.route_to("notes")

    def _enable_drag_and_drop(self) -> None:
        try:
            TkinterDnD.require(self)
            # CTk itself derives directly from tkinter.Tk, while CTkFrame
            # derives from BaseWidget where tkinterdnd2 installs its methods.
            self.main_pane.drop_target_register(DND_FILES)
            self.main_pane.dnd_bind("<<DropEnter>>", self._on_drop_enter)
            self.main_pane.dnd_bind("<<DropLeave>>", self._on_drop_leave)
            self.main_pane.dnd_bind("<<Drop>>", self._on_audio_drop)
        except Exception as exc:
            log_exception("ドラッグ＆ドロップの初期化に失敗しました", exc)
            self.main_pane.update_status(
                "ドラッグ＆ドロップは使えません。左の「取込」ボタンから取り込んでください。"
            )

    def _on_drop_enter(self, event) -> str:
        del event
        self.main_pane.update_status("音声ファイルをドロップして取り込みます")
        return "copy"

    def _on_drop_leave(self, event) -> str:
        del event
        self.main_pane.update_status("ドロップをキャンセルしました")
        return "copy"

    def _on_audio_drop(self, event) -> str:
        try:
            paths = [Path(value) for value in self.tk.splitlist(event.data)]
            self._accept_dropped_paths(paths)
        except (RuntimeError, ValueError) as exc:
            # 自前の平易な日本語メッセージ（1つだけ・形式違いなど）。
            messagebox.showerror("ドラッグ＆ドロップ", str(exc), parent=self)
        except Exception as exc:
            self._show_friendly_error(
                "ドラッグ＆ドロップ",
                "このファイルを取り込めませんでした。\n"
                "もう一度試してください。",
                exc,
            )
        return "copy"

    def _accept_dropped_paths(self, paths: list[Path]) -> None:
        if self.controller is None:
            raise RuntimeError("バックエンドに接続されていません。")
        if self.controller.busy:
            raise RuntimeError("現在処理中です。完了後にお試しください。")
        if len(paths) != 1:
            raise ValueError("音声ファイルを1つだけドロップしてください。")
        path = paths[0]
        if not path.is_file():
            raise ValueError("フォルダーではなく音声ファイルをドロップしてください。")
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            supported = " / ".join(sorted(AUDIO_EXTENSIONS))
            raise ValueError(f"対応していない形式です。対応形式: {supported}")
        self._begin_audio_import(path)

    def _begin_audio_import(self, path: Path) -> None:
        assert self.controller is not None
        dialog = _ModeDialog(
            self,
            "文字起こしモードを選択",
            detail=path.name,
        )
        if dialog.result is None:
            self.route_to("notes")
            return
        try:
            self.controller.import_audio_async(path, dialog.result)
        except FileNotFoundError as exc:
            # 例外文はファイルパスそのものなので、平易な文に置き換える。
            self._show_friendly_error(
                "取り込み",
                "選んだ音声ファイルが見つかりませんでした。\n"
                "ファイルを選び直してください。",
                exc,
            )
            self.route_to("notes")
        except (RuntimeError, ValueError) as exc:
            messagebox.showerror("取り込み", str(exc), parent=self)
            self.route_to("notes")

    def _setup_mac_menu_bar(self) -> None:
        """macOS向けの最小構成のネイティブメニューバー（画面上部）。

        アプリメニューに「OtoWeaveについて」（バージョン/ライセンス画面への
        導線）を追加し、Tkの作法（`tk::mac::Quit` / `tk::mac::ShowPreferences`
        への `createcommand`）でQuit（Cmd+Q）と環境設定相当の導線を割り当てる。
        Windowsではこのメソッド自体が呼ばれないため、Windows側の挙動・UIは
        一切変わらない。
        """
        menu_bar = Menu(self)
        app_menu = Menu(menu_bar, name="apple")
        menu_bar.add_cascade(menu=app_menu)
        app_menu.add_command(label="OtoWeaveについて", command=self._show_about)
        self.tk.createcommand("tk::mac::ShowPreferences", self._show_about)
        self.tk.createcommand("tk::mac::Quit", self._on_close)
        self.config(menu=menu_bar)

    def _show_about(self) -> None:
        """「OtoWeaveについて」: バージョン/ライセンス情報の画面に遷移する。"""
        self.route_to("settings")
        self._show_context("settings", "モデルとライセンス")

    def _on_close(self) -> None:
        if getattr(self, "_file_dialog_active", False):
            self._cancel_audio_file_dialog()
        self._tts.close()
        if self.controller:
            # close() は進行中の文字起こし・要約へ中止を要求する。録音停止の
            # 確定保存はキャンセルせず、完了を待ってから閉じる。
            self.controller.close()
            if self.controller.busy:
                self._close_wait_started = time.monotonic()
                self.main_pane.update_status(close_wait_status(0.0))
                self.after(200, self._wait_and_close)
                return
        self.destroy()

    def _wait_and_close(self) -> None:
        if self.controller and self.controller.busy:
            waited = time.monotonic() - getattr(
                self,
                "_close_wait_started",
                time.monotonic(),
            )
            # 15秒を超えても保存が続くときは、止まっていないことが分かる
            # 文言に切り替えて待ち続ける（強制終了はデータ破損のもと）。
            self.main_pane.update_status(close_wait_status(waited))
            self.after(200, self._wait_and_close)
            return
        self.destroy()

    def _on_stop_recording(self) -> None:
        if self.controller:
            self.controller.stop_lesson_async()

    def _on_pause_recording(self) -> None:
        if self.controller and self.controller.recording:
            paused = self.controller.toggle_recording_pause()
            self.main_pane.pause_button.configure(text="▶ 再開" if paused else "⏸")

    def _on_transcribe_recording(self) -> None:
        if self.controller is None or self._active_folder is None:
            messagebox.showinfo(
                "文字起こし",
                "先に録音済みのノートを選択してください。",
                parent=self,
            )
            return
        if self.controller.busy:
            messagebox.showinfo(
                "文字起こし",
                "要約・録音・音声処理の実行中は開始できません。\n"
                "メモリ不足を避けるため、現在の処理が終わってからお試しください。",
                parent=self,
            )
            return
        try:
            lesson = self.controller.select_lesson(self._active_folder)
        except Exception as exc:
            self._show_friendly_error(
                "文字起こし",
                "このノートを開けませんでした。\n"
                "ノートを選び直して、もう一度試してください。",
                exc,
            )
            return
        if self.controller.current_audio_path() is None:
            messagebox.showinfo(
                "文字起こし",
                "文字起こしできる録音データがありません。",
                parent=self,
            )
            return
        if lesson.segments and not messagebox.askyesno(
            "文字起こしをやり直す",
            "現在の文字起こしを、新しい言語設定で置き換えます。\n"
            "重要・不明瞭・質問のマークは可能な範囲で引き継ぎます。\n\n"
            "続けますか？",
            parent=self,
        ):
            return

        dialog = _TranscriptionModeDialog(
            self,
            "文字起こしモードを選択",
            detail="要約などの処理とは同時に実行されません。",
            diarization_available=diarization_available(self.controller.project_root),
        )
        if dialog.result is None:
            return
        diarization_speakers = getattr(dialog, "diarization_speakers", None)
        extra_kwargs = (
            {"diarization_speakers": diarization_speakers}
            if diarization_speakers is not None
            else {}
        )
        try:
            self.controller.transcribe_current_audio_async(
                dialog.result, **extra_kwargs
            )
            self.main_pane.set_transcribing(True)
            self.main_pane.update_status(
                "文字起こしを開始しました。完了までお待ちください。"
            )
        except (RuntimeError, ValueError) as exc:
            messagebox.showinfo("文字起こし", str(exc), parent=self)

    def _on_diarize_lesson(self) -> None:
        if self.controller is None or self._active_folder is None:
            return
        if self.controller.busy:
            messagebox.showinfo(
                "話者を推定",
                "録音・文字起こし・要約の処理中は開始できません。",
                parent=self,
            )
            return
        try:
            lesson = self.controller.select_lesson(self._active_folder)
        except Exception as exc:
            self._show_friendly_error(
                "話者を推定",
                "このノートを開けませんでした。\n"
                "ノートを選び直して、もう一度試してください。",
                exc,
            )
            return
        if self.controller.current_audio_path() is None:
            messagebox.showinfo(
                "話者を推定",
                "話者を推定できる録音データがありません。",
                parent=self,
            )
            return
        if not lesson.segments:
            messagebox.showinfo(
                "話者を推定",
                "話者を推定できる文字起こしがありません。",
                parent=self,
            )
            return

        already_has_speakers = any(
            segment.speaker for segment in lesson.segments
        )
        if already_has_speakers and not messagebox.askyesno(
            "話者を推定",
            "自動でつけた「話者1」などのラベルは、あらためて推定し直します。\n"
            "あなたが変更した名前はそのまま残ります。\n\n"
            "続けますか？",
            parent=self,
        ):
            return

        dialog = _DiarizeSpeakerCountDialog(self)
        if dialog.result is None:
            return
        try:
            self.controller.diarize_lesson_async(dialog.result)
        except (RuntimeError, ValueError) as exc:
            messagebox.showinfo("話者を推定", str(exc), parent=self)
            return
        self.main_pane.set_diarizing(True)
        self.main_pane.update_status("話者を推定しています…")

    def _save_transcript_text(self, text: str) -> bool:
        if self.controller is None or self._active_folder is None:
            return False
        if self.controller.busy:
            messagebox.showinfo(
                "文字起こしを編集",
                "録音・文字起こし・要約の処理中は保存できません。",
                parent=self,
            )
            return False
        try:
            self.controller.select_lesson(self._active_folder)
            self.controller.replace_transcript_text(text)
            self.controller.reset_chat()
        except (RuntimeError, ValueError) as exc:
            # 自前の平易な日本語メッセージ（処理中・空にできない等）。
            messagebox.showerror(
                "文字起こしを編集",
                str(exc),
                parent=self,
            )
            return False
        except OSError as exc:
            self._show_friendly_error(
                "文字起こしを編集",
                "文字起こしを保存できませんでした。\n"
                "パソコンの空き容量を確認して、もう一度試してください。",
                exc,
            )
            return False
        self._load_lessons()
        self.right_pane.clear_chat()
        self.main_pane.update_status(
            "文字起こしを保存しました。既存の要約は再作成してください。"
        )
        return True

    def _on_rename_speakers(self) -> None:
        if self.controller is None or self._active_folder is None:
            return
        if self.controller.busy:
            messagebox.showinfo(
                "話者名を変更",
                "処理が終わってから話者名を変更してください。",
                parent=self,
            )
            return
        try:
            lesson = self.controller.select_lesson(self._active_folder)
        except Exception as exc:
            self._show_friendly_error(
                "話者名を変更",
                "このノートを開けませんでした。\n"
                "ノートを選び直して、もう一度試してください。",
                exc,
            )
            return

        speakers: list[str] = []
        for segment in lesson.segments:
            if segment.speaker and segment.speaker not in speakers:
                speakers.append(segment.speaker)
        if not speakers:
            messagebox.showinfo(
                "話者名を変更",
                "この記録には話者の情報がありません。",
                parent=self,
            )
            return

        dialog = _SpeakerRenameDialog(self, speakers)
        mapping = dialog.result
        if not mapping:
            return

        try:
            for old_name, new_name in mapping.items():
                # Snapshot the affected ids before rename_speaker mutates
                # segment.speaker in place, then reuse the existing
                # per-segment save path (controller.update_segment_speaker
                # already locks, marks the segment edited, and persists
                # the whole lesson) once per matched segment to save.
                matched_ids = [
                    segment.id
                    for segment in lesson.segments
                    if segment.speaker == old_name
                ]
                rename_speaker(lesson.segments, old_name, new_name)
                for segment_id in matched_ids:
                    self.controller.update_segment_speaker(segment_id, new_name)
        except (RuntimeError, KeyError) as exc:
            messagebox.showerror("話者名を変更", str(exc), parent=self)
            return
        except Exception as exc:
            self._show_friendly_error(
                "話者名を変更",
                "話者名を変更できませんでした。\n"
                "もう一度試してください。",
                exc,
            )
            return

        note = _lesson_to_note(self._active_folder, lesson)
        self._note_map[note["id"]] = note
        self._apply_note(note, update_route=False)
        self.main_pane.update_status("話者名を変更しました。")

    def _on_rename_note(self) -> None:
        if self.controller is None or self._active_folder is None:
            return
        if self.controller.busy:
            messagebox.showinfo(
                "名前変更",
                "処理が終わってから名前を変更してください。",
                parent=self,
            )
            return
        try:
            lesson = self.controller.select_lesson(self._active_folder)
        except Exception as exc:
            self._show_friendly_error(
                "名前変更",
                "このノートを開けませんでした。\n"
                "ノートを選び直して、もう一度試してください。",
                exc,
            )
            return
        dialog = ctk.CTkInputDialog(
            text="新しいノート名を入力してください",
            title="ノート名を変更",
        )
        new_title = dialog.get_input()
        if new_title is None or not new_title.strip():
            return
        try:
            self.controller.rename_current_lesson(new_title.strip())
        except RuntimeError as exc:
            # 自前の平易な日本語メッセージ（処理中など）。
            messagebox.showerror("名前変更", str(exc), parent=self)
            return
        except Exception as exc:
            self._show_friendly_error(
                "名前変更",
                "名前を変更できませんでした。\n"
                "もう一度試してください。",
                exc,
            )
            return
        self.main_pane.update_status(
            f"「{lesson.title}」の名前を変更しています…"
        )

    def _on_delete_note(self) -> None:
        if self.controller is None or self._active_folder is None:
            return
        if self.controller.busy:
            messagebox.showinfo(
                "削除",
                "処理が終わってから削除してください。",
                parent=self,
            )
            return
        try:
            lesson = self.controller.select_lesson(self._active_folder)
        except Exception as exc:
            self._show_friendly_error(
                "削除",
                "このノートを開けませんでした。\n"
                "ノートを選び直して、もう一度試してください。",
                exc,
            )
            return
        source = (
            f"\n元ファイル: {lesson.source_audio_name}"
            if lesson.source_audio_name
            else ""
        )
        if not messagebox.askyesno(
            "ノートを削除",
            f"「{lesson.suggested_title or lesson.title}」を削除しますか？"
            f"{source}\n\n"
            "ノートは保存フォルダー内の「_trash」へ移動し、\n"
            "30日後に完全に削除されます。\n"
            "取り込み元の音声ファイルは削除されません。",
            parent=self,
        ):
            return
        try:
            self.controller.delete_current_lesson()
        except RuntimeError as exc:
            # 自前の平易な日本語メッセージ（処理中・対象外フォルダなど）。
            messagebox.showerror("削除", str(exc), parent=self)
        except Exception as exc:
            self._show_friendly_error(
                "削除",
                "ノートを削除できませんでした。\n"
                "もう一度試してください。",
                exc,
            )

    def _start_elapsed_timer(self) -> None:
        self._elapsed_start = time.monotonic()
        self._tick_elapsed()

    def _tick_elapsed(self) -> None:
        if not self.controller or not self.controller.recording:
            return
        recorder = self.controller.recorder
        if recorder is not None:
            elapsed = int(recorder.elapsed_seconds)
        else:
            elapsed = int(time.monotonic() - self._elapsed_start)
        m, s = divmod(elapsed, 60)
        self.main_pane.set_elapsed(f"{m:02d}:{s:02d}")
        self._elapsed_after_id = self.after(1000, self._tick_elapsed)

    def _stop_elapsed_timer(self) -> None:
        if self._elapsed_after_id:
            self.after_cancel(self._elapsed_after_id)
            self._elapsed_after_id = ""

    # ------------------------------------------------------------------
    # Text-to-speech (offline, Windows built-in voice)
    # ------------------------------------------------------------------

    def _notify_tts_finished(self) -> None:
        """Called from the TTS watcher thread; marshal to the UI thread."""
        self._ui_results.put((self._on_tts_finished, None, None))

    def _notify_tts_error(self, message: str) -> None:
        self._ui_results.put((self.main_pane.update_status, message, None))

    def _on_tts_finished(self, _payload=None) -> None:
        self._tts_target = ""
        self.main_pane.set_speaking_transcript(False)
        self.right_pane.set_speaking_summary(False)

    def _speak(self, target: str, text: str) -> None:
        """Start reading, or stop when the same button is pressed again."""
        if self._tts_target == target and self._tts.speaking:
            self._tts.stop()
            return
        if self.controller is not None:
            self.controller.player.stop()
        if self._tts.speak(text):
            self._tts_target = target
            self.main_pane.set_speaking_transcript(target == "transcript")
            self.right_pane.set_speaking_summary(target == "summary")
            self.main_pane.update_status("読み上げ中（もう一度押すと停止します）")
        else:
            self.main_pane.update_status("読み上げる内容がありません")

    def _on_speak_transcript(self, text: str) -> None:
        self._speak("transcript", text)

    def _on_speak_summary(self, text: str) -> None:
        self._speak("summary", text)

    # ------------------------------------------------------------------
    # Audio playback
    # ------------------------------------------------------------------

    def _on_play_toggle(self) -> None:
        if self.controller is None or self._active_folder is None:
            return
        self._tts.stop()
        player = self.controller.player
        if player.playing:
            paused = player.toggle_pause()
            self.main_pane.set_player_playing(not paused)
            return
        note = self._note_map.get(self.active_note_id) or {}
        if not note.get("has_audio"):
            self.main_pane.update_status("再生できる音声がありません")
            return
        try:
            lesson = self.controller.select_lesson(self._active_folder)
        except Exception as exc:
            log_exception("音声の再生準備に失敗しました", exc)
            self.main_pane.update_status(
                "再生できませんでした。ノートを選び直して、もう一度試してください。"
            )
            return
        path = self.controller.current_audio_path()
        if path is None:
            self.main_pane.update_status("再生できる音声がありません")
            return
        duration = max(
            (seg.end for seg in lesson.segments),
            default=float(note.get("duration_seconds") or 0.0),
        )
        self._player_duration = duration
        player.play(path, 0.0, max(60.0, duration + 1.0))
        self.main_pane.set_player_playing(True)
        self.main_pane.update_status("再生中")

    def _on_segment_play(self, seconds: float) -> None:
        """Play the note audio from a clicked timestamp."""
        if self.controller is None or self._active_folder is None:
            return
        self._tts.stop()
        note = self._note_map.get(self.active_note_id) or {}
        if not note.get("has_audio"):
            self.main_pane.update_status("再生できる音声がありません")
            return
        try:
            lesson = self.controller.select_lesson(self._active_folder)
        except Exception as exc:
            log_exception("音声の再生準備に失敗しました", exc)
            self.main_pane.update_status(
                "再生できませんでした。ノートを選び直して、もう一度試してください。"
            )
            return
        path = self.controller.current_audio_path()
        if path is None:
            self.main_pane.update_status("再生できる音声がありません")
            return
        duration = max(
            (seg.end for seg in lesson.segments),
            default=float(note.get("duration_seconds") or 0.0),
        )
        start = max(0.0, min(float(seconds), duration))
        self._player_duration = duration
        self.controller.player.play(
            path,
            start,
            max(60.0, duration - start + 1.0),
        )
        self.main_pane.set_player_playing(True)
        self.main_pane.update_status(
            f"{int(start) // 60:02d}:{int(start) % 60:02d} から再生中"
        )

    def _on_lesson_finished(self, folder: Path, lesson: LessonRecord) -> None:
        self._stop_elapsed_timer()
        self._load_lessons()
        # Navigate to the finished lesson
        note = _lesson_to_note(folder, lesson)
        self._note_map[note["id"]] = note
        self._apply_note(note)

    def _on_chat_question(self, question: str) -> None:
        if self._active_folder is None or self.controller is None:
            self.right_pane.set_thinking(False)
            return
        from otoweave_app import llm_chat
        model_path = llm_chat.find_chat_model(self.controller.project_root)
        if model_path is None:
            self.right_pane.set_thinking(False)
            self.right_pane.append_answer(
                "⚠ AIの準備ができていません。"
                "アプリを用意した先生・保護者の方に伝えてください。"
            )
            return
        try:
            self.controller.chat_async(question, self._active_folder, model_path)
        except RuntimeError as exc:
            self.right_pane.set_thinking(False)
            self.right_pane.append_answer(f"⚠ {exc}")

    def _on_summarize(self, template_id: str = "lesson_record") -> None:
        if not getattr(self, "_summarize_available", True):
            # 生成UIは非表示のはずだが、万一呼ばれても安全に何もしない。
            messagebox.showinfo(
                "AIようやく",
                SUMMARY_UNAVAILABLE_MESSAGE,
                parent=self,
            )
            return
        if self.controller is None or self._active_folder is None:
            messagebox.showinfo(
                "要約",
                "先に文字起こし済みのノートを選択してください。",
                parent=self,
            )
            return
        folder = self._active_folder
        try:
            lesson = self.controller.store.load(folder)
        except Exception as exc:
            self._show_friendly_error(
                "要約",
                "ノートを読み込めませんでした。\n"
                "ノートを選び直して、もう一度試してください。",
                exc,
            )
            return
        if not lesson.segments:
            messagebox.showinfo(
                "要約",
                "文字起こしデータがありません。",
                parent=self,
            )
            return

        from otoweave_app import llm_chat
        templates = getattr(self, "_summary_templates", None)
        if not templates:
            templates = load_templates(Path(""))
        selected_template = dict(
            template_by_id(templates, template_id)
        )
        selected_template["dictionary"] = glossary_prompt(
            getattr(self, "_dictionary_entries", [])
        )
        self._selected_summary_template_id = str(
            selected_template["id"]
        )

        model_path = llm_chat.find_summarize_model(self.controller.project_root)
        if model_path is None:
            get_logger().error(
                "要約用モデルファイルが見つかりません（models フォルダを確認）"
            )
            messagebox.showerror(
                "要約",
                "AIの準備ができていません。\n"
                "アプリを用意した先生・保護者の方に伝えてください。",
                parent=self,
            )
            return
        cached = inspect_cached_summary(
            folder,
            lesson,
            selected_template,
        )
        if cached["status"] == "generated" and not messagebox.askyesno(
            "要約を再作成",
            "このテンプレートの要約は生成済みです。再生成しますか？",
            parent=self,
        ):
            return
        try:
            self.controller.summarize_async(
                lesson,
                folder,
                model_path,
                selected_template,
            )
            self.main_pane.set_transcribing_blocked(True)
            self.right_pane.set_summarizing(True)
            self.main_pane.update_status(
                "要約を生成中です。数分かかる場合があります。"
            )
        except RuntimeError as exc:
            messagebox.showinfo("要約", str(exc), parent=self)

    def _on_cancel_summary(self) -> None:
        if self.controller is None:
            return
        if self.controller.cancel_summary():
            self.main_pane.update_status("要約をキャンセルしています…")

    def _on_cancel_processing(self) -> None:
        """取り込み・あとから文字起こしの「中止」ボタン。

        実際の停止は次のチャンク境界で行われるため、完了イベント
        （import_finished / transcription_finished）が届くまでは
        「中止中…」表示のまま待つ。
        """
        if self.controller is None:
            return
        if self.controller.cancel_transcription():
            self.main_pane.set_cancel_processing_pending()
            self.main_pane.update_status(
                "中止しています。少しお待ちください…"
            )

    def _manage_summary_templates(self) -> None:
        _TemplateManagerDialog(
            self,
            self._summary_templates,
            self._save_summary_templates,
        )

    def _save_summary_templates(
        self,
        templates: list[dict[str, Any]],
    ) -> None:
        if self._summary_templates_path is None:
            return
        save_custom_templates(self._summary_templates_path, templates)
        self._summary_templates = load_templates(
            self._summary_templates_path
        )
        available_ids = {
            str(value["id"])
            for value in self._summary_templates
        }
        if self._selected_summary_template_id not in available_ids:
            self._selected_summary_template_id = str(
                self._summary_templates[0]["id"]
            )
        self.right_pane.set_templates(
            self._summary_templates,
            self._selected_summary_template_id,
        )
        if self._active_folder is not None:
            self._load_summary(self._active_folder)

    def _on_summary_template_selected(self, template_id: str) -> None:
        self._selected_summary_template_id = template_id
        if self.controller is not None:
            self.controller.reset_chat()
        self.right_pane.clear_chat()
        if self._active_folder is not None:
            self._load_summary(self._active_folder)

    def _manage_dictionary(self) -> None:
        _DictionaryManagerDialog(
            self,
            self._dictionary_entries,
            self._save_dictionary_entries,
        )

    def _save_dictionary_entries(
        self,
        entries: list[dict[str, Any]],
    ) -> None:
        if self._dictionary_path is None:
            return
        save_dictionary(self._dictionary_path, entries)
        self._dictionary_entries = load_dictionary(
            self._dictionary_path
        )
        if self.controller is not None:
            self.controller.reload_dictionary()
        if self.current_route == "dictionary":
            self.main_pane.show_dictionary(
                self._dictionary_entries,
                "登録した言葉",
            )

    def _toggle_ai_focus(self) -> None:
        if self.current_route != "notes":
            return
        self._set_ai_focus(not self._ai_focus_mode)

    def _set_ai_focus(self, focused: bool) -> None:
        self._ai_focus_mode = focused
        if focused:
            self._set_detail_visible(False)
            focused_width = min(
                680,
                max(480, int(self.winfo_width() * 0.42)),
            )
            self._right_width = focused_width
            self.grid_columnconfigure(5, minsize=focused_width)
        else:
            self._right_width = 310
            self.grid_columnconfigure(5, minsize=self._right_width)
            self._set_detail_visible(True)
        self.right_pane.set_focus_mode(focused)

    def _poll_events(self) -> None:
        assert self.controller is not None
        try:
            while True:
                try:
                    kind, payload = self.controller.events.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle_controller_event(kind, payload)
                except Exception as exc:
                    # One bad event must not stop the polling loop; the
                    # backend keeps sending progress and errors through it.
                    log_exception("イベントの表示処理に失敗しました", exc)
                    self.main_pane.update_status(
                        "画面の更新に失敗しました。ノートを選び直してください。"
                    )
        finally:
            self.after(100, self._poll_events)

    def _handle_controller_event(self, kind: str, payload: Any) -> None:
        if kind == "lesson_started":
            folder, lesson = payload
            title = f"録音中  —  {lesson.suggested_title or lesson.title}"
            self.main_pane.show_recording(title)
            self._start_elapsed_timer()
        elif kind == "import_started":
            folder, lesson = payload
            self.main_pane.show_importing(lesson.source_audio_name or "音声ファイル")
        elif kind == "segments_changed":
            lesson = payload
            lines: list[str] = []
            for seg in lesson.segments:
                m, s = divmod(int(seg.start), 60)
                lines.append(f"{m:02d}:{s:02d}  {seg.text}")
            self.main_pane.update_live_transcript("\n\n".join(lines))
        elif kind == "lesson_finished":
            folder, lesson = payload
            self._on_lesson_finished(folder, lesson)
        elif kind == "import_finished":
            folder, lesson = payload
            self.main_pane.finish_importing()
            self._on_lesson_finished(folder, lesson)
        elif kind == "transcription_started":
            self.main_pane.set_transcribing(True)
            self.main_pane.update_status("文字起こしを準備中です…")
        elif kind == "transcription_finished":
            folder, lesson = payload
            self.main_pane.set_transcribing(False)
            if folder == self._active_folder:
                note = _lesson_to_note(folder, lesson)
                self._note_map[note["id"]] = note
                self._apply_note(note, update_route=False)
            self._load_lessons()
        elif kind == "diarization_started":
            self.main_pane.set_diarizing(True)
            self.main_pane.update_status("話者を推定しています…")
        elif kind == "diarization_finished":
            folder, lesson = payload
            self.main_pane.set_diarizing(False)
            if folder == self._active_folder:
                note = _lesson_to_note(folder, lesson)
                self._note_map[note["id"]] = note
                self._apply_note(note, update_route=False)
            self._load_lessons()
        elif kind == "lesson_renamed":
            folder, lesson = payload
            self._active_folder = Path(folder)
            self.active_note_id = lesson.lesson_id
            self.main_pane.update_status("ノート名を変更しました")
            self._load_lessons()
        elif kind == "lesson_deleted":
            deleted = Path(payload)
            if self._active_folder == deleted:
                self._active_folder = None
                self.active_note_id = ""
            self.main_pane.update_status("ノートを削除しました")
            self._load_lessons()
        elif kind == "playback_position":
            self.main_pane.set_player_progress(
                float(payload),
                self._player_duration,
            )
        elif kind == "playback_finished":
            self.main_pane.set_player_playing(False)
        elif kind == "status":
            self.main_pane.update_status(str(payload))
        elif kind == "error":
            messagebox.showerror("エラー", str(payload), parent=self)
        elif kind == "llm_chat_thinking":
            self.right_pane.set_thinking(True)
        elif kind == "llm_chat_chunk":
            # 逐次届く差分テキストをストリーミング中の吹き出しへ追記する。
            # ノート切り替え後に届いた古いノート宛ての差分は無視する。
            # 最終回答は llm_chat_done が全文で置き換えるため重複しない。
            text, folder = payload
            if folder == self._active_folder:
                self.right_pane.append_answer_chunk(str(text))
        elif kind == "llm_started":
            self.main_pane.set_transcribing_blocked(True)
            self.right_pane.set_summarizing(True)
            self.main_pane.update_status(str(payload))
        elif kind == "summary_progress":
            # 要約サブプロセスの進捗。どのステージでも平易な文へ変換する。
            _folder, progress = payload
            text = summary_progress_text(progress)
            self.right_pane.set_summary_progress(text)
            self.main_pane.update_status(text)
        elif kind == "llm_summary_done":
            folder = Path(payload)
            self.main_pane.set_transcribing_blocked(False)
            self.right_pane.set_summarizing(False)
            self.main_pane.update_status("要約が完成しました")
            if folder == self._active_folder:
                self._load_summary(folder)
        elif kind == "llm_chat_done":
            answer, folder = payload
            if folder == self._active_folder:
                self.right_pane.append_answer(str(answer))
            else:
                self.right_pane.set_thinking(False)
        elif kind == "llm_cancelled":
            folder = Path(payload)
            self.main_pane.set_transcribing_blocked(False)
            self.right_pane.set_summarizing(False)
            self.main_pane.update_status("要約をキャンセルしました")
            if folder == self._active_folder:
                self._load_summary(folder)
        elif kind == "llm_error":
            self.main_pane.set_transcribing_blocked(False)
            self.right_pane.set_summarizing(False)
            # 元のエラー文（スタックトレース等を含み得る）はログにだけ残す。
            get_logger().error("要約の実行に失敗しました: %s", payload)
            self.main_pane.update_status(summary_error_status(payload))
        elif kind == "llm_chat_error":
            message, folder = payload
            get_logger().error("チャット回答の生成に失敗しました: %s", message)
            self.right_pane.set_thinking(False)
            if folder == self._active_folder:
                self.right_pane.append_answer(
                    "⚠ 回答を作れませんでした。もう一度質問してみてください。"
                )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_to(self, route: str) -> None:
        if route not in self.ACTION_ROUTES | self.VIEW_ROUTES:
            return
        if self._ai_focus_mode:
            self._set_ai_focus(False)
        self.current_route = route
        self.activity_bar.set_active(route)

        if route in self.ACTION_ROUTES:
            self._set_detail_visible(False)
            self._set_right_visible(route != "live")
            if route == "record":
                self._show_recording_setup()
            elif self.controller and route == "import":
                self._start_import()
            else:
                self.main_pane.show_route(route)
                if route != "live":
                    self.right_pane.show_route(route)
            return

        self._set_detail_visible(True)
        self._set_right_visible(True)
        self.detail_pane.show_view(route)

        if route == "notes":
            if self.active_note_id in self._note_map:
                self._apply_note(self._note_map[self.active_note_id], update_route=False)
            elif self.active_note_id in NOTE_BY_ID:
                self._apply_note(NOTE_BY_ID[self.active_note_id], update_route=False)
        elif route == "dictionary":
            self.main_pane.show_dictionary(
                self._dictionary_entries,
                "登録した言葉",
            )
            self.right_pane.show_route(route)
        else:
            self.main_pane.show_route(route)
            self.right_pane.show_route(route)

    def _set_detail_visible(self, visible: bool) -> None:
        if visible:
            self.grid_columnconfigure(
                1,
                weight=0,
                minsize=self._detail_width,
            )
            self.detail_pane.grid()
            self.grid_columnconfigure(2, weight=0, minsize=6)
            self.detail_resizer.grid()
        else:
            self.detail_pane.grid_remove()
            self.grid_columnconfigure(1, weight=0, minsize=0)
            self.detail_resizer.grid_remove()
            self.grid_columnconfigure(2, weight=0, minsize=0)

    def _set_right_visible(self, visible: bool) -> None:
        self.right_visible = visible
        if visible:
            self.grid_columnconfigure(4, weight=0, minsize=6)
            self.right_resizer.grid()
            self.grid_columnconfigure(
                5,
                weight=0,
                minsize=self._right_width,
            )
            self.right_pane.grid()
        else:
            self.right_pane.grid_remove()
            self.grid_columnconfigure(5, weight=0, minsize=0)
            self.right_resizer.grid_remove()
            self.grid_columnconfigure(4, weight=0, minsize=0)
        self.main_pane.set_right_visible(visible)

    def _toggle_right_pane(self) -> None:
        if self.current_route == "live":
            return
        self._set_right_visible(not self.right_visible)

    def _request_note(
        self,
        note_id: str,
        search_query: str = "",
    ) -> None:
        note = self._note_map.get(note_id) or NOTE_BY_ID.get(note_id)
        if note:
            self._apply_note(note, search_query=search_query)

    def _apply_note(
        self,
        note: dict,
        *,
        update_route: bool = True,
        search_query: str = "",
    ) -> None:
        self.active_note_id = note["id"]
        new_folder = Path(note["_folder"]) if "_folder" in note else None

        if new_folder != self._active_folder:
            if self.controller:
                self.controller.reset_chat()
                self.controller.player.stop()
            self._tts.stop()
            self.right_pane.clear_chat()
            self._active_folder = new_folder

        self.detail_pane.set_active_note(note["id"])
        self.main_pane.show_note(note, search_query=search_query)
        self.right_pane.show_note(note)

        if new_folder and self.controller:
            if not note.get("_loaded", True):
                self._load_note_body(new_folder, search_query)
            self._load_summary(new_folder)

        if update_route:
            self.current_route = "notes"
            self.activity_bar.set_active("notes")
            self.detail_pane.show_view("notes")
            self._set_detail_visible(True)
            self._set_right_visible(True)

    def _request_content_search(self, query: str) -> None:
        """Search transcript bodies in the background (list stays live)."""
        if self.controller is None:
            return
        query_snapshot = query

        def worker() -> set[str]:
            return self.controller.store.search_transcripts(query_snapshot)

        def done(matched_ids: set[str]) -> None:
            self.detail_pane.apply_content_search(query_snapshot, matched_ids)

        self.run_background(worker, done)

    def _load_note_body(self, folder: Path, search_query: str = "") -> None:
        """Load the transcript body of a metadata-only note in the
        background and re-apply it if the note is still selected."""
        assert self.controller is not None
        folder_snapshot = folder

        def worker() -> dict:
            lesson = self.controller.store.load(folder_snapshot)
            return _lesson_to_note(folder_snapshot, lesson)

        def done(full_note: dict) -> None:
            self._note_map[full_note["id"]] = full_note
            if self._active_folder == folder_snapshot:
                self._apply_note(
                    full_note,
                    update_route=False,
                    search_query=search_query,
                )

        self.run_background(worker, done)

    def _load_summary(self, folder: Path) -> None:
        folder_snapshot = folder
        selected_id = getattr(
            self,
            "_selected_summary_template_id",
            "lesson_record",
        )
        dictionary = glossary_prompt(
            getattr(self, "_dictionary_entries", [])
        )
        templates = [
            {**value, "dictionary": dictionary}
            for value in getattr(
                self,
                "_summary_templates",
                load_templates(Path("")),
            )
        ]

        def worker() -> tuple[str, Path, str, dict[str, str], str]:
            if self.controller is None:
                return (
                    "（要約がありません。）",
                    folder_snapshot,
                    "missing",
                    {},
                    selected_id,
                )
            lesson = self.controller.store.load(folder_snapshot)
            statuses: dict[str, str] = {}
            selected_result: dict[str, Any] | None = None
            for template in templates:
                result = inspect_cached_summary(
                    folder_snapshot,
                    lesson,
                    template,
                )
                template_id = str(template["id"])
                statuses[template_id] = str(result["status"])
                if template_id == selected_id:
                    selected_result = result
            if selected_result is None:
                selected_result = {
                    "status": "missing",
                    "text": "",
                }
            activate_cached_summary(folder_snapshot, selected_result)
            status = str(selected_result["status"])
            text = summary_display_text(
                status,
                str(selected_result.get("text", "")),
                getattr(self, "_summarize_available", True),
            )
            return (
                text,
                folder_snapshot,
                status,
                statuses,
                selected_id,
            )

        def apply_if_current(
            result: tuple[str, Path, str, dict[str, str], str],
        ) -> None:
            text, result_folder, status, statuses, result_template_id = result
            if (
                result_folder == self._active_folder
                and result_template_id
                == self._selected_summary_template_id
            ):
                self.right_pane.set_templates(
                    self._summary_templates,
                    result_template_id,
                    statuses,
                )
                self.right_pane.set_summary(text, status)

        self.run_background(worker, apply_if_current)

    def _show_context(self, route: str, item: str) -> None:
        if route not in {"dictionary", "settings"}:
            return
        self.current_route = route
        self.activity_bar.set_active(route)
        self.detail_pane.show_view(route)
        self.detail_pane.set_active_context(route, item)
        self._set_detail_visible(True)
        self._set_right_visible(True)
        if route == "dictionary":
            self.main_pane.show_dictionary(
                self._dictionary_entries,
                item,
            )
        elif route == "settings" and item == "モデルとライセンス":
            lines: list[str] = []
            for model in model_disclosures(self.controller.project_root if self.controller else Path.cwd()):
                lines.extend(
                    (
                        model.name,
                        f"用途: {model.purpose}",
                        f"ライセンス: {model.license_name}",
                        f"状態: {model.status}",
                        f"配布元: {model.source_url}",
                        "",
                    )
                )
            self.main_pane.show_license_info("\n".join(lines).strip())
        else:
            self.main_pane.show_route(route, item)
        self.right_pane.show_route(route, item)

    @staticmethod
    def _text_size_points(text_size: str) -> int:
        return {
            "Small": 12,
            "Standard": 14,
            "Large": 16,
            "Extra Large": 18,
        }.get(text_size, 14)

    @staticmethod
    def _points_text_size(points: int) -> str:
        return {
            12: "Small",
            14: "Standard",
            16: "Large",
            18: "Extra Large",
        }.get(points, "Standard")

    def _preferences_from_settings(self) -> dict[str, Any]:
        settings = self._display_settings
        return {
            "font_family": settings.font_family,
            "font_size": self._text_size_points(settings.text_size),
            "line_spacing": settings.line_spacing,
            "text_width": settings.text_width,
            "color_mode": settings.color_mode,
            "live_follow": settings.live_follow,
            "highlight_current": settings.highlight_current,
        }

    def _change_display_preferences(
        self,
        preferences: dict[str, Any],
    ) -> None:
        # DetailPane re-requests after rebuilding its note buttons; only
        # write to disk when a preference actually changed.
        persist = preferences != self._preferences_from_settings()
        self._apply_display_preferences(preferences, persist=persist)

    def _apply_display_preferences(
        self,
        preferences: dict[str, Any] | None = None,
        *,
        persist: bool = False,
    ) -> None:
        merged = self._preferences_from_settings()
        if preferences:
            merged.update(preferences)
        font_family = str(merged["font_family"])
        font_size = int(merged["font_size"])
        line_spacing = str(merged["line_spacing"])
        text_width = str(merged["text_width"])
        color_mode = str(merged["color_mode"])
        live_follow = bool(merged["live_follow"])
        highlight_current = bool(merged["highlight_current"])

        ctk.set_appearance_mode(color_mode)

        delta = font_size - BASE_FONT_SIZE
        seen: set[int] = set()

        def visit(widget) -> None:
            try:
                font = widget.cget("font")
            except Exception:
                font = None
            if isinstance(font, ctk.CTkFont):
                key = id(font)
                if key not in self._font_baselines:
                    self._font_baselines[key] = (font, int(font.cget("size")))
                if key not in seen:
                    _, base_size = self._font_baselines[key]
                    font.configure(
                        family=font_family,
                        size=max(9, base_size + delta),
                    )
                    seen.add(key)
            for child in widget.winfo_children():
                visit(child)

        visit(self)
        # Drop baselines of fonts whose widgets no longer exist. Note lists
        # are rebuilt frequently, so keeping them would leak fonts.
        self._font_baselines = {
            key: value
            for key, value in self._font_baselines.items()
            if key in seen
        }

        spacing = 12 if line_spacing == "Comfortable" else 6
        padx = 80 if text_width == "Reading Width" else 18
        self.main_pane.apply_text_layout(spacing=spacing, padx=padx)
        self.right_pane.apply_text_layout(spacing=max(4, spacing - 4))
        self.main_pane.set_live_options(live_follow, highlight_current)
        # Plain tkinter Text tag colors (timestamp underline, speaker
        # colors) don't auto-update on a theme/font change like CTk
        # widget colors do, so re-apply them explicitly here.
        self.main_pane.refresh_transcript_tags()

        settings = self._display_settings
        settings.font_family = font_family
        settings.text_size = self._points_text_size(font_size)
        settings.line_spacing = line_spacing
        settings.text_width = text_width
        settings.color_mode = color_mode
        settings.live_follow = live_follow
        settings.highlight_current = highlight_current
        if persist and self._display_settings_path is not None:
            save_display_settings(
                self._display_settings_path,
                settings,
            )
        self.main_pane.update_status(
            f"表示設定: {font_family} / {font_size}pt"
        )

    # ------------------------------------------------------------------
    # Background runner + UI result drain
    # ------------------------------------------------------------------

    def run_background(
        self,
        worker: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
    ) -> None:
        def run() -> None:
            try:
                result = worker()
                self._ui_results.put((on_success, result, None))
            except Exception as exc:
                self._ui_results.put((None, None, exc))
        threading.Thread(target=run, daemon=True).start()

    def _drain_ui_results(self) -> None:
        try:
            while True:
                try:
                    callback, result, error = self._ui_results.get_nowait()
                except queue.Empty:
                    break
                if error is not None:
                    log_exception("バックグラウンド処理に失敗しました", error)
                    self.main_pane.status_label.configure(
                        text="うまくいきませんでした。もう一度試してください。"
                    )
                elif callback is not None:
                    try:
                        callback(result)
                    except Exception as exc:
                        log_exception("画面の更新に失敗しました", exc)
                        self.main_pane.status_label.configure(
                            text="うまくいきませんでした。もう一度試してください。"
                        )
        finally:
            self.after(40, self._drain_ui_results)


def main() -> int:
    import argparse

    # データルート確定前の失敗も拾えるよう、まず一時的な場所へログを構える。
    try:
        setup_logging(default_log_dir())
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="OtoWeave local transcription app")
    parser.add_argument("--data-root", default="", help="Local lesson storage folder")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create a sample lesson when storage is empty",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    default_root = (
        Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Documents" / "OtoWeave"
    )
    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else default_root
    )
    # データルートが決まったので、ログを <データルート>/logs へ切り替える。
    try:
        setup_logging(data_root / "logs")
    except Exception:
        pass
    get_logger().info("OtoWeave を起動します")
    controller = LearningAccessController(project_root, data_root)
    if args.demo:
        controller.create_demo_lesson()
    app = OtoWeaveApp(controller)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
