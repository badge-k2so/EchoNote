"""Integration tests for the OtoWeaveApp layer (customtkinter_app.py).

Covers:
  - _lesson_to_note() pure conversion
  - _apply_lessons() grouping and note-map population
  - _handle_controller_event() llm_chat_done folder check (session mixing fix)
  - _apply_note() stale-folder summary race fix
  - DetailPane._request_search() real-data vs. mock-data selection
"""
from __future__ import annotations

import unittest
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from otoweave_app.models import LessonRecord, TranscriptSegment
from otoweave_app.customtkinter_views import (
    COLORS,
    SPEAKER_COLOR_KEYS,
    ActivityBar,
    DetailPane,
    MainPane,
    theme_color,
)

# Importing customtkinter_app runs ctk.set_appearance_mode / set_default_color_theme
# which are safe on Windows without a visible window.
from customtkinter_app import OtoWeaveApp, _lesson_to_note
from otoweave_app.otoweave_app import _build_speaker_rename_mapping


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_lesson(**kwargs) -> LessonRecord:
    lesson = LessonRecord.create("japanese", "microphone")
    for k, v in kwargs.items():
        setattr(lesson, k, v)
    return lesson


def _fake_app() -> SimpleNamespace:
    """A minimal stand-in for OtoWeaveApp that exercises real logic without Tk."""
    return SimpleNamespace(
        detail_pane=SimpleNamespace(
            populate_notes=Mock(),
            set_active_note=Mock(),
            show_view=Mock(),
        ),
        main_pane=SimpleNamespace(
            show_note=Mock(),
            show_recording=Mock(),
            update_live_transcript=Mock(),
            update_status=Mock(),
            show_route=Mock(),
            show_license_info=Mock(),
            set_transcribing=Mock(),
            set_transcribing_blocked=Mock(),
            set_empty_note=Mock(),
            set_diarizing=Mock(),
        ),
        right_pane=SimpleNamespace(
            show_note=Mock(),
            set_summary=Mock(),
            append_answer=Mock(),
            set_thinking=Mock(),
            set_summarizing=Mock(),
            show_route=Mock(),
            clear_chat=Mock(),
            set_empty_note=Mock(),
            set_templates=Mock(),
        ),
        activity_bar=SimpleNamespace(set_active=Mock()),
        active_note_id="",
        current_route="notes",
        right_visible=True,
        _active_folder=None,
        _note_map={},
        _elapsed_start=0.0,
        _elapsed_after_id="",
        _file_dialog_active=False,
        _file_dialog_process=None,
        _selected_summary_template_id="lesson_record",
        _summary_templates=[],
        _dictionary_entries=[],
        controller=None,
        route_to=Mock(),
        run_background=Mock(),
        after=Mock(),
        after_cancel=Mock(),
        _apply_note=Mock(),
        _load_lessons=Mock(),
        _load_summary=Mock(),
        _apply_display_preferences=Mock(),
        _display_settings=SimpleNamespace(
            font_family="Meiryo",
            text_size="Standard",
        ),
        _text_size_points=OtoWeaveApp._text_size_points,
        _set_detail_visible=Mock(),
        _set_right_visible=Mock(),
        _tts=SimpleNamespace(
            speak=Mock(return_value=True),
            stop=Mock(),
            close=Mock(),
            speaking=False,
        ),
        _tts_target="",
    )


# ---------------------------------------------------------------------------
# 1. _lesson_to_note() conversion
# ---------------------------------------------------------------------------

class LessonToNoteConversionTests(unittest.TestCase):
    def test_date_and_duration_appear_in_meta(self) -> None:
        lesson = LessonRecord.create(
            "japanese", "microphone",
            now=datetime.fromisoformat("2026-04-15T10:00:00+09:00"),
        )
        lesson.segments = [TranscriptSegment("seg_0001", 0.0, 300.0, "テスト")]
        note = _lesson_to_note(Path("/tmp/lesson"), lesson)
        self.assertIn("4/15", note["label"])
        self.assertIn("2026年4月15日", note["meta"])
        self.assertIn("05:00", note["meta"])

    def test_empty_segments_show_placeholder(self) -> None:
        lesson = _make_lesson()
        note = _lesson_to_note(Path("/tmp/lesson"), lesson)
        self.assertIn("文字起こしデータがありません", note["transcript"])

    def test_important_and_question_flags_in_keywords(self) -> None:
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 5.0, "重要", important=True),
            TranscriptSegment("seg_0002", 5.0, 10.0, "質問", question=True),
        ]
        note = _lesson_to_note(Path("/tmp/lesson"), lesson)
        self.assertIn("★重要", note["keywords"])
        self.assertIn("？質問", note["keywords"])

    def test_folder_path_stored_in_note(self) -> None:
        lesson = _make_lesson()
        folder = Path("/tmp/my_lesson")
        note = _lesson_to_note(folder, lesson)
        self.assertEqual(note["_folder"], str(folder))

    def test_lesson_id_used_as_note_id(self) -> None:
        lesson = _make_lesson()
        note = _lesson_to_note(Path("/tmp/lesson"), lesson)
        self.assertEqual(note["id"], lesson.lesson_id)

    def test_invalid_date_falls_back_gracefully(self) -> None:
        lesson = _make_lesson()
        lesson.date = "not-a-date"
        note = _lesson_to_note(Path("/tmp/lesson"), lesson)
        self.assertIsNotNone(note)
        self.assertEqual(note["id"], lesson.lesson_id)

    def test_transcript_lines_include_timestamps(self) -> None:
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 65.0, 70.0, "一行目"),
            TranscriptSegment("seg_0002", 125.0, 130.0, "二行目"),
        ]
        note = _lesson_to_note(Path("/tmp/lesson"), lesson)
        self.assertIn("01:05", note["transcript"])
        self.assertIn("02:05", note["transcript"])
        self.assertIn("一行目", note["transcript"])
        self.assertIn("二行目", note["transcript"])

    def test_speaker_lines_and_has_speakers_reflect_segments(self) -> None:
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 5.0, "おはよう", speaker="話者1"),
            TranscriptSegment("seg_0002", 5.0, 10.0, "おはよう", speaker="話者2"),
        ]
        note = _lesson_to_note(Path("/tmp/lesson"), lesson)
        self.assertEqual(note["speaker_lines"], ["話者1", "話者2"])
        self.assertTrue(note["has_speakers"])

    def test_no_speakers_reports_has_speakers_false(self) -> None:
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 5.0, "普通の文"),
        ]
        note = _lesson_to_note(Path("/tmp/lesson"), lesson)
        self.assertEqual(note["speaker_lines"], [""])
        self.assertFalse(note["has_speakers"])

    def test_no_marks_shows_transcribed_label(self) -> None:
        lesson = _make_lesson()
        lesson.segments = [TranscriptSegment("seg_0001", 0.0, 5.0, "普通の文")]
        note = _lesson_to_note(Path("/tmp/lesson"), lesson)
        self.assertEqual(note["keywords"], "文字起こし済み")

    def test_imported_source_filename_is_visible_in_note(self) -> None:
        lesson = _make_lesson()
        lesson.source_audio_name = "2025-12-22 消防本部見学.ogg"
        note = _lesson_to_note(Path("/tmp/lesson"), lesson)
        self.assertEqual(
            note["source_audio_name"],
            "2025-12-22 消防本部見学.ogg",
        )
        self.assertIn("元ファイル: 2025-12-22 消防本部見学.ogg", note["meta"])

    def test_saved_audio_without_segments_is_marked_transcribable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            audio = folder / "audio.opus"
            audio.write_bytes(b"OggS")
            lesson = _make_lesson()
            lesson.audio_file = audio.name
            lesson.segments = []

            note = _lesson_to_note(folder, lesson)

        self.assertTrue(note["has_audio"])
        self.assertFalse(note["has_transcript"])


# ---------------------------------------------------------------------------
# 2. _apply_lessons() — note-map grouping
# ---------------------------------------------------------------------------

class OtoWeaveAppLessonLoadingTests(unittest.TestCase):
    def test_activity_bar_keeps_notes_and_removes_duplicate_search_route(self) -> None:
        routes = [item[0] for item in ActivityBar.ITEMS]
        self.assertIn("notes", routes)
        self.assertNotIn("search", routes)
        self.assertNotIn("live", routes)
        self.assertEqual(routes.count("record"), 1)

    def test_activity_icons_are_colored_images(self) -> None:
        icons = {
            route: ActivityBar._draw_activity_icon(route)
            for route, _symbol, _label in ActivityBar.ITEMS
        }

        self.assertTrue(all(icon.mode == "RGBA" for icon in icons.values()))
        visible_colors = {
            route: {
                pixel[:3]
                for pixel in icon.get_flattened_data()
                if pixel[3] > 0
            }
            for route, icon in icons.items()
        }
        self.assertTrue(all(len(colors) >= 2 for colors in visible_colors.values()))
        self.assertEqual(
            len({icon.getpixel((32, 32)) for icon in icons.values()}),
            5,
        )

    def test_activity_tooltip_labels_explain_each_action(self) -> None:
        labels = {
            route: label
            for route, _symbol, label in ActivityBar.ITEMS
        }

        self.assertEqual(labels["import"], "音声ファイルを取り込む")
        self.assertEqual(labels["dictionary"], "補正辞書")
        self.assertEqual(labels["settings"], "設定")

    def test_apply_lessons_builds_note_map_for_all_lessons(self) -> None:
        app = _fake_app()
        lesson1 = LessonRecord.create(
            "japanese", "microphone",
            now=datetime.fromisoformat("2026-04-15T10:00:00+09:00"),
        )
        lesson2 = LessonRecord.create(
            "japanese", "microphone",
            now=datetime.fromisoformat("2026-05-10T10:00:00+09:00"),
        )
        OtoWeaveApp._apply_lessons(app, [
            (Path("/tmp/l1"), lesson1),
            (Path("/tmp/l2"), lesson2),
        ])
        self.assertIn(lesson1.lesson_id, app._note_map)
        self.assertIn(lesson2.lesson_id, app._note_map)

    def test_apply_lessons_groups_by_year_month(self) -> None:
        app = _fake_app()
        lesson1 = LessonRecord.create(
            "japanese", "microphone",
            now=datetime.fromisoformat("2026-04-15T10:00:00+09:00"),
        )
        lesson2 = LessonRecord.create(
            "japanese", "microphone",
            now=datetime.fromisoformat("2026-05-10T10:00:00+09:00"),
        )
        OtoWeaveApp._apply_lessons(app, [
            (Path("/tmp/l1"), lesson1),
            (Path("/tmp/l2"), lesson2),
        ])
        groups = app.detail_pane.populate_notes.call_args[0][0]
        self.assertIn("2026年 4月", groups)
        self.assertIn("2026年 5月", groups)

    def test_apply_lessons_calls_route_to_notes(self) -> None:
        app = _fake_app()
        lesson = LessonRecord.create(
            "japanese", "microphone",
            now=datetime.fromisoformat("2026-04-15T10:00:00+09:00"),
        )
        OtoWeaveApp._apply_lessons(app, [(Path("/tmp/l1"), lesson)])
        app.route_to.assert_called_once_with("notes")

    def test_apply_lessons_calls_apply_note_with_first_lesson(self) -> None:
        app = _fake_app()
        lesson1 = LessonRecord.create(
            "japanese", "microphone",
            now=datetime.fromisoformat("2026-04-15T10:00:00+09:00"),
        )
        lesson2 = LessonRecord.create(
            "japanese", "microphone",
            now=datetime.fromisoformat("2026-05-10T10:00:00+09:00"),
        )
        OtoWeaveApp._apply_lessons(app, [
            (Path("/tmp/l1"), lesson1),
            (Path("/tmp/l2"), lesson2),
        ])
        app._apply_note.assert_called_once()
        first_note_arg = app._apply_note.call_args[0][0]
        self.assertEqual(first_note_arg["id"], lesson1.lesson_id)

    def test_apply_lessons_on_empty_list_skips_apply_note(self) -> None:
        app = _fake_app()
        OtoWeaveApp._apply_lessons(app, [])
        app._apply_note.assert_not_called()
        app.route_to.assert_called_once_with("notes")


# ---------------------------------------------------------------------------
# 3. _handle_controller_event() — llm_chat_done folder check
# ---------------------------------------------------------------------------

class OtoWeaveAppChatSessionTests(unittest.TestCase):
    def test_chat_answer_for_current_folder_is_shown(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson_a")
        app._active_folder = folder

        OtoWeaveApp._handle_controller_event(app, "llm_chat_done", ("answer text", folder))

        app.right_pane.append_answer.assert_called_once_with("answer text")

    def test_chat_answer_for_different_folder_is_ignored(self) -> None:
        app = _fake_app()
        app._active_folder = Path("/tmp/lesson_a")

        OtoWeaveApp._handle_controller_event(
            app, "llm_chat_done", ("stale answer", Path("/tmp/lesson_b"))
        )

        app.right_pane.append_answer.assert_not_called()

    def test_chat_answer_when_no_folder_is_active_is_ignored(self) -> None:
        app = _fake_app()
        app._active_folder = None

        OtoWeaveApp._handle_controller_event(
            app, "llm_chat_done", ("answer", Path("/tmp/lesson_a"))
        )

        app.right_pane.append_answer.assert_not_called()

    def test_llm_chat_thinking_sets_thinking_state(self) -> None:
        app = _fake_app()
        OtoWeaveApp._handle_controller_event(app, "llm_chat_thinking", None)
        app.right_pane.set_thinking.assert_called_once_with(True)

    def test_llm_error_shows_in_status_bar_not_chat(self) -> None:
        """llm_error is summary-side; must not write to the chat area."""
        app = _fake_app()
        OtoWeaveApp._handle_controller_event(app, "llm_error", "summary failed")
        app.main_pane.update_status.assert_called_once()
        shown = app.main_pane.update_status.call_args[0][0]
        # 技術的なエラー文はログへ送り、画面には平易な日本語だけを出す。
        self.assertNotIn("summary failed", shown)
        self.assertIn("要約", shown)
        app.right_pane.append_answer.assert_not_called()

    def test_llm_chat_error_for_current_folder_shows_in_chat(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson_a")
        app._active_folder = folder
        OtoWeaveApp._handle_controller_event(app, "llm_chat_error", ("inference failed", folder))
        app.right_pane.set_thinking.assert_called_once_with(False)
        app.right_pane.append_answer.assert_called_once()
        # 生の例外文は表示せず、平易な日本語の吹き出しを出す。
        shown = app.right_pane.append_answer.call_args[0][0]
        self.assertNotIn("inference failed", shown)
        self.assertIn("もう一度", shown)

    def test_llm_chat_error_for_wrong_folder_clears_thinking_silently(self) -> None:
        """Fix 2: error from stale session must not appear in current session."""
        app = _fake_app()
        app._active_folder = Path("/tmp/lesson_a")
        OtoWeaveApp._handle_controller_event(
            app, "llm_chat_error", ("inference failed", Path("/tmp/lesson_b"))
        )
        app.right_pane.set_thinking.assert_called_once_with(False)
        app.right_pane.append_answer.assert_not_called()

    def test_stale_chat_done_releases_thinking_state(self) -> None:
        """Fix 2: stale answer must unlock the input field."""
        app = _fake_app()
        app._active_folder = Path("/tmp/lesson_a")

        OtoWeaveApp._handle_controller_event(
            app, "llm_chat_done", ("stale answer", Path("/tmp/lesson_b"))
        )

        app.right_pane.set_thinking.assert_called_once_with(False)
        app.right_pane.append_answer.assert_not_called()


class OtoWeaveAppSummarizeTests(unittest.TestCase):
    def test_summarize_uses_the_visibly_selected_folder(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/selected_lesson")
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 5.0, "要約する本文")
        ]
        model = Path("/tmp/Qwen3.5-4B-Q4_K_M.gguf")
        app._active_folder = folder
        app.controller = SimpleNamespace(
            store=SimpleNamespace(load=Mock(return_value=lesson)),
            project_root=Path("/tmp"),
            summarize_async=Mock(),
        )

        with (
            patch(
                "otoweave_app.llm_chat.find_summarize_model",
                return_value=model,
            ),
            patch(
                "otoweave_app.llm_chat.has_summary",
                return_value=False,
            ),
        ):
            OtoWeaveApp._on_summarize(app)

        app.controller.store.load.assert_called_once_with(folder)
        call = app.controller.summarize_async.call_args
        self.assertEqual(call.args[:3], (lesson, folder, model))
        self.assertEqual(call.args[3]["id"], "lesson_record")
        app.right_pane.set_summarizing.assert_called_once_with(True)

    def test_summary_done_refreshes_only_the_active_folder(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/selected_lesson")
        app._active_folder = folder

        OtoWeaveApp._handle_controller_event(app, "llm_summary_done", folder)

        app.right_pane.set_summarizing.assert_called_once_with(False)
        app._load_summary.assert_called_once_with(folder)

    def test_summary_done_does_not_replace_another_note(self) -> None:
        app = _fake_app()
        app._active_folder = Path("/tmp/current")

        OtoWeaveApp._handle_controller_event(
            app,
            "llm_summary_done",
            Path("/tmp/old"),
        )

        app._load_summary.assert_not_called()


class OtoWeaveAppTranscriptionTests(unittest.TestCase):
    def test_recording_setup_is_forwarded_to_controller(self) -> None:
        source = SimpleNamespace(kind="microphone")
        controller = SimpleNamespace(
            busy=False,
            start_lesson=Mock(),
        )
        app = SimpleNamespace(
            controller=controller,
            _tts=SimpleNamespace(stop=Mock()),
        )
        options = {
            "source": source,
            "noise_reduction": True,
            "sensitivity": 1.4,
            "automatic_gain_control": True,
            "speaker_label": "自分",
        }

        with patch(
            "otoweave_app.otoweave_app._ModeDialog",
            return_value=SimpleNamespace(result="japanese"),
        ):
            OtoWeaveApp._start_record(app, options)

        call = controller.start_lesson.call_args
        self.assertEqual(call.args, ("japanese", source))
        self.assertTrue(call.kwargs["processing"].noise_reduction)
        self.assertEqual(call.kwargs["processing"].sensitivity, 1.4)
        self.assertTrue(call.kwargs["processing"].automatic_gain_control)
        self.assertEqual(call.kwargs["speaker_label"], "自分")

    def test_summary_busy_state_blocks_transcription_before_loading_audio(self) -> None:
        app = _fake_app()
        app._active_folder = Path("/tmp/recording")
        app.controller = SimpleNamespace(
            busy=True,
            select_lesson=Mock(),
        )

        with patch("otoweave_app.otoweave_app.messagebox.showinfo") as showinfo:
            OtoWeaveApp._on_transcribe_recording(app)

        app.controller.select_lesson.assert_not_called()
        self.assertIn("メモリ不足", showinfo.call_args[0][1])

    def test_saved_recording_can_start_low_memory_mixed_transcription(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/recording")
        audio = folder / "audio.opus"
        lesson = _make_lesson()
        lesson.segments = []
        app._active_folder = folder
        app.controller = SimpleNamespace(
            busy=False,
            project_root=Path("/tmp/project_root"),
            select_lesson=Mock(return_value=lesson),
            current_audio_path=Mock(return_value=audio),
            transcribe_current_audio_async=Mock(),
        )

        with patch(
            "otoweave_app.otoweave_app._TranscriptionModeDialog",
            return_value=SimpleNamespace(result="mixed"),
        ):
            OtoWeaveApp._on_transcribe_recording(app)

        app.controller.select_lesson.assert_called_once_with(folder)
        app.controller.transcribe_current_audio_async.assert_called_once_with(
            "mixed"
        )
        app.main_pane.set_transcribing.assert_called_once_with(True)

    def test_diarization_choice_is_forwarded_to_controller(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/recording")
        audio = folder / "audio.opus"
        lesson = _make_lesson()
        lesson.segments = []
        app._active_folder = folder
        app.controller = SimpleNamespace(
            busy=False,
            project_root=Path("/tmp/project_root"),
            select_lesson=Mock(return_value=lesson),
            current_audio_path=Mock(return_value=audio),
            transcribe_current_audio_async=Mock(),
        )

        with patch(
            "otoweave_app.otoweave_app._TranscriptionModeDialog",
            return_value=SimpleNamespace(result="japanese", diarization_speakers=2),
        ):
            OtoWeaveApp._on_transcribe_recording(app)

        app.controller.transcribe_current_audio_async.assert_called_once_with(
            "japanese", diarization_speakers=2
        )

    def test_diarization_dialog_receives_availability_flag(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/recording")
        audio = folder / "audio.opus"
        lesson = _make_lesson()
        lesson.segments = []
        app._active_folder = folder
        app.controller = SimpleNamespace(
            busy=False,
            project_root=Path("/tmp/project_root"),
            select_lesson=Mock(return_value=lesson),
            current_audio_path=Mock(return_value=audio),
            transcribe_current_audio_async=Mock(),
        )

        with (
            patch(
                "otoweave_app.otoweave_app.diarization_available",
                return_value=True,
            ) as mock_available,
            patch(
                "otoweave_app.otoweave_app._TranscriptionModeDialog",
                return_value=SimpleNamespace(result="japanese"),
            ) as mock_dialog,
        ):
            OtoWeaveApp._on_transcribe_recording(app)

        mock_available.assert_called_once_with(app.controller.project_root)
        self.assertTrue(mock_dialog.call_args.kwargs["diarization_available"])

    def test_existing_transcript_can_be_reprocessed_in_another_language(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/recording")
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 3.0, "誤った言語の結果")
        ]
        app._active_folder = folder
        app.controller = SimpleNamespace(
            busy=False,
            project_root=Path("/tmp/project_root"),
            select_lesson=Mock(return_value=lesson),
            current_audio_path=Mock(return_value=folder / "audio.opus"),
            transcribe_current_audio_async=Mock(),
        )

        with (
            patch("otoweave_app.otoweave_app.messagebox.askyesno", return_value=True),
            patch(
                "otoweave_app.otoweave_app._TranscriptionModeDialog",
                return_value=SimpleNamespace(result="english"),
            ),
        ):
            OtoWeaveApp._on_transcribe_recording(app)

        app.controller.transcribe_current_audio_async.assert_called_once_with(
            "english"
        )

    def test_transcription_finished_releases_busy_button(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/recording")
        lesson = _make_lesson()
        app._active_folder = Path("/tmp/another")

        OtoWeaveApp._handle_controller_event(
            app,
            "transcription_finished",
            (folder, lesson),
        )

        app.main_pane.set_transcribing.assert_called_once_with(False)


class OtoWeaveAppDiarizeLessonTests(unittest.TestCase):
    """後がけ話者分離: 「話者を推定」ボタン→人数ダイアログ→controller呼び出し。"""

    def _controller(self, lesson, folder: Path, **overrides) -> SimpleNamespace:
        base = dict(
            busy=False,
            project_root=Path("/tmp/project_root"),
            select_lesson=Mock(return_value=lesson),
            current_audio_path=Mock(return_value=folder / "audio.opus"),
            diarize_lesson_async=Mock(),
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_button_to_dialog_to_controller_call(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 5.0, "おはよう"),
        ]
        app._active_folder = folder
        app.controller = self._controller(lesson, folder)
        dialog = SimpleNamespace(result=3)

        with patch(
            "otoweave_app.otoweave_app._DiarizeSpeakerCountDialog",
            return_value=dialog,
        ) as dialog_cls:
            OtoWeaveApp._on_diarize_lesson(app)

        dialog_cls.assert_called_once_with(app)
        app.controller.diarize_lesson_async.assert_called_once_with(3)
        app.main_pane.set_diarizing.assert_called_once_with(True)
        app.main_pane.update_status.assert_called_once()

    def test_cancelled_dialog_does_not_call_controller(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        lesson.segments = [TranscriptSegment("seg_0001", 0.0, 5.0, "おはよう")]
        app._active_folder = folder
        app.controller = self._controller(lesson, folder)
        dialog = SimpleNamespace(result=None)

        with patch(
            "otoweave_app.otoweave_app._DiarizeSpeakerCountDialog",
            return_value=dialog,
        ):
            OtoWeaveApp._on_diarize_lesson(app)

        app.controller.diarize_lesson_async.assert_not_called()
        app.main_pane.set_diarizing.assert_not_called()

    def test_blocked_while_controller_busy(self) -> None:
        app = _fake_app()
        app._active_folder = Path("/tmp/lesson")
        app.controller = SimpleNamespace(busy=True, select_lesson=Mock())

        with patch("otoweave_app.otoweave_app.messagebox.showinfo"):
            OtoWeaveApp._on_diarize_lesson(app)

        app.controller.select_lesson.assert_not_called()

    def test_no_audio_shows_info_and_stops(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        lesson.segments = [TranscriptSegment("seg_0001", 0.0, 5.0, "おはよう")]
        app._active_folder = folder
        app.controller = self._controller(
            lesson, folder, current_audio_path=Mock(return_value=None)
        )

        with patch(
            "otoweave_app.otoweave_app._DiarizeSpeakerCountDialog",
        ) as dialog_cls, patch(
            "otoweave_app.otoweave_app.messagebox.showinfo",
        ):
            OtoWeaveApp._on_diarize_lesson(app)

        dialog_cls.assert_not_called()

    def test_empty_transcript_shows_info_and_stops(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        lesson.segments = []
        app._active_folder = folder
        app.controller = self._controller(lesson, folder)

        with patch(
            "otoweave_app.otoweave_app._DiarizeSpeakerCountDialog",
        ) as dialog_cls, patch(
            "otoweave_app.otoweave_app.messagebox.showinfo",
        ):
            OtoWeaveApp._on_diarize_lesson(app)

        dialog_cls.assert_not_called()

    def test_existing_speakers_require_confirmation_before_dialog(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 5.0, "おはよう", speaker="話者1"),
        ]
        app._active_folder = folder
        app.controller = self._controller(lesson, folder)

        with patch(
            "otoweave_app.otoweave_app.messagebox.askyesno",
            return_value=False,
        ) as confirm, patch(
            "otoweave_app.otoweave_app._DiarizeSpeakerCountDialog",
        ) as dialog_cls:
            OtoWeaveApp._on_diarize_lesson(app)

        confirm.assert_called_once()
        dialog_cls.assert_not_called()
        app.controller.diarize_lesson_async.assert_not_called()

    def test_existing_speakers_confirmed_reaches_dialog(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 5.0, "おはよう", speaker="話者1"),
        ]
        app._active_folder = folder
        app.controller = self._controller(lesson, folder)
        dialog = SimpleNamespace(result=2)

        with patch(
            "otoweave_app.otoweave_app.messagebox.askyesno",
            return_value=True,
        ), patch(
            "otoweave_app.otoweave_app._DiarizeSpeakerCountDialog",
            return_value=dialog,
        ):
            OtoWeaveApp._on_diarize_lesson(app)

        app.controller.diarize_lesson_async.assert_called_once_with(2)

    def test_synchronous_controller_error_is_shown_as_friendly_message(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        lesson.segments = [TranscriptSegment("seg_0001", 0.0, 5.0, "おはよう")]
        app._active_folder = folder
        app.controller = self._controller(
            lesson,
            folder,
            diarize_lesson_async=Mock(
                side_effect=RuntimeError("別の録音処理が進行中です。")
            ),
        )
        dialog = SimpleNamespace(result=2)

        with patch(
            "otoweave_app.otoweave_app._DiarizeSpeakerCountDialog",
            return_value=dialog,
        ), patch(
            "otoweave_app.otoweave_app.messagebox.showinfo",
        ) as showinfo:
            OtoWeaveApp._on_diarize_lesson(app)

        self.assertIn("別の録音処理が進行中です。", showinfo.call_args[0][1])
        app.main_pane.set_diarizing.assert_not_called()

    def test_diarization_started_event_shows_busy_state(self) -> None:
        app = _fake_app()

        OtoWeaveApp._handle_controller_event(app, "diarization_started", None)

        app.main_pane.set_diarizing.assert_called_once_with(True)

    def test_diarization_finished_releases_busy_and_refreshes_active_note(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 5.0, "おはよう", speaker="話者1"),
        ]
        app._active_folder = folder

        OtoWeaveApp._handle_controller_event(
            app, "diarization_finished", (folder, lesson)
        )

        app.main_pane.set_diarizing.assert_called_once_with(False)
        app._apply_note.assert_called_once()
        app._load_lessons.assert_called_once_with()

    def test_diarization_finished_for_inactive_note_still_releases_busy(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        app._active_folder = Path("/tmp/another")

        OtoWeaveApp._handle_controller_event(
            app, "diarization_finished", (folder, lesson)
        )

        app.main_pane.set_diarizing.assert_called_once_with(False)
        app._apply_note.assert_not_called()
        app._load_lessons.assert_called_once_with()


class OtoWeaveAppTranscriptEditingTests(unittest.TestCase):
    def test_edited_transcript_is_saved_to_selected_note(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/editing")
        app._active_folder = folder
        app.controller = SimpleNamespace(
            busy=False,
            select_lesson=Mock(),
            replace_transcript_text=Mock(),
            reset_chat=Mock(),
        )

        saved = OtoWeaveApp._save_transcript_text(
            app,
            "訂正した最初の段落。\n\n訂正した次の段落。",
        )

        self.assertTrue(saved)
        app.controller.select_lesson.assert_called_once_with(folder)
        app.controller.replace_transcript_text.assert_called_once()
        app._load_lessons.assert_called_once_with()

    def test_edit_is_blocked_during_background_processing(self) -> None:
        app = _fake_app()
        app._active_folder = Path("/tmp/editing")
        app.controller = SimpleNamespace(
            busy=True,
            select_lesson=Mock(),
            replace_transcript_text=Mock(),
            reset_chat=Mock(),
        )

        with patch("otoweave_app.otoweave_app.messagebox.showinfo"):
            saved = OtoWeaveApp._save_transcript_text(app, "訂正文")

        self.assertFalse(saved)
        app.controller.replace_transcript_text.assert_not_called()


class OtoWeaveAppNoteManagementTests(unittest.TestCase):
    def test_rename_uses_selected_folder_and_new_title(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson(title="古い名前")
        app._active_folder = folder
        app.controller = SimpleNamespace(
            busy=False,
            select_lesson=Mock(return_value=lesson),
            rename_current_lesson=Mock(),
        )
        dialog = SimpleNamespace(get_input=Mock(return_value="新しい名前"))

        with patch("otoweave_app.otoweave_app.ctk.CTkInputDialog", return_value=dialog):
            OtoWeaveApp._on_rename_note(app)

        app.controller.select_lesson.assert_called_once_with(folder)
        app.controller.rename_current_lesson.assert_called_once_with(
            "新しい名前"
        )

    def test_delete_requires_confirmation_and_keeps_source_file(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson(title="削除するノート")
        lesson.source_audio_name = "original.ogg"
        app._active_folder = folder
        app.controller = SimpleNamespace(
            busy=False,
            select_lesson=Mock(return_value=lesson),
            delete_current_lesson=Mock(),
        )

        with patch(
            "otoweave_app.otoweave_app.messagebox.askyesno",
            return_value=True,
        ) as confirm:
            OtoWeaveApp._on_delete_note(app)

        app.controller.delete_current_lesson.assert_called_once()
        message = confirm.call_args[0][1]
        self.assertIn("original.ogg", message)
        self.assertIn("取り込み元の音声ファイルは削除されません", message)

    def test_rename_speakers_applies_mapping_and_saves_each_matched_segment(
        self,
    ) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 5.0, "おはよう", speaker="話者1"),
            TranscriptSegment("seg_0002", 5.0, 10.0, "どうも", speaker="話者2"),
            TranscriptSegment("seg_0003", 10.0, 15.0, "またね", speaker="話者1"),
        ]
        app._active_folder = folder
        app.controller = SimpleNamespace(
            busy=False,
            select_lesson=Mock(return_value=lesson),
            update_segment_speaker=Mock(),
        )
        dialog = SimpleNamespace(result={"話者1": "先生"})

        with patch(
            "otoweave_app.otoweave_app._SpeakerRenameDialog",
            return_value=dialog,
        ) as dialog_cls:
            OtoWeaveApp._on_rename_speakers(app)

        # The dialog was offered both distinct speakers, in order seen.
        self.assertEqual(dialog_cls.call_args[0][1], ["話者1", "話者2"])
        # Every segment previously spoken by 話者1 is renamed in memory...
        self.assertEqual(lesson.segments[0].speaker, "先生")
        self.assertEqual(lesson.segments[2].speaker, "先生")
        self.assertEqual(lesson.segments[1].speaker, "話者2")
        # ...and persisted through the existing per-segment save path.
        self.assertEqual(
            app.controller.update_segment_speaker.call_args_list,
            [
                call("seg_0001", "先生"),
                call("seg_0003", "先生"),
            ],
        )
        app._apply_note.assert_called_once()
        app.main_pane.update_status.assert_called_once()

    def test_rename_speakers_cancelled_dialog_changes_nothing(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 5.0, "おはよう", speaker="話者1"),
        ]
        app._active_folder = folder
        app.controller = SimpleNamespace(
            busy=False,
            select_lesson=Mock(return_value=lesson),
            update_segment_speaker=Mock(),
        )
        dialog = SimpleNamespace(result=None)

        with patch(
            "otoweave_app.otoweave_app._SpeakerRenameDialog",
            return_value=dialog,
        ):
            OtoWeaveApp._on_rename_speakers(app)

        app.controller.update_segment_speaker.assert_not_called()
        app._apply_note.assert_not_called()

    def test_rename_speakers_blocked_while_controller_busy(self) -> None:
        app = _fake_app()
        app._active_folder = Path("/tmp/lesson")
        app.controller = SimpleNamespace(busy=True, select_lesson=Mock())

        with patch("otoweave_app.otoweave_app.messagebox.showinfo"):
            OtoWeaveApp._on_rename_speakers(app)

        app.controller.select_lesson.assert_not_called()

    def test_rename_speakers_without_speaker_data_shows_info_and_stops(self) -> None:
        app = _fake_app()
        folder = Path("/tmp/lesson")
        lesson = _make_lesson()
        lesson.segments = [
            TranscriptSegment("seg_0001", 0.0, 5.0, "普通の文"),
        ]
        app._active_folder = folder
        app.controller = SimpleNamespace(busy=False, select_lesson=Mock(return_value=lesson))

        with patch(
            "otoweave_app.otoweave_app._SpeakerRenameDialog",
        ) as dialog_cls, patch(
            "otoweave_app.otoweave_app.messagebox.showinfo",
        ):
            OtoWeaveApp._on_rename_speakers(app)

        dialog_cls.assert_not_called()


# ---------------------------------------------------------------------------
# 4. _apply_note() — stale-folder summary race fix
# ---------------------------------------------------------------------------

class OtoWeaveAppSummaryRaceTests(unittest.TestCase):
    def _note_for_folder(self, folder: Path) -> dict:
        return {
            "id": "test_lesson",
            "label": "Test",
            "title": "Test lesson",
            "meta": "2026年1月1日",
            "keywords": "",
            "transcript": "transcript text",
            "summary": "（読み込み中…）",
            "_folder": str(folder),
        }

    def test_summary_for_current_folder_is_applied(self) -> None:
        app = _fake_app()
        app.controller = SimpleNamespace(reset_chat=Mock(), player=SimpleNamespace(stop=Mock()))
        # Use the real _load_summary so the background callback is captured.
        app._load_summary = OtoWeaveApp._load_summary.__get__(app)
        folder = Path("/tmp/lesson_a")

        OtoWeaveApp._apply_note(app, self._note_for_folder(folder), update_route=False)

        # _active_folder must match folder_a
        self.assertEqual(app._active_folder, folder)
        on_success = app.run_background.call_args[0][1]
        on_success(
            (
                "summary text",
                folder,
                "generated",
                {"lesson_record": "generated"},
                "lesson_record",
            )
        )

        app.right_pane.set_summary.assert_called_once_with(
            "summary text",
            "generated",
        )

    def test_summary_for_stale_folder_is_not_applied(self) -> None:
        app = _fake_app()
        app.controller = SimpleNamespace(reset_chat=Mock(), player=SimpleNamespace(stop=Mock()))
        # Use the real _load_summary so the background callback is captured.
        app._load_summary = OtoWeaveApp._load_summary.__get__(app)
        folder_a = Path("/tmp/lesson_a")
        folder_b = Path("/tmp/lesson_b")

        OtoWeaveApp._apply_note(app, self._note_for_folder(folder_a), update_route=False)

        # Navigate to a different lesson before the background job finishes
        app._active_folder = folder_b

        on_success = app.run_background.call_args[0][1]
        on_success(
            (
                "stale summary",
                folder_a,
                "stale",
                {"lesson_record": "stale"},
                "lesson_record",
            )
        )

        app.right_pane.set_summary.assert_not_called()

    def test_folder_change_resets_chat_history(self) -> None:
        app = _fake_app()
        app.controller = SimpleNamespace(reset_chat=Mock(), player=SimpleNamespace(stop=Mock()))
        folder_a = Path("/tmp/lesson_a")
        folder_b = Path("/tmp/lesson_b")

        app._active_folder = folder_a
        OtoWeaveApp._apply_note(app, self._note_for_folder(folder_b), update_route=False)

        app.controller.reset_chat.assert_called_once()
        self.assertEqual(app._active_folder, folder_b)

    def test_same_folder_does_not_reset_chat_history(self) -> None:
        app = _fake_app()
        app.controller = SimpleNamespace(reset_chat=Mock(), player=SimpleNamespace(stop=Mock()))
        folder = Path("/tmp/lesson_a")

        app._active_folder = folder
        OtoWeaveApp._apply_note(app, self._note_for_folder(folder), update_route=False)

        app.controller.reset_chat.assert_not_called()

    def test_folder_change_clears_chat_ui(self) -> None:
        """Fix 1A: switching lesson must wipe the chat widget."""
        app = _fake_app()
        app.controller = SimpleNamespace(reset_chat=Mock(), player=SimpleNamespace(stop=Mock()))
        folder_a = Path("/tmp/lesson_a")
        folder_b = Path("/tmp/lesson_b")

        app._active_folder = folder_a
        OtoWeaveApp._apply_note(app, self._note_for_folder(folder_b), update_route=False)

        app.right_pane.clear_chat.assert_called_once()

    def test_same_folder_does_not_clear_chat_ui(self) -> None:
        app = _fake_app()
        app.controller = SimpleNamespace(reset_chat=Mock(), player=SimpleNamespace(stop=Mock()))
        folder = Path("/tmp/lesson_a")

        app._active_folder = folder
        OtoWeaveApp._apply_note(app, self._note_for_folder(folder), update_route=False)

        app.right_pane.clear_chat.assert_not_called()


# ---------------------------------------------------------------------------
# 5. DetailPane._request_search() — real data vs. mock fallback
# ---------------------------------------------------------------------------

class SearchDataSourceTests(unittest.TestCase):
    def _fake_detail_pane(self, search_notes: list[dict] | None, query: str) -> SimpleNamespace:
        run_background = Mock()
        return SimpleNamespace(
            _search_notes=search_notes,
            search_entry=SimpleNamespace(get=lambda: query),
            run_background=run_background,
            _render_search_results=Mock(),
        )

    def test_search_uses_real_notes_when_populated(self) -> None:
        real_notes = [{"title": "フランス革命", "keywords": "身分制", "transcript": "身分ごとの権利", "id": "r1"}]
        dp = self._fake_detail_pane(real_notes, "身分")

        from otoweave_app.customtkinter_views import DetailPane
        DetailPane._request_search(dp)

        worker_fn = dp.run_background.call_args[0][0]
        results = worker_fn()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "r1")

    def test_search_falls_back_to_mock_data_when_not_yet_loaded(self) -> None:
        dp = self._fake_detail_pane(None, "こころ")

        from otoweave_app.customtkinter_views import DetailPane
        from otoweave_app.customtkinter_mock_data import NOTE_BY_ID
        DetailPane._request_search(dp)

        worker_fn = dp.run_background.call_args[0][0]
        results = worker_fn()
        # "こころ" appears in the mock "kokoro" lesson title
        self.assertTrue(any("こころ" in n["title"].lower() or "こころ" in n["title"] for n in results))

    def test_empty_query_returns_all_real_notes(self) -> None:
        real_notes = [
            {"title": "数学", "keywords": "微分", "transcript": "接線", "id": "math"},
            {"title": "英語", "keywords": "関係代名詞", "transcript": "who which", "id": "eng"},
        ]
        dp = self._fake_detail_pane(real_notes, "")

        from otoweave_app.customtkinter_views import DetailPane
        DetailPane._request_search(dp)

        worker_fn = dp.run_background.call_args[0][0]
        results = worker_fn()
        self.assertEqual(len(results), 2)

    def test_query_filters_by_title_and_transcript(self) -> None:
        real_notes = [
            {"title": "数学", "keywords": "微分", "transcript": "接線の傾き", "id": "math"},
            {"title": "英語", "keywords": "関係代名詞", "transcript": "who which", "id": "eng"},
        ]
        dp = self._fake_detail_pane(real_notes, "接線")

        from otoweave_app.customtkinter_views import DetailPane
        DetailPane._request_search(dp)

        worker_fn = dp.run_background.call_args[0][0]
        results = worker_fn()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "math")

    def test_query_filters_by_source_audio_filename(self) -> None:
        real_notes = [
            {
                "title": "録音",
                "keywords": "",
                "transcript": "",
                "source_audio_name": "消防本部見学.ogg",
                "id": "fire",
            },
            {
                "title": "録音",
                "keywords": "",
                "transcript": "",
                "source_audio_name": "英語授業.wav",
                "id": "english",
            },
        ]
        dp = self._fake_detail_pane(real_notes, "消防")

        from otoweave_app.customtkinter_views import DetailPane
        DetailPane._request_search(dp)

        results = dp.run_background.call_args[0][0]()
        self.assertEqual([note["id"] for note in results], ["fire"])

    def test_notes_pane_search_filters_without_a_separate_activity(self) -> None:
        render = Mock()
        dp = SimpleNamespace(
            notes_search_entry=SimpleNamespace(get=lambda: "消防"),
            _note_groups={
                "2026年": [
                    {
                        "title": "見学",
                        "source_audio_name": "消防本部.ogg",
                        "keywords": "",
                        "transcript": "",
                    },
                    {
                        "title": "英語",
                        "source_audio_name": "lesson.wav",
                        "keywords": "",
                        "transcript": "",
                    },
                ]
            },
            _render_note_groups=render,
            _request_display_change=Mock(),
            request_content_search=None,
        )
        dp._render_filtered_note_groups = (
            DetailPane._render_filtered_note_groups.__get__(dp)
        )

        DetailPane._filter_notes(dp)

        filtered = render.call_args[0][0]
        self.assertEqual(len(filtered["2026年"]), 1)
        self.assertEqual(
            filtered["2026年"][0]["source_audio_name"],
            "消防本部.ogg",
        )

    def test_notes_pane_search_includes_transcript_body(self) -> None:
        render = Mock()
        dp = SimpleNamespace(
            notes_search_entry=SimpleNamespace(get=lambda: "光合成"),
            _note_groups={
                "2026年": [
                    {
                        "title": "理科",
                        "source_audio_name": "lesson.ogg",
                        "keywords": "",
                        "transcript": "植物は光合成によって養分を作ります。",
                    }
                ]
            },
            _render_note_groups=render,
            _request_display_change=Mock(),
            request_content_search=None,
        )
        dp._render_filtered_note_groups = (
            DetailPane._render_filtered_note_groups.__get__(dp)
        )

        DetailPane._filter_notes(dp)

        self.assertEqual(
            render.call_args[0][0]["2026年"][0]["title"],
            "理科",
        )

    def test_note_month_group_can_be_collapsed_and_expanded(self) -> None:
        pane = SimpleNamespace(
            _collapsed_note_groups=set(),
            _visible_note_groups={"2026年 6月": [{"id": "lesson-1"}]},
            _render_note_groups=Mock(),
            active_note_id="lesson-1",
            set_active_note=Mock(),
            _request_display_change=Mock(),
        )

        DetailPane._toggle_note_group(pane, "2026年 6月")

        self.assertEqual(pane._collapsed_note_groups, {"2026年 6月"})
        pane._render_note_groups.assert_called_once_with(pane._visible_note_groups)
        pane.set_active_note.assert_called_once_with("lesson-1")

        DetailPane._toggle_note_group(pane, "2026年 6月")

        self.assertEqual(pane._collapsed_note_groups, set())
        self.assertEqual(pane._render_note_groups.call_count, 2)

    def test_search_query_is_recorded_so_collapsed_groups_can_auto_expand(self) -> None:
        render = Mock()
        pane = SimpleNamespace(
            notes_search_entry=SimpleNamespace(get=lambda: "光合成"),
            _note_groups={
                "2026年 6月": [
                    {
                        "title": "理科",
                        "source_audio_name": "",
                        "keywords": "",
                        "transcript": "植物は光合成をします。",
                    }
                ]
            },
            _render_note_groups=render,
            _request_display_change=Mock(),
            request_content_search=None,
        )
        pane._render_filtered_note_groups = (
            DetailPane._render_filtered_note_groups.__get__(pane)
        )

        DetailPane._filter_notes(pane)

        self.assertEqual(pane._active_note_query, "光合成")
        self.assertIn("2026年 6月", render.call_args[0][0])


class SearchJumpTests(unittest.TestCase):
    @staticmethod
    def _pane_for_search(raw_text: Mock) -> SimpleNamespace:
        widget = lambda: SimpleNamespace(
            grid=Mock(),
            grid_remove=Mock(),
            configure=Mock(),
        )
        pane = SimpleNamespace(
            textbox=SimpleNamespace(_textbox=raw_text),
            status_label=SimpleNamespace(configure=Mock()),
            search_position_label=widget(),
            search_previous_button=widget(),
            search_next_button=widget(),
            _search_matches=[],
            _search_match_index=-1,
            _search_query="",
        )
        pane._set_search_navigation_visible = (
            lambda visible: MainPane._set_search_navigation_visible(
                pane,
                visible,
            )
        )
        pane._show_search_match = (
            lambda index: MainPane._show_search_match(pane, index)
        )
        return pane

    def test_note_request_forwards_search_query_to_main_view(self) -> None:
        app = _fake_app()
        note = {"id": "science"}
        app._note_map = {"science": note}

        OtoWeaveApp._request_note(app, "science", "光合成")

        app._apply_note.assert_called_once_with(
            note,
            search_query="光合成",
        )

    def test_multiple_body_matches_are_highlighted_and_first_is_shown(self) -> None:
        raw_text = Mock()
        raw_text.search.side_effect = ["2.3", "5.1", ""]
        pane = self._pane_for_search(raw_text)

        MainPane._highlight_search_matches(pane, "光合成")

        self.assertEqual(raw_text.tag_add.call_count, 3)
        raw_text.see.assert_called_once_with("2.3")
        raw_text.mark_set.assert_called_once_with("insert", "2.3")
        status = pane.status_label.configure.call_args.kwargs["text"]
        self.assertIn("2件", status)
        pane.search_position_label.configure.assert_called_with(text="1 / 2")

    def test_title_only_match_does_not_force_text_scroll(self) -> None:
        raw_text = Mock()
        raw_text.search.return_value = ""
        pane = self._pane_for_search(raw_text)

        MainPane._highlight_search_matches(pane, "授業タイトル")

        raw_text.see.assert_not_called()
        status = pane.status_label.configure.call_args.kwargs["text"]
        self.assertIn("ノート名または元ファイル名", status)

    def test_next_and_previous_move_between_matches(self) -> None:
        raw_text = Mock()
        pane = self._pane_for_search(raw_text)
        pane._search_query = "光合成"
        pane._search_matches = [
            ("2.3", "2.6"),
            ("5.1", "5.4"),
            ("8.0", "8.3"),
        ]
        pane._search_match_index = 0

        MainPane._show_next_search_match(pane)
        self.assertEqual(pane._search_match_index, 1)
        raw_text.see.assert_called_with("5.1")

        MainPane._show_previous_search_match(pane)
        self.assertEqual(pane._search_match_index, 0)
        raw_text.see.assert_called_with("2.3")


class SpeakerRenameMappingTests(unittest.TestCase):
    """_build_speaker_rename_mapping(): the pure validation function behind
    _SpeakerRenameDialog._ok(). Blank/unchanged entries are skipped, and a
    new name of "teacher" is rejected because models.TranscriptSegment
    silently maps that exact word back to "" on reload."""

    def test_blank_and_unchanged_entries_are_skipped(self) -> None:
        mapping, error = _build_speaker_rename_mapping(
            {"話者1": "  ", "話者2": "話者2"}
        )
        self.assertEqual(mapping, {})
        self.assertEqual(error, "")

    def test_non_blank_entries_are_trimmed_and_included(self) -> None:
        mapping, error = _build_speaker_rename_mapping(
            {"話者1": "  先生  ", "話者2": ""}
        )
        self.assertEqual(mapping, {"話者1": "先生"})
        self.assertEqual(error, "")

    def test_teacher_is_rejected_case_insensitively(self) -> None:
        for candidate in ("teacher", "Teacher", "TEACHER"):
            mapping, error = _build_speaker_rename_mapping({"話者1": candidate})
            self.assertEqual(mapping, {}, candidate)
            self.assertNotEqual(error, "", candidate)

    def test_rejection_stops_before_applying_earlier_valid_entries(self) -> None:
        # dict preserves insertion order; the second (invalid) entry
        # should still block the whole mapping, not just itself.
        mapping, error = _build_speaker_rename_mapping(
            {"話者1": "先生", "話者2": "teacher"}
        )
        self.assertEqual(mapping, {})
        self.assertNotEqual(error, "")


class SpeakerColorTaggingTests(unittest.TestCase):
    """MainPane._tag_speaker_colors: assistive color + bold on the
    "Speaker:" prefix, driven by note["speaker_lines"] rather than by
    re-parsing the rendered text."""

    @staticmethod
    def _pane_with_lines(lines: list[str], speaker_lines: list[str]) -> SimpleNamespace:
        raw = Mock()
        raw.index.return_value = f"{len(lines)}.0"

        def get(start: str, _end: str) -> str:
            lineno = int(start.split(".")[0])
            return lines[lineno - 1] if 1 <= lineno <= len(lines) else ""

        raw.get.side_effect = get
        fake_font = SimpleNamespace(
            cget=lambda key: {"family": "Meiryo", "size": 12}[key]
        )
        return SimpleNamespace(
            textbox=SimpleNamespace(_textbox=raw, cget=lambda _key: fake_font),
            _speaker_lines=speaker_lines,
        )

    def test_each_distinct_speaker_prefix_is_tagged_with_its_own_color(self) -> None:
        lines = [
            "00:00  話者1: おはよう",
            "",
            "00:05  話者2: どうも",
            "",
            "00:10  話者1: またね",
        ]
        pane = self._pane_with_lines(lines, ["話者1", "話者2", "話者1"])

        MainPane._tag_speaker_colors(pane)

        raw = pane.textbox._textbox
        self.assertEqual(
            raw.tag_add.call_args_list,
            [
                call("speaker_color_0", "1.7", "1.11"),
                call("speaker_color_1", "3.7", "3.11"),
                call("speaker_color_0", "5.7", "5.11"),
            ],
        )
        # Every palette tag gets configured (not just used ones), each with
        # its own color and a bold variant of the current textbox font.
        configured = {c.args[0]: c.kwargs for c in raw.tag_configure.call_args_list}
        self.assertEqual(
            configured["speaker_color_0"]["foreground"],
            theme_color(COLORS[SPEAKER_COLOR_KEYS[0]]),
        )
        self.assertEqual(configured["speaker_color_0"]["font"], ("Meiryo", 12, "bold"))
        self.assertEqual(
            configured["speaker_color_1"]["foreground"],
            theme_color(COLORS[SPEAKER_COLOR_KEYS[1]]),
        )

    def test_speaker_palette_wraps_after_four_distinct_speakers(self) -> None:
        lines = [f"00:0{i}  話者{i}: 発話{i}" for i in range(5)]
        pane = self._pane_with_lines(
            lines,
            [f"話者{i}" for i in range(5)],
        )

        MainPane._tag_speaker_colors(pane)

        raw = pane.textbox._textbox
        tags_used = [c.args[0] for c in raw.tag_add.call_args_list]
        # 5th distinct speaker (index 4) wraps back to the first color tag.
        self.assertEqual(tags_used[0], tags_used[4])

    def test_no_speakers_removes_tags_and_adds_none(self) -> None:
        lines = ["00:00  ただの文章"]
        pane = self._pane_with_lines(lines, [""])

        MainPane._tag_speaker_colors(pane)

        raw = pane.textbox._textbox
        self.assertEqual(raw.tag_remove.call_count, len(SPEAKER_COLOR_KEYS))
        raw.tag_add.assert_not_called()
        raw.tag_configure.assert_not_called()


class DisplayPreferenceMappingTests(unittest.TestCase):
    def test_supported_text_sizes_map_to_persisted_names(self) -> None:
        self.assertEqual(OtoWeaveApp._points_text_size(12), "Small")
        self.assertEqual(OtoWeaveApp._points_text_size(18), "Extra Large")
        self.assertEqual(OtoWeaveApp._text_size_points("Large"), 16)


class ColumnResizeTests(unittest.TestCase):
    def test_detail_width_tracks_pointer_but_respects_limits(self) -> None:
        self.assertEqual(
            OtoWeaveApp._resized_width(280, 70, 1, 220, 480),
            350,
        )
        self.assertEqual(
            OtoWeaveApp._resized_width(280, -200, 1, 220, 480),
            220,
        )

    def test_right_width_moves_in_the_opposite_direction(self) -> None:
        self.assertEqual(
            OtoWeaveApp._resized_width(310, -60, -1, 260, 520),
            370,
        )


# ---------------------------------------------------------------------------
# 6. _on_close() — deferred destroy while busy
# ---------------------------------------------------------------------------

class OtoWeaveAppCloseTests(unittest.TestCase):
    def _fake_app_with_controller(self, busy: bool) -> SimpleNamespace:
        app = _fake_app()
        app.controller = SimpleNamespace(
            busy=busy,
            close=Mock(),
            reset_chat=Mock(),
        )
        app.destroy = Mock()
        app._wait_and_close = Mock()
        return app

    def test_close_destroys_immediately_when_not_busy(self) -> None:
        app = self._fake_app_with_controller(busy=False)
        OtoWeaveApp._on_close(app)
        app.controller.close.assert_called_once()
        app.destroy.assert_called_once()
        app._wait_and_close.assert_not_called()

    def test_close_defers_destroy_when_busy(self) -> None:
        app = self._fake_app_with_controller(busy=True)
        OtoWeaveApp._on_close(app)
        app.controller.close.assert_called_once()
        app.destroy.assert_not_called()
        app.after.assert_called()

    def test_wait_and_close_destroys_when_no_longer_busy(self) -> None:
        app = _fake_app()
        app.controller = SimpleNamespace(busy=False)
        app.destroy = Mock()
        OtoWeaveApp._wait_and_close(app)
        app.destroy.assert_called_once()

    def test_wait_and_close_reschedules_when_still_busy(self) -> None:
        app = _fake_app()
        app.controller = SimpleNamespace(busy=True)
        app.destroy = Mock()
        # _wait_and_close references self._wait_and_close as the after-callback
        app._wait_and_close = Mock()
        OtoWeaveApp._wait_and_close(app)
        app.destroy.assert_not_called()
        app.after.assert_called()


class OtoWeaveAppDropTests(unittest.TestCase):
    def _drop_app(self) -> SimpleNamespace:
        return SimpleNamespace(
            controller=SimpleNamespace(busy=False),
            _begin_audio_import=Mock(),
        )

    def test_single_audio_file_starts_shared_import_flow(self) -> None:
        app = self._drop_app()
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "lesson audio.ogg"
            audio.write_bytes(b"OggS")

            OtoWeaveApp._accept_dropped_paths(app, [audio])

        app._begin_audio_import.assert_called_once_with(audio)

    def test_directory_drop_is_rejected(self) -> None:
        app = self._drop_app()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "フォルダー"):
                OtoWeaveApp._accept_dropped_paths(app, [Path(temporary)])

    def test_unsupported_file_drop_is_rejected(self) -> None:
        app = self._drop_app()
        with tempfile.TemporaryDirectory() as temporary:
            text = Path(temporary) / "notes.txt"
            text.write_text("not audio", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "対応していない"):
                OtoWeaveApp._accept_dropped_paths(app, [text])

    def test_multiple_files_are_rejected(self) -> None:
        app = self._drop_app()
        with self.assertRaisesRegex(ValueError, "1つだけ"):
            OtoWeaveApp._accept_dropped_paths(
                app,
                [Path("one.ogg"), Path("two.wav")],
            )

    def test_file_dialog_selection_defers_import_until_focus_returns(self) -> None:
        app = SimpleNamespace(
            _file_dialog_active=True,
            main_pane=SimpleNamespace(update_status=Mock()),
            after=Mock(),
            route_to=Mock(),
            _start_selected_audio_import=Mock(),
        )
        selected = Path("C:/recordings/lesson audio.ogg")

        OtoWeaveApp._finish_audio_file_dialog(
            app,
            ("selected", str(selected)),
        )

        self.assertFalse(app._file_dialog_active)
        delay, callback = app.after.call_args[0]
        self.assertEqual(delay, 150)
        callback()
        app._start_selected_audio_import.assert_called_once_with(selected)


if __name__ == "__main__":
    unittest.main()
