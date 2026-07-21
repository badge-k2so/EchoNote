from __future__ import annotations

import enum
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .app_logging import log_exception
from .asr import (
    LiveAsrWorker,
    RecognizedSentence,
    AsrThreadConfig,
    select_asr_threads,
)
from .audio import (
    AudioPlayer,
    AudioRecorder,
    AudioProcessingOptions,
    AudioSource,
    SpeechChunk,
    convert_pcm_to_opus,
)
from .demo_content import DemoContent
from .llm_session import LlmSession
from .models import LessonRecord, TranscriptSegment, append_readable_segment
from .segment_editing import (
    mark_time,
    next_segment_id,
    remap_edited_blocks,
)
from .storage import LessonStore
from .transcription_service import (
    TranscriptionService,
    recording_datetime,
    set_suggested_title,
)
from .user_dictionary import (
    correct_text,
    dictionary_path,
    load_dictionary,
)


class SessionState(enum.Enum):
    """One audio job at a time: the controller is IDLE or in exactly one
    of these states. LLM work is tracked separately by LlmSession."""

    IDLE = "idle"
    RECORDING = "recording"
    STOPPING = "stopping"
    IMPORTING = "importing"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"


class LearningAccessController:
    """Session state machine and event hub.

    Heavy work is delegated to collaborators:
    - TranscriptionService: import / re-transcription pipelines
    - LlmSession: summary and chat model lifecycle
    - DemoContent: demo lesson creation and repair
    Pure transcript editing helpers live in segment_editing.
    """

    # Recording watchdog: how often it checks the recorder, and how long
    # without incoming audio frames counts as "the mic is not delivering".
    WATCHDOG_INTERVAL_SECONDS = 2.0
    RECORDER_STALL_SECONDS = 10.0
    # Live light saves are throttled to one every few seconds: every save
    # rewrites transcript.json + metadata + marks with fsync, so a save per
    # confirmed sentence becomes O(n^2) disk work over a 90-minute lesson.
    # A crash loses at most a few seconds of recognized text; the raw audio
    # is written to disk continuously by the recorder.
    LIVE_SAVE_INTERVAL_SECONDS = 3.0

    def __init__(self, project_root: Path, data_root: Path) -> None:
        self.project_root = Path(project_root)
        self.store = LessonStore(data_root)
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.current_folder: Path | None = None
        self.current_lesson: LessonRecord | None = None
        self.recorder: AudioRecorder | None = None
        self.asr: LiveAsrWorker | None = None
        self._lock = threading.RLock()
        # Guards only the RECORDING->STOPPING hand-off. Kept separate from
        # _lock so callers (Tk thread, recording watchdog) never contend
        # with longer _lock critical sections.
        self._stop_lock = threading.Lock()
        self._state = SessionState.IDLE
        # Folder currently written by a background import/transcription,
        # used to block conflicting edits on that lesson only.
        self._processing_folder: Path | None = None
        self._recording_speaker_label = ""
        self._watchdog_stop: threading.Event | None = None
        self._last_live_save_monotonic = 0.0
        self._live_save_dirty = False
        self._dictionary_entries = load_dictionary(
            dictionary_path(self.store.root)
        )
        self.ffmpeg = self.project_root / "engines" / "ffmpeg" / "ffmpeg.exe"
        self.player = AudioPlayer(
            on_position=lambda value: self.events.put(("playback_position", value)),
            on_finished=lambda: self.events.put(("playback_finished", None)),
            on_error=lambda message: self.events.put(("error", message)),
            ffmpeg=self.ffmpeg,
        )
        self.llm_session = LlmSession(self.project_root, self.events)
        self.transcription = TranscriptionService(
            self.project_root,
            self.store,
            self.events,
            self.ffmpeg,
            correct_text=self._correct_recognized_text,
            on_lesson_ready=self._set_current_lesson,
        )
        self.demo = DemoContent(self.store)
        self.demo.repair_demo_audio()
        try:
            self.store.purge_trash()
        except OSError:
            pass

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def recording(self) -> bool:
        return self.recorder is not None

    @property
    def busy(self) -> bool:
        return (
            self.recorder is not None
            or self._state is not SessionState.IDLE
            or self.llm_session.busy
        )

    def reload_dictionary(self) -> None:
        self._dictionary_entries = load_dictionary(
            dictionary_path(self.store.root)
        )

    def _correct_recognized_text(self, text: str) -> str:
        return correct_text(text, self._dictionary_entries)

    def _set_current_lesson(self, folder: Path, lesson: LessonRecord) -> None:
        self.current_folder = folder
        self.current_lesson = lesson

    # ------------------------------------------------------------------
    # Live recording
    # ------------------------------------------------------------------

    def start_lesson(
        self,
        mode: str,
        source: AudioSource,
        processing: AudioProcessingOptions | None = None,
        speaker_label: str = "",
    ) -> None:
        if self.busy:
            if self.llm_session.busy:
                raise RuntimeError("LLMが処理中です。完了後に録音を開始してください。")
            raise RuntimeError("A lesson is already being recorded")
        self.release_chat_model()
        self.player.stop()
        lesson = LessonRecord.create(mode, source.kind)
        folder = self.store.create_lesson(lesson)
        self.current_lesson = lesson
        self.current_folder = folder
        self._state = SessionState.RECORDING
        self._recording_speaker_label = " ".join(speaker_label.split()).strip()

        if mode in {"japanese", "english"}:
            thread_config = select_asr_threads("live")
            self._record_asr_thread_config(lesson, folder, thread_config)
            self.asr = LiveAsrWorker(
                mode=mode,
                project_root=self.project_root,
                on_sentences=self._on_sentences,
                on_status=lambda value: self.events.put(("status", value)),
                on_error=lambda value: self.events.put(("error", value)),
                num_threads=thread_config.num_threads,
            )
            self.asr.start()
        else:
            self.asr = None

        self.recorder = AudioRecorder(
            source=source,
            output_pcm=folder / "recording.pcm",
            on_speech_chunk=self._on_speech_chunk,
            on_error=lambda value: self.events.put(("error", value)),
            processing=processing,
        )
        try:
            self.recorder.start()
        except Exception:
            if self.asr is not None:
                self.asr.stop(timeout=5)
            self.asr = None
            self.recorder = None
            self._state = SessionState.IDLE
            lesson.status = "recording_failed"
            self.store.save(folder, lesson)
            raise
        self._last_live_save_monotonic = 0.0
        self._live_save_dirty = False
        self._start_recording_watchdog(self.recorder)
        self.events.put(("lesson_started", (folder, lesson)))
        self.events.put(("status", "録音中" if mode == "record_only" else "録音中・モデル準備中"))

    def toggle_recording_pause(self) -> bool:
        if self.recorder is None:
            return False
        paused = self.recorder.toggle_pause()
        self.events.put(("status", "一時停止中" if paused else "録音中"))
        return paused

    def stop_lesson_async(self) -> None:
        # Called from the Tk thread and from the recording watchdog; the
        # atomic hand-off ensures only one lesson-finalizer thread spawns
        # even when both request a stop at the same moment.
        with self._stop_lock:
            if self.recorder is None or self._state is SessionState.STOPPING:
                return
            self._state = SessionState.STOPPING
        watchdog_stop = self._watchdog_stop
        if watchdog_stop is not None:
            watchdog_stop.set()
        self.events.put(("status", "録音を保存中"))
        threading.Thread(target=self._finish_lesson, name="lesson-finalizer", daemon=True).start()

    # ------------------------------------------------------------------
    # Recording watchdog
    # ------------------------------------------------------------------

    def _start_recording_watchdog(self, recorder: AudioRecorder) -> None:
        self._watchdog_stop = threading.Event()
        threading.Thread(
            target=self._recording_watchdog,
            args=(recorder, self._watchdog_stop),
            name="recording-watchdog",
            daemon=True,
        ).start()

    def _recording_watchdog(self, recorder: AudioRecorder, stop: threading.Event) -> None:
        """Detect a silently failed recording (disk full, mic gone).

        Runs only while this recorder is the active RECORDING session.
        IMPORTANT (past deadlock C-2): never holds _lock and never joins
        threads here; a detected failure is routed through the existing
        asynchronous stop path (stop_lesson_async), which finalizes and
        saves everything recorded so far on its own thread."""
        warned_stalled = False
        while not stop.wait(self.WATCHDOG_INTERVAL_SECONDS):
            if self.recorder is not recorder or self._state is not SessionState.RECORDING:
                return
            if recorder.failed:
                self.events.put(
                    (
                        "error",
                        "録音を保存できなくなったため停止しました。"
                        "パソコンの空き容量を確認してください。"
                        "ここまでの録音は保存されています。",
                    )
                )
                self.stop_lesson_async()
                return
            if recorder.paused:
                continue
            stalled = recorder.seconds_since_last_data >= self.RECORDER_STALL_SECONDS
            if stalled and not warned_stalled:
                warned_stalled = True
                self.events.put(
                    ("error", "マイクの音が届いていません。マイクの接続を確認してください。")
                )
            elif not stalled:
                # 回復したら、次にまた途絶えたとき再度警告できるようにする。
                warned_stalled = False

    def _record_asr_thread_config(
        self,
        lesson: LessonRecord,
        folder: Path,
        config: AsrThreadConfig,
    ) -> None:
        lesson.asr_processing_mode = config.processing_mode
        lesson.asr_threads = config.num_threads
        lesson.detected_logical_cpus = config.detected_logical_cpus
        self.store.save(folder, lesson)

    def _on_speech_chunk(self, chunk: SpeechChunk) -> None:
        if self.asr is not None:
            self.asr.submit(chunk)

    def _on_sentences(self, sentences: list[RecognizedSentence]) -> None:
        with self._lock:
            if self.current_lesson is None:
                return
            for sentence in sentences:
                append_readable_segment(
                    self.current_lesson.segments,
                    TranscriptSegment(
                        id=next_segment_id(self.current_lesson.segments),
                        start=sentence.start,
                        end=sentence.end,
                        text=self._correct_recognized_text(sentence.text),
                        speaker=self._recording_speaker_label,
                    ),
                )
            # Confirmed live sentences arrive continuously; throttle their
            # saves. User actions (marks, edits) still save immediately.
            self._save_current(throttle=True)
            self.events.put(("segments_changed", self.current_lesson))

    def _finish_lesson(self) -> None:
        try:
            recorder = self.recorder
            asr = self.asr
            if recorder is not None:
                recorder.stop()
            if asr is not None:
                self.events.put(("status", "残りの文字起こしを処理中"))
                asr.stop()
            with self._lock:
                lesson = self.current_lesson
                folder = self.current_folder
            if lesson is None or folder is None:
                return
            pcm_path = folder / "recording.pcm"
            opus_path = folder / "audio.opus"
            # ffmpeg can take tens of seconds after a 90-minute lesson;
            # run it outside _lock so UI operations that need the lock
            # (selecting notes, toggling marks) stay responsive.
            converted = False
            if pcm_path.exists() and pcm_path.stat().st_size > 0:
                self.events.put(("status", "音声を圧縮中"))
                convert_pcm_to_opus(self.ffmpeg, pcm_path, opus_path)
                converted = True
            with self._lock:
                if converted:
                    lesson.audio_file = opus_path.name
                    pcm_path.unlink(missing_ok=True)
                lesson.status = "complete"
                set_suggested_title(lesson)
                # Unconditional flush: any sentences skipped by the live
                # save throttling are persisted here.
                self.store.save(folder, lesson)
                self._live_save_dirty = False
            self.events.put(("lesson_finished", (folder, lesson)))
            self.events.put(("status", "保存しました"))
        except Exception as exc:
            log_exception("録音の保存処理に失敗", exc)
            with self._lock:
                if self.current_lesson is not None:
                    self.current_lesson.status = "needs_attention"
                    self._save_current()
            self.events.put(
                (
                    "error",
                    "録音の保存を完了できませんでした。"
                    "アプリを開き直すと修復を試みます。",
                )
            )
        finally:
            self.recorder = None
            self.asr = None
            self._state = SessionState.IDLE

    # ------------------------------------------------------------------
    # Import and re-transcription (delegated to TranscriptionService)
    # ------------------------------------------------------------------

    def import_audio_async(self, source_path: Path, mode: str) -> None:
        source_path = Path(source_path).resolve()
        if self.busy:
            raise RuntimeError("別の録音処理が進行中です。")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if mode not in {"japanese", "english", "record_only"}:
            raise ValueError(f"Unsupported import mode: {mode}")

        self.release_chat_model()
        self.player.stop()
        lesson_time = recording_datetime(source_path)
        lesson = LessonRecord.create(mode, "imported", now=lesson_time)
        lesson.source_audio_name = source_path.name
        lesson.imported_at = datetime.now().astimezone().isoformat(timespec="seconds")
        lesson.status = "importing"
        folder = self.store.create_lesson(lesson)
        self.current_folder = folder
        self.current_lesson = lesson
        self._state = SessionState.IMPORTING
        self._processing_folder = folder
        self.events.put(("import_started", (folder, lesson)))
        threading.Thread(
            target=self._run_import,
            args=(source_path, mode, folder, lesson),
            name="audio-importer",
            daemon=True,
        ).start()

    def _run_import(
        self,
        source_path: Path,
        mode: str,
        folder: Path,
        lesson: LessonRecord,
    ) -> None:
        try:
            self.transcription.import_audio(source_path, mode, folder, lesson)
        finally:
            self._processing_folder = None
            self._state = SessionState.IDLE

    def transcribe_current_audio_async(
        self, mode: str, diarization_speakers: int | None = None
    ) -> None:
        if self.busy:
            raise RuntimeError("別の録音処理が進行中です。")
        if mode not in {"japanese", "mixed", "mixed_qwen17", "english"}:
            raise ValueError(f"Unsupported transcription mode: {mode}")
        if diarization_speakers is not None and int(diarization_speakers) < 1:
            raise ValueError("diarization_speakers must be a positive integer or None")
        folder = self.current_folder
        lesson = self.current_lesson
        audio_path = self.current_audio_path()
        if folder is None or lesson is None or audio_path is None:
            raise RuntimeError("文字起こしできる音声がありません。")

        self.release_chat_model()
        self.player.stop()
        self._state = SessionState.TRANSCRIBING
        self._processing_folder = folder
        self.events.put(("transcription_started", (folder, lesson)))
        threading.Thread(
            target=self._run_transcribe_existing,
            args=(audio_path, mode, folder, lesson, diarization_speakers),
            name="saved-audio-transcriber",
            daemon=True,
        ).start()

    def cancel_transcription(self) -> bool:
        """進行中の取り込み・あとから文字起こしに中止を要求する。

        Cancellation is honoured at the next chunk boundary; the pipeline
        cleans up its recognizer and temporary files and always emits its
        finished event so the UI leaves the busy state."""
        if self._state not in (SessionState.IMPORTING, SessionState.TRANSCRIBING):
            return False
        self.transcription.cancel_current()
        return True

    def _run_transcribe_existing(
        self,
        audio_path: Path,
        mode: str,
        folder: Path,
        lesson: LessonRecord,
        diarization_speakers: int | None = None,
    ) -> None:
        try:
            self.transcription.transcribe_existing_audio(
                audio_path,
                mode,
                folder,
                lesson,
                diarization_speakers=diarization_speakers,
            )
        finally:
            self._processing_folder = None
            self._state = SessionState.IDLE

    def diarize_lesson_async(self, num_speakers: int) -> None:
        """Speaker-diarize the currently selected lesson's saved audio,
        without re-running ASR. Applies to whatever lesson is currently
        selected (self.current_folder / self.current_lesson), the same
        target as transcribe_current_audio_async.

        UI contract: emits ("diarization_started", (folder, lesson)) when
        the job begins, and ("diarization_finished", (folder, lesson))
        unconditionally when it ends (success or failure) so the UI can
        leave its busy state; the human-readable result ("話者分離が完了
        しました" / an error message) arrives separately through the
        existing ("status", ...) / ("error", ...) events."""
        if self.busy:
            if self.llm_session.busy:
                raise RuntimeError("LLMが処理中です。完了後に再試行してください。")
            raise RuntimeError("別の録音処理が進行中です。")
        if num_speakers is None or int(num_speakers) < 1:
            raise ValueError("num_speakers must be a positive integer")
        folder = self.current_folder
        lesson = self.current_lesson
        audio_path = self.current_audio_path()
        if folder is None or lesson is None or audio_path is None:
            raise RuntimeError("話者分離できる音声がありません。")
        if not lesson.segments:
            raise RuntimeError("話者分離できる文字起こしがありません。")

        self.release_chat_model()
        self.player.stop()
        self._state = SessionState.DIARIZING
        self._processing_folder = folder
        self.events.put(("diarization_started", (folder, lesson)))
        threading.Thread(
            target=self._run_diarize_lesson,
            args=(folder, lesson, int(num_speakers)),
            name="lesson-diarizer",
            daemon=True,
        ).start()

    def _run_diarize_lesson(
        self,
        folder: Path,
        lesson: LessonRecord,
        num_speakers: int,
    ) -> None:
        try:
            self.transcription.diarize_existing_lesson(folder, lesson, num_speakers)
        finally:
            self._processing_folder = None
            self._state = SessionState.IDLE
            self.events.put(("diarization_finished", (folder, lesson)))

    # ------------------------------------------------------------------
    # Lesson selection and management
    # ------------------------------------------------------------------

    def select_lesson(self, folder: Path) -> LessonRecord:
        with self._lock:
            self.player.stop()
            root = self.store.root.resolve()
            target = Path(folder).resolve()
            if target == root or not target.is_relative_to(root):
                raise ValueError("保存フォルダー外の記録は開けません。")
            lesson = self.store.load(target)
            self.current_folder = target
            self.current_lesson = lesson
            return lesson

    def rename_current_lesson(self, new_title: str) -> None:
        with self._lock:
            if self.current_folder is None or self.current_lesson is None:
                return
            if self._current_lesson_is_processing():
                raise RuntimeError("処理中のノートは名前を変更できません。")
            self.player.stop()
            self.current_folder = self.store.rename_lesson(
                self.current_folder, self.current_lesson, new_title
            )
            self.events.put(("lesson_renamed", (self.current_folder, self.current_lesson)))

    def delete_current_lesson(self) -> None:
        if self.busy:
            if self.llm_session.busy:
                raise RuntimeError("LLMが処理中のため削除できません。完了後に再試行してください。")
            raise RuntimeError("録音または取り込み中は削除できません。")
        with self._lock:
            folder = self.current_folder
            if folder is None:
                raise RuntimeError("削除する授業が選択されていません。")
            root = self.store.root.resolve()
            target = folder.resolve()
            if target == root or not target.is_relative_to(root):
                raise RuntimeError("保存フォルダー外のデータは削除できません。")
            recognizable = any(
                (target / name).is_file()
                for name in ("metadata.json", "transcript.json", "transcript.json.bak")
            )
            if not recognizable:
                raise RuntimeError("OtoWeaveの記録フォルダーとして確認できません。")
            self.player.stop()
            # Move to the store trash (kept for 30 days) instead of an
            # immediate permanent delete.
            self.store.trash_lesson(target)
            self.current_folder = None
            self.current_lesson = None
            self.events.put(("lesson_deleted", target))

    def play_segment(self, segment: TranscriptSegment) -> None:
        path = self.current_audio_path()
        if path is None:
            self.events.put(("error", "この授業には再生できる音声がありません。"))
            return
        self.player.play(path, max(0.0, segment.start))

    def current_audio_path(self) -> Path | None:
        if self.current_folder is None or self.current_lesson is None:
            return None
        path = self.current_folder / self.current_lesson.audio_file
        return path if path.exists() else None

    def create_demo_lesson(self) -> None:
        self.demo.create_demo_lesson()

    def close(self) -> None:
        if self.recorder is not None:
            self.stop_lesson_async()
        # A running import / re-transcription could otherwise keep busy
        # high for tens of minutes while the close loop polls it.
        self.cancel_transcription()
        self.player.stop()
        # A running summary would otherwise block closing for up to an
        # hour; the subprocess is killed and its cache is left untouched.
        self.llm_session.cancel_summary()
        self.release_chat_model()

    # ------------------------------------------------------------------
    # Marks and transcript editing
    # ------------------------------------------------------------------

    def _current_lesson_is_processing(self) -> bool:
        """True when a background job is writing the selected lesson."""
        return (
            self._processing_folder is not None
            and self.current_folder == self._processing_folder
        )

    def mark_latest(self, mark_type: str) -> None:
        if mark_type not in {"important", "unclear", "question"}:
            raise ValueError(f"Unsupported mark type: {mark_type}")
        with self._lock:
            if self.current_lesson is None or self._current_lesson_is_processing():
                return
            if self.current_lesson.segments:
                segment = self.current_lesson.segments[-1]
                if mark_type == "important":
                    segment.important = not segment.important
                    segment.important_at = mark_time() if segment.important else ""
                elif mark_type == "unclear":
                    segment.unclear = not segment.unclear
                    segment.unclear_at = mark_time() if segment.unclear else ""
                else:
                    segment.question = not segment.question
                    segment.question_at = mark_time() if segment.question else ""
            elif self.recorder is not None:
                timestamp = self.recorder.elapsed_seconds
                segment = TranscriptSegment(
                    id=next_segment_id(self.current_lesson.segments),
                    start=timestamp,
                    end=timestamp + 0.1,
                    text={
                        "important": "ここは重要",
                        "unclear": "ここをあとで確認",
                        "question": "ここで質問したい",
                    }[mark_type],
                    important=mark_type == "important",
                    unclear=mark_type == "unclear",
                    question=mark_type == "question",
                    important_at=mark_time() if mark_type == "important" else "",
                    unclear_at=mark_time() if mark_type == "unclear" else "",
                    question_at=mark_time() if mark_type == "question" else "",
                )
                self.current_lesson.segments.append(segment)
            else:
                return
            self._save_current()
            self.events.put(("segments_changed", self.current_lesson))

    def toggle_segment_mark(self, segment_id: str, mark_type: str) -> None:
        if mark_type not in {"important", "unclear", "question"}:
            raise ValueError(f"Unsupported mark type: {mark_type}")
        with self._lock:
            if self.current_lesson is None or self._current_lesson_is_processing():
                return
            segment = next((item for item in self.current_lesson.segments if item.id == segment_id), None)
            if segment is None:
                return
            if mark_type == "important":
                segment.important = not segment.important
                segment.important_at = mark_time() if segment.important else ""
            elif mark_type == "unclear":
                segment.unclear = not segment.unclear
                segment.unclear_at = mark_time() if segment.unclear else ""
            else:
                segment.question = not segment.question
                segment.question_at = mark_time() if segment.question else ""
            self._save_current()
            self.events.put(("segments_changed", self.current_lesson))

    def update_segment_text(self, segment_id: str, new_text: str) -> None:
        cleaned = " ".join(new_text.split()).strip()
        if not cleaned:
            raise ValueError("文字起こしを空にはできません。")
        with self._lock:
            if (self.recorder is not None or self._state is not SessionState.IDLE) or self.current_lesson is None or self.current_folder is None:
                raise RuntimeError("処理中は訂正できません。")
            segment = next((item for item in self.current_lesson.segments if item.id == segment_id), None)
            if segment is None:
                raise KeyError(segment_id)
            before = segment.text
            if before == cleaned:
                return
            segment.text = cleaned
            segment.edited = True
            self.store.append_correction(
                self.current_folder,
                self.current_lesson.lesson_id,
                segment.id,
                before,
                cleaned,
            )
            self._save_current()
            self.events.put(("segments_changed", self.current_lesson))

    def replace_transcript_text(self, new_text: str) -> None:
        blocks = [
            " ".join(block.split()).strip()
            for block in re.split(r"\n\s*\n", new_text)
            if block.strip()
        ]
        if not blocks:
            raise ValueError("文字起こしを空にはできません。")
        with self._lock:
            if (self.recorder is not None or self._state is not SessionState.IDLE) or self.current_lesson is None or self.current_folder is None:
                raise RuntimeError("処理中は訂正できません。")
            segments = self.current_lesson.segments
            if not segments:
                raise RuntimeError("訂正する文字起こしがありません。")

            if len(blocks) == len(segments):
                changed = False
                for segment, cleaned in zip(segments, blocks):
                    if segment.text == cleaned:
                        continue
                    self.store.append_correction(
                        self.current_folder,
                        self.current_lesson.lesson_id,
                        segment.id,
                        segment.text,
                        cleaned,
                    )
                    segment.text = cleaned
                    segment.edited = True
                    changed = True
                if not changed:
                    return
            else:
                before = "\n\n".join(segment.text for segment in segments)
                after = "\n\n".join(blocks)
                self.store.append_correction(
                    self.current_folder,
                    self.current_lesson.lesson_id,
                    "full_transcript",
                    before,
                    after,
                )
                self.current_lesson.segments = remap_edited_blocks(segments, blocks)

            self._save_current()
            stale_summary = (
                self.current_folder
                / "postprocess"
                / "school_record.md"
            )
            stale_summary.unlink(missing_ok=True)
            self.events.put(("segments_changed", self.current_lesson))

    def update_segment_speaker(self, segment_id: str, speaker: str) -> None:
        with self._lock:
            if self.busy or self.current_lesson is None:
                raise RuntimeError("処理中は話者名を変更できません。")
            segment = next((item for item in self.current_lesson.segments if item.id == segment_id), None)
            if segment is None:
                raise KeyError(segment_id)
            segment.speaker = " ".join(speaker.split()).strip()
            segment.edited = True
            self._save_current()
            self.events.put(("segments_changed", self.current_lesson))

    def split_segment(self, segment_id: str, left_text: str, right_text: str) -> None:
        left = " ".join(left_text.split()).strip()
        right = " ".join(right_text.split()).strip()
        if not left or not right:
            raise ValueError("分割する両方の文章が必要です。")
        with self._lock:
            if self.busy or self.current_lesson is None:
                raise RuntimeError("処理中は区間を分割できません。")
            index = next(
                (i for i, item in enumerate(self.current_lesson.segments) if item.id == segment_id),
                None,
            )
            if index is None:
                raise KeyError(segment_id)
            segment = self.current_lesson.segments[index]
            original_end = segment.end
            ratio = len(left) / max(1, len(left) + len(right))
            split_time = segment.start + max(0.1, original_end - segment.start) * ratio
            segment.text = left
            segment.end = split_time
            segment.edited = True
            new_segment = TranscriptSegment(
                id=next_segment_id(self.current_lesson.segments),
                start=split_time,
                end=max(split_time + 0.1, original_end),
                text=right,
                speaker=segment.speaker,
                edited=True,
            )
            self.current_lesson.segments.insert(index + 1, new_segment)
            self._save_current()
            self.events.put(("segments_changed", self.current_lesson))

    def merge_segment_with_next(self, segment_id: str) -> None:
        with self._lock:
            if self.busy or self.current_lesson is None:
                raise RuntimeError("処理中は区間を結合できません。")
            index = next(
                (i for i, item in enumerate(self.current_lesson.segments) if item.id == segment_id),
                None,
            )
            if index is None or index + 1 >= len(self.current_lesson.segments):
                raise ValueError("次に結合できる区間がありません。")
            segment = self.current_lesson.segments[index]
            following = self.current_lesson.segments.pop(index + 1)
            separator = "、" if re.search(r"[ぁ-んァ-ヴ一-龯]", segment.text + following.text) else " "
            segment.text = segment.text.rstrip(" 、,") + separator + following.text.lstrip(" 、,")
            segment.end = max(segment.end, following.end)
            segment.important = segment.important or following.important
            segment.unclear = segment.unclear or following.unclear
            segment.question = segment.question or following.question
            segment.important_at = segment.important_at or following.important_at
            segment.unclear_at = segment.unclear_at or following.unclear_at
            segment.question_at = segment.question_at or following.question_at
            segment.edited = True
            self._save_current()
            self.events.put(("segments_changed", self.current_lesson))

    def _save_current(self, throttle: bool = False) -> None:
        if self.current_folder is None or self.current_lesson is None:
            return
        if throttle:
            # Time-based throttling for live recognition results: within
            # LIVE_SAVE_INTERVAL_SECONDS of the previous save only mark the
            # lesson dirty; the next confirmed sentence after the interval
            # (or the unconditional full save at stop/error) writes it out.
            # A crash therefore loses at most a few seconds of text.
            now = time.monotonic()
            if now - self._last_live_save_monotonic < self.LIVE_SAVE_INTERVAL_SECONDS:
                self._live_save_dirty = True
                return
        self._last_live_save_monotonic = time.monotonic()
        self._live_save_dirty = False
        # During recording every confirmed sentence triggers a save;
        # skip the markdown regeneration until the final save.
        self.store.save(
            self.current_folder,
            self.current_lesson,
            light=self.recorder is not None,
        )

    # ------------------------------------------------------------------
    # LLM summarisation and chat Q&A (delegated to LlmSession)
    # ------------------------------------------------------------------

    def summarize_async(
        self,
        lesson: LessonRecord,
        folder: Path,
        model_path: Path,
        template: dict[str, Any] | None = None,
    ) -> None:
        """Export the transcript and generate a local summary."""
        if self.busy:
            if self.llm_session.busy:
                raise RuntimeError("LLMは現在処理中です。完了後に再試行してください。")
            raise RuntimeError("録音または音声処理中は要約を開始できません。")
        self.release_chat_model()
        self.llm_session.summarize_async(lesson, folder, model_path, template)

    def chat_async(
        self,
        question: str,
        lesson_folder: Path,
        model_path: Path,
    ) -> None:
        """Answer a question about the current session in a background thread."""
        if self.busy:
            if self.llm_session.busy:
                raise RuntimeError("LLMは現在処理中です。回答が届くまでお待ちください。")
            raise RuntimeError("録音または音声処理中はチャットを利用できません。")
        self.llm_session.chat_async(question, lesson_folder, model_path)

    def cancel_summary(self) -> bool:
        """Cancel the running summary subprocess, if any."""
        return self.llm_session.cancel_summary()

    def reset_chat(self) -> None:
        """Clear conversation history so the next question starts fresh."""
        self.llm_session.reset_chat()

    def release_chat_model(self) -> bool:
        """Release the resident chat model, or defer release until inference ends."""
        return self.llm_session.release_model()
