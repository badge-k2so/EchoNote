"""Batch transcription pipelines (import / re-transcribe), separated from
the controller.

The service always saves to the lesson folder it was given, never to the
controller's currently selected lesson, so selecting another note during a
long import can no longer overwrite that note's data.
"""
from __future__ import annotations

import gc
import json
import queue
import re
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .app_logging import log_exception
from .asr import (
    AsrThreadConfig,
    EnglishRecognizer,
    JapaneseRecognizer,
    Qwen3AsrRecognizer,
    SPEECHBRAIN_LANGUAGE_ID_DIR,
    SpeechBrainLanguageIdentifier,
    select_asr_threads,
    split_text_with_times,
)
from .audio import (
    SAMPLE_RATE,
    AdaptiveVad,
    SpeechChunk,
    convert_audio_to_pcm,
    convert_pcm_to_opus,
)
from .diarization import (
    EMBEDDING_MODEL as DIARIZATION_EMBEDDING_MODEL,
    SEGMENTATION_MODEL as DIARIZATION_SEGMENTATION_MODEL,
    SpeakerDiarizer,
    assign_speakers,
)
from .models import LessonRecord, TranscriptSegment, append_readable_segment
from .segment_editing import mark_time, transfer_marks
from .storage import LessonStore


def set_suggested_title(lesson: LessonRecord) -> None:
    transcript = "。".join(
        segment.text
        for segment in lesson.segments
        if segment.text and segment.text != "ここをあとで確認"
    )
    if not transcript:
        return
    try:
        from scripts.production.record_filename import extract_title

        title, source = extract_title(transcript, max_chars=36)
        if source != "fallback_no_meaningful_text":
            lesson.suggested_title = title
    except Exception:
        pass


def set_import_title(source_path: Path, lesson: LessonRecord) -> None:
    try:
        from scripts.production.record_filename import extract_filename_label

        label = extract_filename_label(source_path)
    except Exception:
        label = ""
    if label:
        lesson.suggested_title = label
    else:
        set_suggested_title(lesson)


def recording_datetime(source_path: Path) -> datetime:
    modified = datetime.fromtimestamp(source_path.stat().st_mtime).astimezone()
    try:
        from scripts.production.record_filename import extract_recorded_date

        recorded_date, _ = extract_recorded_date(source_path)
        return datetime.combine(recorded_date, modified.timetz())
    except Exception:
        return modified


class TranscriptionCancelled(Exception):
    """Raised inside a pipeline when the user requested cancellation."""


_DIARIZED_SPEAKER_LABEL_RE = re.compile(r"^話者\d+$")


def _is_diarizable_speaker_label(speaker: str) -> bool:
    """Whether a segment's current speaker label may be overwritten by a
    fresh diarization pass: empty, or a previous "話者N" auto-label. Any
    other value is a name the user typed in by hand and must survive a
    post-hoc diarization run untouched."""
    return speaker == "" or bool(_DIARIZED_SPEAKER_LABEL_RE.match(speaker))


class TranscriptionService:
    # Import progress saves are throttled to one every few seconds: each
    # save rewrites transcript.json + metadata + marks with fsync, so a
    # save per recognized chunk becomes O(n^2) disk work on long audio.
    # A crash loses at most the last few seconds of recognized text; the
    # terminal save (success, cancel or error) always flushes everything.
    SAVE_INTERVAL_SECONDS = 3.0

    def __init__(
        self,
        project_root: Path,
        store: LessonStore,
        events: "queue.Queue[tuple[str, Any]]",
        ffmpeg: Path,
        correct_text: Callable[[str], str],
        on_lesson_ready: Callable[[Path, LessonRecord], None],
    ) -> None:
        self.project_root = project_root
        self.store = store
        self.events = events
        self.ffmpeg = ffmpeg
        self._correct_text = correct_text
        # Called when a pipeline finishes so the controller can make the
        # processed lesson the current selection.
        self._on_lesson_ready = on_lesson_ready
        # Cancellation is requested from another thread and honoured at
        # chunk boundaries inside the running pipeline.
        self._cancel_event = threading.Event()
        self._last_progress_save_monotonic = 0.0
        self._progress_save_dirty = False

    def cancel_current(self) -> None:
        """Ask the running pipeline to stop at the next chunk boundary.

        Safe to call at any time, also when nothing is running (the flag
        is cleared when the next pipeline starts)."""
        self._cancel_event.set()

    def _begin_pipeline(self) -> None:
        self._cancel_event.clear()
        self._last_progress_save_monotonic = 0.0
        self._progress_save_dirty = False

    # ------------------------------------------------------------------
    # Import pipeline
    # ------------------------------------------------------------------

    def import_audio(
        self,
        source_path: Path,
        mode: str,
        folder: Path,
        lesson: LessonRecord,
    ) -> None:
        pcm_path = folder / "recording.pcm"
        cancelled = False
        try:
            self._begin_pipeline()
            self.events.put(("status", "録音データを16 kHzモノへ変換中"))
            convert_audio_to_pcm(self.ffmpeg, source_path, pcm_path)
            lesson.audio_file = pcm_path.name
            self.store.save(folder, lesson)

            if mode in {"japanese", "english"}:
                self.events.put(("status", "ローカル音声認識モデルを準備中"))
                thread_config = select_asr_threads("file")
                self._record_asr_thread_config(lesson, folder, thread_config)
                recognizer = None
                try:
                    recognizer = (
                        JapaneseRecognizer(num_threads=thread_config.num_threads)
                        if mode == "japanese"
                        else EnglishRecognizer(
                            self.project_root / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
                            num_threads=thread_config.num_threads,
                        )
                    )
                    vad = AdaptiveVad(max_chunk_seconds=12.0)
                    total_bytes = pcm_path.stat().st_size
                    processed_bytes = 0
                    block_samples = 1600
                    with pcm_path.open("rb") as stream:
                        while True:
                            if self._cancel_event.is_set():
                                # Keep what has been recognized so far and
                                # finish the import normally (audio is kept).
                                cancelled = True
                                break
                            data = stream.read(block_samples * 2)
                            if not data:
                                break
                            samples = np.frombuffer(data, dtype=np.int16).copy()
                            start = processed_bytes / 2 / SAMPLE_RATE
                            processed_bytes += len(data)
                            for chunk in vad.process(samples, start):
                                self._recognize_import_chunk(
                                    recognizer, chunk, folder, lesson
                                )
                            if processed_bytes % (block_samples * 2 * 50) == 0:
                                percent = min(99, int(processed_bytes * 100 / max(1, total_bytes)))
                                self.events.put(("status", f"録音データを文字起こし中 {percent}%"))
                        if not cancelled:
                            for chunk in vad.flush(force=True):
                                self._recognize_import_chunk(
                                    recognizer, chunk, folder, lesson
                                )
                finally:
                    if recognizer is not None:
                        del recognizer
                    gc.collect()

            self.events.put(("status", "音声をOpusへ圧縮中"))
            opus_path = folder / "audio.opus"
            convert_pcm_to_opus(self.ffmpeg, pcm_path, opus_path)
            lesson.audio_file = opus_path.name
            pcm_path.unlink(missing_ok=True)
            lesson.status = "complete"
            set_import_title(source_path, lesson)
            # Terminal flush: also persists chunks skipped by throttling.
            self.store.save(folder, lesson)
            self._progress_save_dirty = False
            self._on_lesson_ready(folder, lesson)
            self.events.put(("import_finished", (folder, lesson)))
            if cancelled:
                self.events.put(
                    ("status", "取り込みを中止しました。途中までの文字起こしを保存しています。")
                )
            else:
                self.events.put(("status", "録音データを取り込みました"))
        except Exception as exc:
            log_exception("音声ファイルの取り込みに失敗", exc)
            lesson.status = "needs_attention"
            lesson.audio_file = pcm_path.name if pcm_path.exists() else ""
            # Terminal flush on error, too: recognized text is kept.
            self.store.save(folder, lesson)
            self._progress_save_dirty = False
            self._on_lesson_ready(folder, lesson)
            self.events.put(("import_finished", (folder, lesson)))
            self.events.put(
                (
                    "error",
                    "録音データを取り込めませんでした。"
                    "途中までの文字起こしは保存されています。もう一度お試しください。",
                )
            )

    # ------------------------------------------------------------------
    # Re-transcription pipeline for saved audio
    # ------------------------------------------------------------------

    def transcribe_existing_audio(
        self,
        audio_path: Path,
        mode: str,
        folder: Path,
        lesson: LessonRecord,
        diarization_speakers: int | None = None,
    ) -> None:
        pcm_path = folder / "transcription_input.pcm"
        try:
            self._begin_pipeline()
            self.events.put(("status", "保存済み音声を文字起こし用に準備中"))
            if audio_path.suffix.lower() == ".pcm":
                shutil.copyfile(audio_path, pcm_path)
            else:
                convert_audio_to_pcm(self.ffmpeg, audio_path, pcm_path)

            self.events.put(("status", "ローカル音声認識モデルを準備中"))
            thread_config = select_asr_threads("file")
            self._record_asr_thread_config(lesson, folder, thread_config)
            if mode == "mixed":
                new_segments = self._transcribe_mixed_pcm(
                    pcm_path,
                    folder,
                    thread_config.num_threads,
                )
            elif mode == "mixed_qwen17":
                self.events.put(("status", "Qwen3-ASR-1.7Bを準備中（実験モード）"))
                recognizer = Qwen3AsrRecognizer(
                    self.project_root,
                    folder / "logs",
                    num_threads=thread_config.num_threads,
                )
                try:
                    new_segments = self._transcribe_single_language_pcm(
                        pcm_path,
                        recognizer,
                        max_chunk_seconds=28.0,
                        status_prefix="Qwen3-ASRで日英混在を文字起こし中",
                    )
                finally:
                    recognizer.close()
                    del recognizer
                    gc.collect()
            else:
                recognizer = None
                try:
                    recognizer = (
                        JapaneseRecognizer(num_threads=thread_config.num_threads)
                        if mode == "japanese"
                        else EnglishRecognizer(
                            self.project_root / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
                            num_threads=thread_config.num_threads,
                        )
                    )
                    new_segments = self._transcribe_single_language_pcm(pcm_path, recognizer)
                finally:
                    if recognizer is not None:
                        del recognizer
                    gc.collect()

            if not new_segments:
                # ASR/VADが何も返さなかったときは既存の文字起こしと
                # マークを上書きせず、失敗として通知するだけにする。
                self.events.put(("transcription_finished", (folder, lesson)))
                self.events.put(
                    (
                        "error",
                        "音声から文字を検出できませんでした。"
                        "元の文字起こしはそのまま残しています。",
                    )
                )
                return

            if diarization_speakers is not None:
                self._run_diarization(
                    pcm_path, audio_path, new_segments, diarization_speakers, thread_config
                )

            transfer_marks(lesson.segments, new_segments)
            lesson.segments = new_segments
            lesson.language_mode = mode
            lesson.status = "complete"
            set_suggested_title(lesson)
            self.store.save(folder, lesson)
            self._on_lesson_ready(folder, lesson)
            self.events.put(("transcription_finished", (folder, lesson)))
            self.events.put(("status", "あとから文字起こしが完了しました"))
        except TranscriptionCancelled:
            # ユーザーによる中止。既存の segments・マーク・language_mode には
            # 一切触れず、処理中表示だけを解除する。recognizer と一時PCMは
            # 既存の finally でクリーンアップ済み。
            self.events.put(("transcription_finished", (folder, lesson)))
            self.events.put(
                (
                    "status",
                    "あとから文字起こしを中止しました。元の文字起こしはそのまま残っています。",
                )
            )
        except Exception as exc:
            log_exception("あとから文字起こしに失敗", exc)
            self.events.put(("transcription_finished", (folder, lesson)))
            self.events.put(
                (
                    "error",
                    "あとから文字起こしを完了できませんでした。"
                    "元の文字起こしはそのまま残っています。もう一度お試しください。",
                )
            )
        finally:
            pcm_path.unlink(missing_ok=True)

    def _transcribe_single_language_pcm(
        self,
        pcm_path: Path,
        recognizer,
        max_chunk_seconds: float = 12.0,
        status_prefix: str = "あとから文字起こし中",
    ) -> list[TranscriptSegment]:
        new_segments: list[TranscriptSegment] = []
        vad = AdaptiveVad(max_chunk_seconds=max_chunk_seconds)
        total_bytes = pcm_path.stat().st_size
        processed_bytes = 0
        block_samples = 1600
        with pcm_path.open("rb") as stream:
            while True:
                self._check_cancelled()
                data = stream.read(block_samples * 2)
                if not data:
                    break
                samples = np.frombuffer(data, dtype=np.int16).copy()
                start = processed_bytes / 2 / SAMPLE_RATE
                processed_bytes += len(data)
                for chunk in vad.process(samples, start):
                    self._recognize_chunk_into(recognizer, chunk, new_segments)
                if processed_bytes % (block_samples * 2 * 50) == 0:
                    percent = min(99, int(processed_bytes * 100 / max(1, total_bytes)))
                    self.events.put(("status", f"{status_prefix} {percent}%"))
            for chunk in vad.flush(force=True):
                self._recognize_chunk_into(recognizer, chunk, new_segments)
        return new_segments

    def _transcribe_mixed_pcm(
        self,
        pcm_path: Path,
        folder: Path,
        num_threads: int,
    ) -> list[TranscriptSegment]:
        chunk_dir = folder / ".mixed_asr_chunks"
        shutil.rmtree(chunk_dir, ignore_errors=True)
        chunk_dir.mkdir(parents=True)
        chunks: list[tuple[float, float, Path]] = []
        try:
            self.events.put(("status", "日英混在: 発話区間を準備中"))
            vad = AdaptiveVad(max_chunk_seconds=12.0)
            processed_samples = 0
            with pcm_path.open("rb") as stream:
                while True:
                    self._check_cancelled()
                    data = stream.read(1600 * 2)
                    if not data:
                        break
                    samples = np.frombuffer(data, dtype=np.int16).copy()
                    for chunk in vad.process(samples, processed_samples / SAMPLE_RATE):
                        path = chunk_dir / f"chunk_{len(chunks):04d}.npy"
                        np.save(path, chunk.samples)
                        chunks.append((chunk.start, chunk.end, path))
                    processed_samples += samples.size
                for chunk in vad.flush(force=True):
                    path = chunk_dir / f"chunk_{len(chunks):04d}.npy"
                    np.save(path, chunk.samples)
                    chunks.append((chunk.start, chunk.end, path))
            if not chunks:
                return []

            self.events.put(("status", "日英混在: 発話言語を判定中"))
            language_id = SpeechBrainLanguageIdentifier(
                self.project_root / SPEECHBRAIN_LANGUAGE_ID_DIR,
                num_threads=num_threads,
            )
            decisions = []
            routing_log = []
            for index, (start, end, path) in enumerate(chunks):
                self._check_cancelled()
                decision = language_id.detect(np.load(path))
                decisions.append(decision)
                routing_log.append(
                    {
                        "chunk": index,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "language": decision.language,
                        "confidence": round(decision.confidence, 6),
                        "margin": round(decision.margin, 6),
                        "japanese_score": round(decision.japanese_score, 6),
                        "english_score": round(decision.english_score, 6),
                        "uncertain": decision.uncertain,
                        "reason": decision.reason,
                    }
                )
                self.events.put(("status", f"日英混在: 言語判定 {index + 1}/{len(chunks)}"))
            log_dir = folder / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "language_routing.json").write_text(
                json.dumps(
                    {
                        "model": "SpeechBrain ECAPA-TDNN VoxLingua107",
                        "threads": num_threads,
                        "chunks": routing_log,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            del language_id
            gc.collect()

            ja_results: dict[int, str] = {}
            japanese_indexes = [
                index for index, decision in enumerate(decisions)
                if decision.language == "japanese"
            ]
            if japanese_indexes:
                self.events.put(("status", "日英混在: 日本語区間を認識中"))
                japanese = JapaneseRecognizer(num_threads=num_threads)
                for progress, index in enumerate(japanese_indexes, start=1):
                    self._check_cancelled()
                    ja_results[index] = japanese.transcribe(np.load(chunks[index][2]))
                    self.events.put(
                        ("status", f"日英混在: 日本語区間 {progress}/{len(japanese_indexes)}")
                    )
                del japanese
                gc.collect()

            en_results: dict[int, str] = {}
            english_indexes = [
                index for index, decision in enumerate(decisions)
                if decision.language == "english"
            ]
            if english_indexes:
                self.events.put(("status", "日英混在: 英語区間を認識中"))
                english = EnglishRecognizer(
                    self.project_root / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
                    num_threads=num_threads,
                )
                for progress, index in enumerate(english_indexes, start=1):
                    self._check_cancelled()
                    en_results[index] = english.transcribe(np.load(chunks[index][2]))
                    self.events.put(
                        ("status", f"日英混在: 英語区間 {progress}/{len(english_indexes)}")
                    )
                del english
                gc.collect()

            segments: list[TranscriptSegment] = []
            for index, ((start, end, _), decision) in enumerate(zip(chunks, decisions)):
                text = (
                    ja_results.get(index, "")
                    if decision.language == "japanese"
                    else en_results.get(index, "")
                )
                text = self._correct_text(text)
                for sentence in split_text_with_times(text, start, end):
                    incoming = TranscriptSegment(
                        id=f"seg_{len(segments) + 1:04d}",
                        start=sentence.start,
                        end=sentence.end,
                        text=sentence.text,
                        unclear=decision.uncertain,
                        unclear_at=mark_time() if decision.uncertain else "",
                    )
                    append_readable_segment(segments, incoming)
            return segments
        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

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

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise TranscriptionCancelled()

    def _run_diarization(
        self,
        pcm_path: Path,
        audio_path: Path,
        segments: list[TranscriptSegment],
        num_speakers: int,
        thread_config: AsrThreadConfig,
    ) -> None:
        """Best-effort speaker diarization: any failure here must not affect
        the ASR result already computed, so every exception is caught and
        only logged, leaving segment.speaker at its default empty value."""
        try:
            self.events.put(("status", "話者分離中"))
            samples = self._load_samples_for_diarization(pcm_path, audio_path)
            diarizer = SpeakerDiarizer(
                self.project_root / DIARIZATION_SEGMENTATION_MODEL,
                self.project_root / DIARIZATION_EMBEDDING_MODEL,
                num_threads=thread_config.num_threads,
            )
            try:
                result = diarizer.diarize(samples, num_speakers)
                assign_speakers(segments, result)
            finally:
                del diarizer
        except Exception as exc:
            log_exception("話者分離に失敗", exc)
        finally:
            gc.collect()

    def _load_samples_for_diarization(self, pcm_path: Path, audio_path: Path) -> np.ndarray:
        """Reuse the 16kHz mono PCM this pipeline already produced whenever
        possible (int16 -> normalized float32), instead of re-decoding the
        original file. pcm_path is expected to exist here (it is only
        cleaned up after this call returns), so the librosa fallback below
        is only a safety net for an unexpected cleanup race."""
        try:
            raw = pcm_path.read_bytes()
            if raw:
                return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except OSError:
            pass
        return self._decode_audio_file_for_diarization(audio_path)

    @staticmethod
    def _decode_audio_file_for_diarization(audio_path: Path) -> np.ndarray:
        """Decode any saved lesson audio (opus/wav/pcm/...) straight to
        16kHz mono float32 for diarization, with no intermediate PCM file
        involved."""
        import librosa

        samples, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        return samples.astype(np.float32)

    # ------------------------------------------------------------------
    # Post-hoc diarization of an already-transcribed lesson
    # ------------------------------------------------------------------

    def diarize_existing_lesson(
        self,
        folder: Path,
        lesson: LessonRecord,
        num_speakers: int,
    ) -> bool:
        """Speaker-diarize a lesson's saved audio without re-running ASR.

        Locates the lesson's audio the same way transcribe_current_audio
        does (folder / lesson.audio_file), decodes it directly to 16kHz
        mono float32 (no intermediate PCM is kept once a lesson is saved),
        runs SpeakerDiarizer, and assigns speaker labels onto the already
        persisted TranscriptSegment objects in place.

        A segment whose speaker the user already renamed by hand (anything
        other than empty or a previous "話者N" auto-label) is excluded from
        the segments passed to assign_speakers, so it can never be
        overwritten here.

        Any failure (missing audio, no transcript, model load/diarize
        error) is logged and reported through self.events; the transcript
        on disk is never touched in that case. Returns True on success,
        False otherwise."""
        audio_path = folder / lesson.audio_file if lesson.audio_file else folder
        if not lesson.audio_file or not audio_path.is_file():
            log_exception(
                "話者分離に失敗",
                FileNotFoundError(f"lesson audio not found: {audio_path}"),
            )
            self.events.put(
                (
                    "error",
                    "話者分離用の音声ファイルが見つかりませんでした。",
                )
            )
            return False
        if not lesson.segments:
            self.events.put(
                ("error", "話者分離できる文字起こしがありません。")
            )
            return False

        try:
            self.events.put(("status", "話者分離用の音声を準備中"))
            samples = self._decode_audio_file_for_diarization(audio_path)
            thread_config = select_asr_threads("file")
            self.events.put(("status", "話者分離中"))
            diarizer = SpeakerDiarizer(
                self.project_root / DIARIZATION_SEGMENTATION_MODEL,
                self.project_root / DIARIZATION_EMBEDDING_MODEL,
                num_threads=thread_config.num_threads,
            )
            try:
                result = diarizer.diarize(samples, num_speakers)
            finally:
                del diarizer
            eligible_segments = [
                segment
                for segment in lesson.segments
                if _is_diarizable_speaker_label(segment.speaker)
            ]
            assign_speakers(eligible_segments, result)
            self.store.save(folder, lesson)
            self._on_lesson_ready(folder, lesson)
            self.events.put(("status", "話者分離が完了しました"))
            return True
        except Exception as exc:
            log_exception("話者分離に失敗", exc)
            self.events.put(
                (
                    "error",
                    "話者分離を完了できませんでした。文字起こしはそのまま残っています。",
                )
            )
            return False
        finally:
            gc.collect()

    def _recognize_import_chunk(
        self,
        recognizer,
        chunk: SpeechChunk,
        folder: Path,
        lesson: LessonRecord,
    ) -> None:
        self._recognize_chunk_into(recognizer, chunk, lesson.segments)
        if lesson.segments:
            # Always save to the import's own folder: the user may have
            # selected a different lesson while this import runs.
            self._save_import_progress(folder, lesson)
            self.events.put(("segments_changed", lesson))

    def _save_import_progress(self, folder: Path, lesson: LessonRecord) -> None:
        """Throttled progress save during import (see SAVE_INTERVAL_SECONDS).

        Within the interval only a dirty flag is set; the terminal save in
        import_audio (success, cancel and error paths) always flushes, so a
        crash can lose at most the last few seconds of recognized text."""
        now = time.monotonic()
        if now - self._last_progress_save_monotonic < self.SAVE_INTERVAL_SECONDS:
            self._progress_save_dirty = True
            return
        self._last_progress_save_monotonic = now
        self._progress_save_dirty = False
        self.store.save(folder, lesson)

    def _recognize_chunk_into(
        self,
        recognizer,
        chunk: SpeechChunk,
        segments: list[TranscriptSegment],
    ) -> None:
        text = self._correct_text(recognizer.transcribe(chunk.samples))
        for sentence in split_text_with_times(text, chunk.start, chunk.end):
            append_readable_segment(
                segments,
                TranscriptSegment(
                    id=f"seg_{len(segments) + 1:04d}",
                    start=sentence.start,
                    end=sentence.end,
                    text=sentence.text,
                ),
            )
