import queue as queue_module
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from otoweave_app.diarization import (
    DiarizationResult,
    DiarizedSpan,
    SpeakerDiarizer,
    assign_speakers,
    diarization_available,
)
from otoweave_app.models import LessonRecord, TranscriptSegment
from otoweave_app.storage import LessonStore
from otoweave_app.transcription_service import TranscriptionService


class AssignSpeakersArgmaxTests(unittest.TestCase):
    """重複時間argmaxで話者を割り当てること（中点方式は禁止）。"""

    def test_dominant_speaker_by_overlap_time_wins_over_midpoint(self) -> None:
        # Segment 0..12, midpoint = 6.0. Speaker 0 owns only a tiny sliver
        # (5.9..6.1) that happens to cover the midpoint; speaker 1 owns the
        # remaining 11.8s. A midpoint/nearest-span heuristic would pick
        # speaker 0; overlap-time argmax must pick speaker 1.
        segment = TranscriptSegment("seg_0001", 0.0, 12.0, "テスト発話です。")
        result = DiarizationResult(
            spans=[
                DiarizedSpan(5.9, 6.1, 0),
                DiarizedSpan(0.0, 5.9, 1),
                DiarizedSpan(6.1, 12.0, 1),
            ]
        )
        assign_speakers([segment], result)
        self.assertEqual(segment.speaker, "話者2")

    def test_low_purity_leaves_segment_unassigned(self) -> None:
        segment = TranscriptSegment("seg_0001", 0.0, 10.0, "混在した発話です。")
        result = DiarizationResult(
            spans=[
                DiarizedSpan(0.0, 6.0, 0),
                DiarizedSpan(6.0, 10.0, 1),
            ]
        )
        # Dominant speaker share is 0.6, below the default 0.9 threshold.
        assign_speakers([segment], result)
        self.assertEqual(segment.speaker, "")

    def test_purity_exactly_at_threshold_is_assigned(self) -> None:
        segment = TranscriptSegment("seg_0001", 0.0, 10.0, "境界のテストです。")
        result = DiarizationResult(
            spans=[
                DiarizedSpan(0.0, 9.0, 0),
                DiarizedSpan(9.0, 10.0, 1),
            ]
        )
        # Dominant share is exactly 0.9: >= threshold must assign.
        assign_speakers([segment], result, purity_threshold=0.9)
        self.assertEqual(segment.speaker, "話者1")

    def test_no_overlap_leaves_segment_unassigned(self) -> None:
        segment = TranscriptSegment("seg_0001", 20.0, 25.0, "重複なしです。")
        result = DiarizationResult(spans=[DiarizedSpan(0.0, 10.0, 0)])
        assign_speakers([segment], result)
        self.assertEqual(segment.speaker, "")

    def test_labels_follow_ascending_speaker_index_not_appearance_order(self) -> None:
        # Spans are given with the higher speaker id first, to prove
        # labeling is by ascending speaker index, not order of appearance.
        segment_a = TranscriptSegment("seg_0001", 0.0, 5.0, "話者7の発話です。")
        segment_b = TranscriptSegment("seg_0002", 5.0, 10.0, "話者3の発話です。")
        result = DiarizationResult(
            spans=[
                DiarizedSpan(0.0, 5.0, 7),
                DiarizedSpan(5.0, 10.0, 3),
            ]
        )
        assign_speakers([segment_a, segment_b], result)
        self.assertEqual(segment_a.speaker, "話者2")  # speaker id 7 -> second-lowest index
        self.assertEqual(segment_b.speaker, "話者1")  # speaker id 3 -> lowest index

    def test_empty_result_leaves_all_segments_unassigned(self) -> None:
        segment = TranscriptSegment("seg_0001", 0.0, 5.0, "話者情報なしです。")
        assign_speakers([segment], DiarizationResult(spans=[]))
        self.assertEqual(segment.speaker, "")


class DiarizationAvailableTests(unittest.TestCase):
    def test_available_true_when_both_files_present_with_min_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seg_dir = root / "models" / "diarization" / "sherpa-onnx-pyannote-segmentation-3-0"
            seg_dir.mkdir(parents=True)
            (seg_dir / "model.onnx").write_bytes(b"\x00" * 2_000_000)
            (root / "models" / "diarization" / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx").write_bytes(
                b"\x00" * 20_000_000
            )
            self.assertTrue(diarization_available(root))

    def test_available_false_when_a_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seg_dir = root / "models" / "diarization" / "sherpa-onnx-pyannote-segmentation-3-0"
            seg_dir.mkdir(parents=True)
            (seg_dir / "model.onnx").write_bytes(b"\x00" * 2_000_000)
            # Embedding model missing entirely.
            self.assertFalse(diarization_available(root))

    def test_available_false_when_file_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seg_dir = root / "models" / "diarization" / "sherpa-onnx-pyannote-segmentation-3-0"
            seg_dir.mkdir(parents=True)
            (seg_dir / "model.onnx").write_bytes(b"\x00" * 10)  # truncated download
            (root / "models" / "diarization" / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx").write_bytes(
                b"\x00" * 20_000_000
            )
            self.assertFalse(diarization_available(root))


class SpeakerDiarizerValidationTests(unittest.TestCase):
    def test_missing_model_files_raise_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                SpeakerDiarizer(root / "seg.onnx", root / "emb.onnx", num_threads=2)

    def test_num_speakers_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seg = root / "seg.onnx"
            emb = root / "emb.onnx"
            seg.write_bytes(b"x")
            emb.write_bytes(b"x")
            diarizer = SpeakerDiarizer(seg, emb, num_threads=2)
            for invalid in (0, -1, None):
                with self.assertRaises(ValueError):
                    diarizer.diarize(__import__("numpy").zeros(1600, dtype="float32"), invalid)


class TeacherNormalizationCoexistenceTests(unittest.TestCase):
    """新しい話者ラベルが既存の teacher 正規化を壊さないこと。"""

    def test_teacher_label_still_normalized_to_empty(self) -> None:
        segment = TranscriptSegment.from_dict(
            {"id": "seg_0001", "start": 0.0, "end": 1.0, "text": "x", "speaker": "teacher"}
        )
        self.assertEqual(segment.speaker, "")

    def test_diarization_label_round_trips_untouched(self) -> None:
        segment = TranscriptSegment.from_dict(
            {"id": "seg_0001", "start": 0.0, "end": 1.0, "text": "x", "speaker": "話者1"}
        )
        self.assertEqual(segment.speaker, "話者1")
        self.assertEqual(segment.to_dict()["speaker"], "話者1")


class TranscriptionServiceDiarizationTests(unittest.TestCase):
    """transcribe_existing_audio への diarization_speakers 組み込みの契約。"""

    def _make_lesson(self, store: LessonStore) -> tuple[Path, LessonRecord]:
        lesson = LessonRecord.create(
            "japanese",
            "microphone",
            now=datetime.fromisoformat("2026-07-01T09:00:00+09:00"),
        )
        lesson.status = "complete"
        lesson.audio_file = "recording.pcm"
        folder = store.create_lesson(lesson)
        (folder / "recording.pcm").write_bytes(b"\x00\x00" * 1600)
        return folder, lesson

    def test_diarization_speakers_none_keeps_existing_behavior(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            folder, lesson = self._make_lesson(store)
            new_segments = [TranscriptSegment("seg_0001", 0.0, 5.0, "話者分離なしです。")]

            events: "queue_module.Queue" = queue_module.Queue()
            service = TranscriptionService(
                project_root,
                store,
                events,
                ffmpeg=Path("ffmpeg"),
                correct_text=lambda text: text,
                on_lesson_ready=lambda f, l: None,
            )

            with patch.object(
                TranscriptionService, "_transcribe_single_language_pcm", return_value=new_segments
            ), patch(
                "otoweave_app.transcription_service.JapaneseRecognizer", return_value=object()
            ), patch.object(
                TranscriptionService, "_run_diarization"
            ) as mock_run_diarization:
                service.transcribe_existing_audio(
                    folder / "recording.pcm", "japanese", folder, lesson
                )

            mock_run_diarization.assert_not_called()
            self.assertEqual(lesson.segments, new_segments)
            self.assertEqual(lesson.segments[0].speaker, "")

    def test_diarization_success_assigns_speaker_labels(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            folder, lesson = self._make_lesson(store)
            new_segments = [
                TranscriptSegment("seg_0001", 0.0, 5.0, "先生の発話です。"),
                TranscriptSegment("seg_0002", 5.0, 10.0, "生徒の発話です。"),
            ]

            class FakeDiarizer:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                def diarize(self, samples, num_speakers):
                    assert num_speakers == 2
                    return DiarizationResult(
                        spans=[
                            DiarizedSpan(0.0, 5.0, 0),
                            DiarizedSpan(5.0, 10.0, 1),
                        ]
                    )

            events: "queue_module.Queue" = queue_module.Queue()
            service = TranscriptionService(
                project_root,
                store,
                events,
                ffmpeg=Path("ffmpeg"),
                correct_text=lambda text: text,
                on_lesson_ready=lambda f, l: None,
            )

            with patch.object(
                TranscriptionService, "_transcribe_single_language_pcm", return_value=new_segments
            ), patch(
                "otoweave_app.transcription_service.JapaneseRecognizer", return_value=object()
            ), patch(
                "otoweave_app.transcription_service.SpeakerDiarizer", FakeDiarizer
            ):
                service.transcribe_existing_audio(
                    folder / "recording.pcm",
                    "japanese",
                    folder,
                    lesson,
                    diarization_speakers=2,
                )

            self.assertEqual(lesson.segments[0].speaker, "話者1")
            self.assertEqual(lesson.segments[1].speaker, "話者2")
            self.assertEqual(lesson.status, "complete")
            kinds = [events.get_nowait()[0] for _ in range(events.qsize())]
            self.assertIn("transcription_finished", kinds)
            self.assertNotIn("error", kinds)

    def test_diarization_failure_does_not_break_asr_result(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            store = LessonStore(Path(temporary) / "LearningAccess")
            folder, lesson = self._make_lesson(store)
            new_segments = [TranscriptSegment("seg_0001", 0.0, 5.0, "文字起こしは成功です。")]

            class RaisingDiarizer:
                def __init__(self, *args, **kwargs) -> None:
                    raise RuntimeError("boom: diarization model failed to load")

            events: "queue_module.Queue" = queue_module.Queue()
            ready_calls: list[tuple[Path, LessonRecord]] = []
            service = TranscriptionService(
                project_root,
                store,
                events,
                ffmpeg=Path("ffmpeg"),
                correct_text=lambda text: text,
                on_lesson_ready=lambda f, l: ready_calls.append((f, l)),
            )

            with patch.object(
                TranscriptionService, "_transcribe_single_language_pcm", return_value=new_segments
            ), patch(
                "otoweave_app.transcription_service.JapaneseRecognizer", return_value=object()
            ), patch(
                "otoweave_app.transcription_service.SpeakerDiarizer", RaisingDiarizer
            ):
                service.transcribe_existing_audio(
                    folder / "recording.pcm",
                    "japanese",
                    folder,
                    lesson,
                    diarization_speakers=2,
                )

            # ASR text/segments must survive a diarization failure untouched.
            self.assertEqual(lesson.segments[0].text, "文字起こしは成功です。")
            self.assertEqual(lesson.segments[0].speaker, "")
            self.assertEqual(lesson.status, "complete")
            self.assertEqual(len(ready_calls), 1)
            kinds = [events.get_nowait()[0] for _ in range(events.qsize())]
            self.assertIn("transcription_finished", kinds)
            self.assertNotIn("error", kinds)


if __name__ == "__main__":
    unittest.main()
