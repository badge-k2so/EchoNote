from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


SENTENCE_END_RE = re.compile(r"(?:[。！？!?]|\.(?:[\"'）)]*))$")
JAPANESE_TEXT_RE = re.compile(r"[ぁ-んァ-ヴ一-龯]")
MARKER_PLACEHOLDER_TEXTS = {"ここは重要", "ここをあとで確認", "ここで質問したい"}


@dataclass
class TranscriptSegment:
    id: str
    start: float
    end: float
    text: str
    speaker: str = ""
    status: str = "final"
    important: bool = False
    unclear: bool = False
    question: bool = False
    important_at: str = ""
    unclear_at: str = ""
    question_at: str = ""
    edited: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            id=str(value["id"]),
            start=float(value["start"]),
            end=float(value["end"]),
            text=str(value.get("text", "")),
            speaker="" if str(value.get("speaker", "")) == "teacher" else str(value.get("speaker", "")),
            status=str(value.get("status", "final")),
            important=bool(value.get("important", False)),
            unclear=bool(value.get("unclear", False)),
            question=bool(value.get("question", False)),
            important_at=str(value.get("important_at", "")),
            unclear_at=str(value.get("unclear_at", "")),
            question_at=str(value.get("question_at", "")),
            edited=bool(value.get("edited", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
            "status": self.status,
            "text": self.text,
            "important": self.important,
            "unclear": self.unclear,
            "question": self.question,
            "edited": self.edited,
        }


@dataclass
class LessonRecord:
    lesson_id: str
    title: str
    date: str
    language_mode: str
    audio_file: str
    started_at: str
    audio_source: str = "microphone"
    segments: list[TranscriptSegment] = field(default_factory=list)
    suggested_title: str = ""
    status: str = "complete"
    source_audio_name: str = ""
    imported_at: str = ""
    asr_processing_mode: str = ""
    asr_threads: int = 0
    detected_logical_cpus: int = 0
    is_demo: bool = False

    @classmethod
    def create(
        cls,
        language_mode: str,
        audio_source: str,
        now: datetime | None = None,
    ) -> "LessonRecord":
        current = now or datetime.now().astimezone()
        lesson_id = current.strftime("%Y-%m-%d_%H%M%S")
        mode_titles = {
            "japanese": "日本語の授業",
            "english": "English Class",
            "record_only": "録音",
        }
        return cls(
            lesson_id=lesson_id,
            title=mode_titles.get(language_mode, "授業"),
            date=current.date().isoformat(),
            language_mode=language_mode,
            audio_file="recording.pcm",
            started_at=current.isoformat(timespec="seconds"),
            audio_source=audio_source,
            status="recording",
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LessonRecord":
        return cls(
            lesson_id=str(value["lesson_id"]),
            title=str(value.get("title", "授業")),
            date=str(value.get("date", date.today().isoformat())),
            language_mode=str(value.get("language_mode", "record_only")),
            audio_file=str(value.get("audio_file", "audio.opus")),
            started_at=str(value.get("started_at", "")),
            audio_source=str(value.get("audio_source", "microphone")),
            segments=[TranscriptSegment.from_dict(item) for item in value.get("segments", [])],
            suggested_title=str(value.get("suggested_title", "")),
            status=str(value.get("status", "complete")),
            source_audio_name=str(value.get("source_audio_name", "")),
            imported_at=str(value.get("imported_at", "")),
            asr_processing_mode=str(value.get("asr_processing_mode", "")),
            asr_threads=int(value.get("asr_threads", 0)),
            detected_logical_cpus=int(value.get("detected_logical_cpus", 0)),
            is_demo=bool(value.get("is_demo", False)),
        )

    def transcript_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "title": self.title,
            "date": self.date,
            "language_mode": self.language_mode,
            "audio_file": self.audio_file,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "title": self.title,
            "date": self.date,
            "started_at": self.started_at,
            "language_mode": self.language_mode,
            "audio_source": self.audio_source,
            "audio_file": self.audio_file,
            "suggested_title": self.suggested_title,
            "status": self.status,
            "source_audio_name": self.source_audio_name,
            "imported_at": self.imported_at,
            "asr_processing_mode": self.asr_processing_mode,
            "asr_threads": self.asr_threads,
            "detected_logical_cpus": self.detected_logical_cpus,
            "is_demo": self.is_demo,
            "segment_count": len(self.segments),
            # Listing fields: the lesson list is built from metadata.json
            # alone, without reading every transcript.
            "duration_seconds": round(
                max((segment.end for segment in self.segments), default=0.0), 3
            ),
            "has_important": any(segment.important for segment in self.segments),
            "has_unclear": any(segment.unclear for segment in self.segments),
            "has_question": any(segment.question for segment in self.segments),
            "schema_version": 4,
        }


def append_readable_segment(
    segments: list[TranscriptSegment],
    incoming: TranscriptSegment,
    max_gap_seconds: float = 1.5,
    max_span_seconds: float = 12.0,
    max_chars: int = 120,
) -> TranscriptSegment:
    """Append a segment, merging ASR fragments that belong to one utterance."""
    if not segments:
        segments.append(incoming)
        return incoming

    previous = segments[-1]
    gap = incoming.start - previous.end
    can_merge = (
        previous.text not in MARKER_PLACEHOLDER_TEXTS
        and incoming.text not in MARKER_PLACEHOLDER_TEXTS
        and -0.2 <= gap <= max_gap_seconds
        and incoming.end - previous.start <= max_span_seconds
        and not SENTENCE_END_RE.search(previous.text.rstrip())
        and len(previous.text) + len(incoming.text) <= max_chars
    )
    if not can_merge:
        segments.append(incoming)
        return incoming

    separator = "、" if JAPANESE_TEXT_RE.search(previous.text + incoming.text) else " "
    previous.text = previous.text.rstrip(" 、,") + separator + incoming.text.lstrip(" 、,")
    previous.end = max(previous.end, incoming.end)
    previous.important = previous.important or incoming.important
    previous.unclear = previous.unclear or incoming.unclear
    previous.question = previous.question or incoming.question
    previous.important_at = previous.important_at or incoming.important_at
    previous.unclear_at = previous.unclear_at or incoming.unclear_at
    previous.question_at = previous.question_at or incoming.question_at
    previous.edited = previous.edited or incoming.edited
    return previous


def coalesce_readable_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    merged: list[TranscriptSegment] = []
    for segment in segments:
        append_readable_segment(merged, segment)
    return merged
