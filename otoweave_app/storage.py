from __future__ import annotations

import calendar
import json
import os
import re
import shutil
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from .models import LessonRecord, TranscriptSegment, coalesce_readable_segments


INVALID_FILENAME_RE = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def safe_name(value: str, fallback: str = "lesson", max_chars: int = 60) -> str:
    cleaned = INVALID_FILENAME_RE.sub("", value)
    cleaned = re.sub(r"\s+", "_", cleaned).strip(" ._-")
    cleaned = cleaned[:max_chars].rstrip(" ._-") or fallback
    if cleaned.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned


def week_bounds(day: date) -> tuple[date, date]:
    start = day - timedelta(days=day.weekday())
    return start, start + timedelta(days=6)


class LessonStore:
    TRASH_DIR = "_trash"
    TRASH_KEEP_DAYS = 30

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def trash_lesson(self, folder: Path) -> Path:
        """Move a lesson folder into the store trash instead of deleting.

        A child's mis-tap plus a mis-click on the confirmation dialog must
        not permanently destroy a class recording."""
        trash_root = self.root / self.TRASH_DIR
        trash_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = trash_root / f"{stamp}_{folder.name}"
        sequence = 2
        while target.exists():
            target = trash_root / f"{stamp}_{folder.name}_{sequence}"
            sequence += 1
        folder.rename(target)
        return target

    def purge_trash(self, keep_days: int | None = None) -> int:
        """Permanently delete trash entries older than keep_days."""
        trash_root = self.root / self.TRASH_DIR
        if not trash_root.is_dir():
            return 0
        cutoff = time.time() - (keep_days or self.TRASH_KEEP_DAYS) * 86400
        removed = 0
        for child in trash_root.iterdir():
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed

    def lesson_parent(self, lesson_date: date) -> Path:
        week_number = lesson_date.isocalendar().week
        start, end = week_bounds(lesson_date)
        month = f"{lesson_date.month:02d}_{calendar.month_name[lesson_date.month]}"
        week = f"Week_{week_number:02d}_{start:%m-%d}_{end:%m-%d}"
        return self.root / str(lesson_date.year) / month / week

    def create_lesson(self, lesson: LessonRecord) -> Path:
        day = date.fromisoformat(lesson.date)
        parent = self.lesson_parent(day)
        base_id = lesson.lesson_id
        base_folder_name = safe_name(base_id)
        sequence = 1
        while True:
            if sequence == 1:
                candidate_id = base_id
                folder_name = base_folder_name
            else:
                suffix = f"_{sequence}"
                candidate_id = f"{base_id}{suffix}"
                stem = base_folder_name[: 60 - len(suffix)].rstrip(" ._-")
                folder_name = f"{stem or 'lesson'}{suffix}"
            folder = parent / folder_name
            try:
                folder.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                sequence += 1
        lesson.lesson_id = candidate_id
        self.save(folder, lesson)
        return folder

    def save(
        self,
        folder: Path,
        lesson: LessonRecord,
        *,
        light: bool = False,
    ) -> None:
        """Persist a lesson. With light=True (used for every confirmed
        sentence during recording) the human-readable transcript.md is not
        regenerated; it is written by the full save when recording stops."""
        folder.mkdir(parents=True, exist_ok=True)
        transcript_path = folder / "transcript.json"
        if transcript_path.exists():
            # Keep the previous good version: a power cut during the write
            # must never cost more than the very latest change.
            os.replace(transcript_path, folder / "transcript.json.bak")
        self._write_json(transcript_path, lesson.transcript_dict())
        self._write_json(folder / "metadata.json", lesson.metadata_dict())
        self._write_json(folder / "marks.json", self._marks_dict(lesson))
        if not light:
            self._write_text(folder / "transcript.md", self._transcript_markdown(lesson))

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def load(self, folder: Path) -> LessonRecord:
        transcript = self._read_json(folder / "transcript.json")
        if transcript is None:
            # Fall back to the previous good version saved before the
            # last write (crash/power-cut recovery).
            transcript = self._read_json(folder / "transcript.json.bak")
        metadata = self._read_json(folder / "metadata.json") or {}
        needs_repair = False
        if transcript is None:
            if not metadata:
                raise ValueError(f"記録を読み取れません: {folder}")
            # The transcript is gone, but the lesson itself (title, date,
            # audio) is still usable — surface it instead of hiding it.
            transcript = {}
            needs_repair = True
        merged = {**metadata, **transcript}
        lesson = LessonRecord.from_dict(merged)
        if needs_repair:
            lesson.status = "needs_repair"
        marks_path = folder / "marks.json"
        marks_payload = self._read_json(marks_path) if marks_path.exists() else None
        if marks_payload is not None:
            marks = marks_payload.get("marks", [])
            segments = {segment.id: segment for segment in lesson.segments}
            for mark in marks:
                segment = segments.get(str(mark.get("segment_id", "")))
                if segment is None:
                    continue
                if mark.get("type") == "important":
                    segment.important = True
                    segment.important_at = str(mark.get("created_at", ""))
                elif mark.get("type") == "unclear":
                    segment.unclear = True
                    segment.unclear_at = str(mark.get("created_at", ""))
                elif mark.get("type") == "question":
                    segment.question = True
                    segment.question_at = str(mark.get("created_at", ""))
        lesson.segments = coalesce_readable_segments(lesson.segments)
        return lesson

    def _lesson_folders(self) -> list[Path]:
        folders = {path.parent for path in self.root.rglob("metadata.json")}
        folders.update(path.parent for path in self.root.rglob("transcript.json"))
        folders.update(path.parent for path in self.root.rglob("transcript.json.bak"))
        # Trashed lessons stay out of every listing and search.
        return sorted(
            folder
            for folder in folders
            if self.TRASH_DIR not in folder.parts
        )

    @staticmethod
    def _repair_placeholder(folder: Path) -> LessonRecord:
        """A minimal lesson entry so a damaged folder stays visible."""
        match = re.match(r"(\d{4}-\d{2}-\d{2})", folder.name)
        lesson_date = match.group(1) if match else date.today().isoformat()
        return LessonRecord(
            lesson_id=folder.name,
            title=folder.name,
            date=lesson_date,
            language_mode="record_only",
            audio_file="audio.opus",
            started_at="",
            status="needs_repair",
        )

    def list_lessons(self) -> list[tuple[Path, LessonRecord]]:
        lessons: list[tuple[Path, LessonRecord]] = []
        for folder in self._lesson_folders():
            try:
                lesson = self.load(folder)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                # Never silently drop a lesson: to the user that reads as
                # "my recording disappeared". Show it as needing repair.
                lesson = self._repair_placeholder(folder)
            lessons.append((folder, lesson))
        lessons.sort(key=lambda item: item[1].started_at or item[1].date, reverse=True)
        return lessons

    def list_lesson_metadata(self) -> list[tuple[Path, dict]]:
        """Lightweight lesson listing that reads metadata.json only.

        With hundreds of lessons, loading every transcript at startup takes
        seconds on a slow disk; the list view only needs metadata. Legacy
        folders without the listing fields are loaded once and backfilled."""
        entries: list[tuple[Path, dict]] = []
        for folder in self._lesson_folders():
            metadata = self._read_json(folder / "metadata.json")
            if metadata is None or "duration_seconds" not in metadata:
                try:
                    lesson = self.load(folder)
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    lesson = self._repair_placeholder(folder)
                metadata = lesson.metadata_dict()
                if lesson.status != "needs_repair":
                    try:
                        self._write_json(folder / "metadata.json", metadata)
                    except OSError:
                        pass
            entries.append((folder, metadata))
        entries.sort(
            key=lambda item: str(
                item[1].get("started_at") or item[1].get("date", "")
            ),
            reverse=True,
        )
        return entries

    def search_transcripts(self, needle: str) -> set[str]:
        """Return lesson_ids whose transcript text contains the query.

        Reads transcript files directly; intended for a background worker
        triggered by an explicit search action."""
        query = needle.strip().lower()
        if not query:
            return set()
        matched: set[str] = set()
        for folder in self._lesson_folders():
            transcript = self._read_json(folder / "transcript.json")
            if transcript is None:
                transcript = self._read_json(folder / "transcript.json.bak")
            if not transcript:
                continue
            text = " ".join(
                str(segment.get("text", ""))
                for segment in transcript.get("segments", [])
                if isinstance(segment, dict)
            )
            if query in text.lower():
                matched.add(str(transcript.get("lesson_id", "")))
        return matched

    def rename_lesson(self, folder: Path, lesson: LessonRecord, new_title: str) -> Path:
        lesson.title = new_title.strip() or lesson.title
        new_folder = folder.with_name(safe_name(f"{lesson.date}_{lesson.title}"))
        if new_folder != folder:
            candidate = new_folder
            suffix = 2
            while candidate.exists():
                candidate = new_folder.with_name(f"{new_folder.name}_{suffix}")
                suffix += 1
            folder.rename(candidate)
            folder = candidate
        self.save(folder, lesson)
        return folder

    def append_correction(
        self,
        folder: Path,
        lesson_id: str,
        segment_id: str,
        before: str,
        after: str,
    ) -> None:
        entry = {
            "lesson_id": lesson_id,
            "segment_id": segment_id,
            "corrected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "before": before,
            "after": after,
        }
        # JSONL is append-only: one write per correction instead of
        # rewriting the whole history each time. If a crash left the file
        # without a trailing newline, terminate that line first so the new
        # entry stays parseable on its own line.
        path = folder / "corrections.jsonl"
        needs_newline = False
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb") as probe:
                probe.seek(-1, 2)
                needs_newline = probe.read(1) != b"\n"
        with path.open("a", encoding="utf-8") as handle:
            if needs_newline:
                handle.write("\n")
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        payload = json.dumps(value, ensure_ascii=False, indent=2)
        LessonStore._write_text(path, payload + "\n")

    @staticmethod
    def _write_text(path: Path, value: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            # Without fsync, NTFS can persist the rename before the data
            # on a power cut, leaving an empty/truncated file behind.
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _marks_dict(lesson: LessonRecord) -> dict:
        marks = []
        for segment in lesson.segments:
            if segment.important:
                marks.append({
                    "segment_id": segment.id,
                    "type": "important",
                    "created_at": segment.important_at,
                })
            if segment.unclear:
                marks.append({
                    "segment_id": segment.id,
                    "type": "unclear",
                    "created_at": segment.unclear_at,
                })
            if segment.question:
                marks.append({
                    "segment_id": segment.id,
                    "type": "question",
                    "created_at": segment.question_at,
                })
        return {"lesson_id": lesson.lesson_id, "marks": marks}

    @staticmethod
    def _transcript_markdown(lesson: LessonRecord) -> str:
        lines = [f"# {lesson.title}", "", f"- Date: {lesson.date}", f"- Mode: {lesson.language_mode}", ""]
        for segment in lesson.segments:
            minute = int(segment.start // 60)
            second = int(segment.start % 60)
            marks = (
                f"{' ★' if segment.important else ''}"
                f"{' ?' if segment.unclear else ''}"
                f"{' !' if segment.question else ''}"
            )
            lines.extend([f"## {minute:02d}:{second:02d}{marks}", "", segment.text, ""])
        return "\n".join(lines).rstrip() + "\n"


def filter_lessons(
    lessons: Iterable[tuple[Path, LessonRecord]],
    mode: str,
    today: date | None = None,
) -> list[tuple[Path, LessonRecord]]:
    current = today or date.today()
    week_start, week_end = week_bounds(current)
    result = []
    for item in lessons:
        lesson = item[1]
        try:
            lesson_day = date.fromisoformat(lesson.date)
        except ValueError:
            # One lesson with a broken date must not break the whole list.
            continue
        if mode == "week" and not week_start <= lesson_day <= week_end:
            continue
        if mode == "important" and not any(segment.important for segment in lesson.segments):
            continue
        if mode == "unclear" and not any(segment.unclear for segment in lesson.segments):
            continue
        if mode == "question" and not any(segment.question for segment in lesson.segments):
            continue
        result.append(item)
    return result
