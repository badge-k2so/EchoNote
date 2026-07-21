from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import LessonRecord


POSTPROCESS_SUBDIR = "postprocess"


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", value).strip("_")
    return cleaned or "summary"


def transcript_fingerprint(lesson: LessonRecord) -> str:
    value = [
        {
            "id": segment.id,
            "start": float(round(segment.start, 3)),
            "end": float(round(segment.end, 3)),
            "speaker": segment.speaker,
            "text": segment.text,
        }
        for segment in lesson.segments
    ]
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def template_fingerprint(template: dict[str, Any]) -> str:
    value = {
        "id": str(template.get("id", "")),
        "name": str(template.get("name", "")),
        "instruction": str(template.get("instruction", "")),
        "sections": list(template.get("sections", [])),
        "dictionary": str(template.get("dictionary", "")),
    }
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_directory(lesson_folder: Path, template_id: str) -> Path:
    return (
        Path(lesson_folder)
        / POSTPROCESS_SUBDIR
        / "summaries"
        / _safe_id(template_id)
    )


def save_cached_summary(
    lesson_folder: Path,
    lesson: LessonRecord,
    template: dict[str, Any],
    text: str,
    model_path: Path,
) -> None:
    folder = cache_directory(lesson_folder, str(template["id"]))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.md").write_text(text, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "template_id": str(template["id"]),
        "template_name": str(template.get("name", template["id"])),
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "transcript_sha256": transcript_fingerprint(lesson),
        "template_sha256": template_fingerprint(template),
        "model": Path(model_path).name,
    }
    (folder / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def inspect_cached_summary(
    lesson_folder: Path,
    lesson: LessonRecord,
    template: dict[str, Any],
) -> dict[str, Any]:
    template_id = str(template["id"])
    folder = cache_directory(lesson_folder, template_id)
    result_path = folder / "result.md"
    metadata_path = folder / "metadata.json"
    if result_path.is_file():
        text = result_path.read_text(encoding="utf-8", errors="replace")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        current = (
            metadata.get("transcript_sha256")
            == transcript_fingerprint(lesson)
            and metadata.get("template_sha256")
            == template_fingerprint(template)
        )
        return {
            "status": "generated" if current else "stale",
            "text": text,
            "metadata": metadata,
        }

    postprocess = Path(lesson_folder) / POSTPROCESS_SUBDIR
    legacy = postprocess / "summaries" / f"{_safe_id(template_id)}.md"
    if legacy.is_file():
        return {
            "status": "stale",
            "text": legacy.read_text(encoding="utf-8", errors="replace"),
            "metadata": {"legacy": True},
        }
    selected_template_path = postprocess / "summary_template.json"
    school_record = postprocess / "school_record.md"
    if selected_template_path.is_file() and school_record.is_file():
        try:
            selected = json.loads(
                selected_template_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            selected = {}
        if str(selected.get("id", "")) == template_id:
            return {
                "status": "stale",
                "text": school_record.read_text(
                    encoding="utf-8",
                    errors="replace",
                ),
                "metadata": {"legacy": True},
            }
    elif template_id == "lesson_record" and school_record.is_file():
        return {
            "status": "stale",
            "text": school_record.read_text(
                encoding="utf-8",
                errors="replace",
            ),
            "metadata": {"legacy": True},
        }
    return {"status": "missing", "text": "", "metadata": {}}


def activate_cached_summary(
    lesson_folder: Path,
    cache_result: dict[str, Any],
) -> None:
    path = (
        Path(lesson_folder)
        / POSTPROCESS_SUBDIR
        / "school_record.md"
    )
    if cache_result.get("status") == "generated":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(cache_result.get("text", "")), encoding="utf-8")
    else:
        path.unlink(missing_ok=True)
