from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "id": "lesson_record",
        "name": "授業の要点",
        "instruction": "授業内容を復習しやすい記録に整理してください。",
        "sections": [
            "今日のテーマ",
            "大事なポイント",
            "出てきた用語",
            "覚えること",
            "あとで確認すること",
        ],
        "builtin": True,
    },
    {
        "id": "easy_japanese",
        "name": "やさしい日本語",
        "instruction": "難しい表現を避け、短く読みやすい日本語で説明してください。",
        "sections": ["かんたんなまとめ", "大切なこと", "ことばの説明"],
        "builtin": True,
    },
    {
        "id": "exam_review",
        "name": "テスト前の復習",
        "instruction": "試験前に短時間で復習できる学習メモを作ってください。",
        "sections": ["必ず覚えること", "重要語句", "間違えやすい点", "確認問題"],
        "builtin": True,
    },
    {
        "id": "vocabulary",
        "name": "重要語句と意味",
        "instruction": "重要な用語を抽出し、文字起こしの内容に沿って説明してください。",
        "sections": ["重要語句", "意味と説明", "使われた文脈"],
        "builtin": True,
    },
    {
        "id": "questions",
        "name": "復習問題を作る",
        "instruction": "理解を確認するための問題と、文字起こしに基づく解答を作ってください。",
        "sections": ["確認問題", "解答", "もう一度聞くとよいところ"],
        "builtin": True,
    },
    {
        "id": "meeting_memo",
        "name": "面談・会議メモ",
        "instruction": "話し合いの内容を、事実と今後の行動が分かるように整理してください。",
        "sections": ["話し合った内容", "決まったこと", "次にやること", "確認が必要な点"],
        "builtin": True,
    },
)


def templates_path(data_root: Path) -> Path:
    return Path(data_root) / "summary_templates.json"


def load_templates(path: Path) -> list[dict[str, Any]]:
    custom: list[dict[str, Any]] = []
    try:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        values = []
    if isinstance(values, list):
        for value in values:
            try:
                custom.append(normalize_template(value, builtin=False))
            except (TypeError, ValueError):
                continue
    return [dict(value) for value in DEFAULT_TEMPLATES] + custom


def save_custom_templates(path: Path, templates: list[dict[str, Any]]) -> None:
    custom = [
        normalize_template(value, builtin=False)
        for value in templates
        if not value.get("builtin")
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(custom, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_template(
    value: dict[str, Any],
    *,
    builtin: bool | None = None,
) -> dict[str, Any]:
    name = " ".join(str(value.get("name", "")).split()).strip()
    # Collapse the instruction to one line: it is inserted into the prompt
    # as 「追加の指示: ...」 and multi-line text could act as fake prompt
    # sections that override the summary safety rules.
    instruction = " ".join(str(value.get("instruction", "")).split()).strip()[:1000]
    if not name or not instruction:
        raise ValueError("テンプレート名と指示内容が必要です。")
    template_id = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        str(value.get("id", "")).strip(),
    ).strip("_")
    if not template_id:
        raise ValueError("テンプレートIDが必要です。")
    sections = [
        " ".join(str(section).split()).strip()
        for section in value.get("sections", [])
        if str(section).strip()
    ]
    if not sections:
        sections = ["要約"]
    return {
        "id": template_id,
        "name": name,
        "instruction": instruction,
        "sections": sections,
        "builtin": bool(value.get("builtin")) if builtin is None else builtin,
    }


def template_by_id(
    templates: list[dict[str, Any]],
    template_id: str,
) -> dict[str, Any]:
    return next(
        (value for value in templates if value["id"] == template_id),
        templates[0],
    )
