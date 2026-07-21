from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any


CATEGORIES = ("学習用語", "読み書き支援", "英語表現", "その他")


def dictionary_path(data_root: Path) -> Path:
    return Path(data_root) / "user_dictionary.json"


def normalize_entry(value: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    term = " ".join(str(value.get("term", "")).split()).strip()
    if not term:
        raise ValueError("正式表記を入力してください。")
    aliases_value = value.get("aliases", [])
    if isinstance(aliases_value, str):
        aliases_value = re.split(r"[,、\n]", aliases_value)
    aliases: list[str] = []
    for alias in aliases_value:
        cleaned = " ".join(str(alias).split()).strip()
        if not cleaned or cleaned == term or cleaned in aliases:
            continue
        if len(cleaned) < 2:
            # A one-character alias like 「かん」→ even worse 「か」 replaces
            # inside unrelated words and can corrupt the whole transcript.
            if strict:
                raise ValueError(
                    f"誤認識候補「{cleaned}」が短すぎます。"
                    "2文字以上で入力してください（1文字は他の言葉まで置き換わる危険があります）。"
                )
            continue
        aliases.append(cleaned)
    entry_id = str(value.get("id", "")).strip() or uuid.uuid4().hex
    category = str(value.get("category", "学習用語")).strip()
    if category not in CATEGORIES:
        category = "その他"
    # Keep the description on one line: it is inserted into the summary
    # prompt as a bullet and must not break the prompt structure.
    description = " ".join(str(value.get("description", "")).split()).strip()
    return {
        "id": entry_id,
        "term": term,
        "reading": " ".join(str(value.get("reading", "")).split()).strip(),
        "aliases": aliases,
        "description": description[:500],
        "category": category,
    }


def load_dictionary(path: Path) -> list[dict[str, Any]]:
    try:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(values, list):
        return []
    entries: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            # Lenient when loading: a too-short alias saved by an older
            # version is dropped, but the entry itself is kept.
            entries.append(normalize_entry(value, strict=False))
        except ValueError:
            continue
    return entries


def save_dictionary(path: Path, entries: list[dict[str, Any]]) -> None:
    normalized = [normalize_entry(value) for value in entries]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def correct_text(text: str, entries: list[dict[str, Any]]) -> str:
    corrected = text
    replacements: list[tuple[str, str]] = []
    for entry in entries:
        replacements.extend(
            (str(alias), str(entry["term"]))
            for alias in entry.get("aliases", [])
            # Defense in depth: never substring-replace 1-character aliases.
            if len(str(alias)) >= 2
        )
    placeholders: list[tuple[str, str]] = []
    for index, (alias, term) in enumerate(sorted(
        replacements,
        key=lambda item: len(item[0]),
        reverse=True,
    )):
        placeholder = f"\ue000{chr(0xE100 + index)}\ue001"
        if re.fullmatch(r"[A-Za-z0-9 _.-]+", alias):
            pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
            corrected = re.sub(
                pattern,
                lambda _match, value=placeholder: value,
                corrected,
                flags=re.IGNORECASE,
            )
        else:
            corrected = corrected.replace(alias, placeholder)
        placeholders.append((placeholder, term))
    for placeholder, term in placeholders:
        corrected = corrected.replace(placeholder, term)
    return corrected


def glossary_prompt(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return ""
    lines = [
        "以下はユーザー登録辞書です。入力内に現れた語の表記と意味を"
        "優先し、入力にない事実は追加しないでください。"
    ]
    for entry in entries:
        reading = f"（{entry['reading']}）" if entry.get("reading") else ""
        description = (
            f": {entry['description']}"
            if entry.get("description")
            else ""
        )
        lines.append(f"- {entry['term']}{reading}{description}")
    return "\n".join(lines)
