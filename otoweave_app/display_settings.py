from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TEXT_SIZES = ("Small", "Standard", "Large", "Extra Large")
LINE_SPACINGS = ("Standard", "Comfortable")
TEXT_WIDTHS = ("Wide", "Reading Width")
COLOR_MODES = ("Light", "Dark")


@dataclass
class DisplaySettings:
    text_size: str = "Standard"
    line_spacing: str = "Comfortable"
    text_width: str = "Reading Width"
    font_family: str = "Yu Gothic UI"
    color_mode: str = "Light"
    live_follow: bool = True
    highlight_current: bool = True


def available_reading_fonts(installed: Iterable[str]) -> tuple[str, ...]:
    families = {name.strip() for name in installed if name.strip() and not name.startswith("@")}
    preferred_groups = (
        (
            "UD デジタル 教科書体 N-R",
            "UD デジタル 教科書体 NP-R",
            "UD デジタル 教科書体 NK-R",
            "UD Digi Kyokasho N-R",
            "UD Digi Kyokasho NP-R",
            "UD Digi Kyokasho NK-R",
        ),
        ("BIZ UDP",),
        ("OpenDyslexic", "Open Dyslexic"),
        ("Yu Gothic UI",),
        ("Meiryo UI",),
    )
    result: list[str] = []
    for prefixes in preferred_groups:
        matches = sorted(
            name for name in families
            if any(name.casefold().startswith(prefix.casefold()) for prefix in prefixes)
        )
        for name in matches:
            if name not in result:
                result.append(name)
    if not result:
        result.append("TkDefaultFont")
    return tuple(result)


def load_display_settings(path: Path) -> DisplaySettings:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DisplaySettings()
    if not isinstance(payload, dict):
        return DisplaySettings()
    settings = DisplaySettings()
    if payload.get("text_size") in TEXT_SIZES:
        settings.text_size = payload["text_size"]
    if payload.get("line_spacing") in LINE_SPACINGS:
        settings.line_spacing = payload["line_spacing"]
    if payload.get("text_width") in TEXT_WIDTHS:
        settings.text_width = payload["text_width"]
    if isinstance(payload.get("font_family"), str) and payload["font_family"].strip():
        settings.font_family = payload["font_family"].strip()
    if payload.get("color_mode") in COLOR_MODES:
        settings.color_mode = payload["color_mode"]
    if isinstance(payload.get("live_follow"), bool):
        settings.live_follow = payload["live_follow"]
    if isinstance(payload.get("highlight_current"), bool):
        settings.highlight_current = payload["highlight_current"]
    return settings


def save_display_settings(path: Path, settings: DisplaySettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
