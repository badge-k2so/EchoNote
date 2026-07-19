from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


TEXT_SIZES = ("Small", "Standard", "Large", "Extra Large")
LINE_SPACINGS = ("Standard", "Comfortable")
TEXT_WIDTHS = ("Wide", "Reading Width")
COLOR_MODES = ("Light", "Dark")


def _default_font_family(platform: str = sys.platform) -> str:
    """既定の表示フォント。設定ファイルが無い初回起動時に使われる。

    macOS にしか無い Windows フォント（Yu Gothic UI）を既定にすると、
    Mac初回起動時に存在しないフォント名が選択済み扱いになってしまうため、
    OS既定のヒラギノ角ゴシックを返す。Windows/Linuxの既定値は変えない。
    """
    # TODO(platform_support): platform_support.py 導入後は共通の
    # is_macos() 判定に置き換える。
    if platform == "darwin":
        return "Hiragino Sans"
    return "Yu Gothic UI"


@dataclass
class DisplaySettings:
    text_size: str = "Standard"
    line_spacing: str = "Comfortable"
    text_width: str = "Reading Width"
    font_family: str = field(default_factory=_default_font_family)
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
        # macOS 標準の日本語UIフォント。Windows 環境の installed には現れない
        # ため、この並びを追加しても Windows 側の候補・順序は変わらない。
        ("Hiragino Sans",),
        ("Hiragino Kaku Gothic ProN",),
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
