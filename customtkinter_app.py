"""Compatibility launcher for the OtoWeave CustomTkinter app.

The implementation lives in otoweave_app.otoweave_app; this file
keeps scripts/prototypes/run_customtkinter_app.ps1 and older imports working.
"""
from __future__ import annotations

from otoweave_app.otoweave_app import (  # noqa: F401
    AUDIO_EXTENSIONS,
    OtoWeaveApp,
    _lesson_to_note,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
