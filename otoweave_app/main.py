from __future__ import annotations

# pythonw.exe 起動では標準エラーが見えないため、アプリ本体の import より
# 先にログとクラッシュ捕捉を構える（import 失敗も excepthook 経由で残る）。
try:
    from .app_logging import default_log_dir, setup_logging

    setup_logging(default_log_dir())
except Exception:
    pass

from .otoweave_app import main

if __name__ == "__main__":
    raise SystemExit(main())
