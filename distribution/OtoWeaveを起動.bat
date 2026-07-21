@echo off
rem OtoWeave launcher (starts the app without a console window).
cd /d "%~dp0"
set "HF_HOME=%~dp0hf-cache"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo [ERROR] Python environment not found: .venv\Scripts\pythonw.exe
    echo Please run "setup.bat" in this folder first, then try again.
    pause
    exit /b 1
)
if not exist "%~dp0otoweave_app\main.py" (
    echo [ERROR] Application files not found: otoweave_app
    echo Please copy the whole OtoWeave folder to this PC, then try again.
    pause
    exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" -m otoweave_app.main
exit /b 0
