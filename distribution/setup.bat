@echo off
rem OtoWeave setup launcher.
rem Runs setup_test_pc.ps1 with "-ExecutionPolicy Bypass" so that setup
rem works even on PCs where PowerShell scripts are blocked by policy.
cd /d "%~dp0"
if not exist "%~dp0setup_test_pc.ps1" (
    echo [ERROR] setup_test_pc.ps1 was not found in this folder.
    echo Please copy the whole OtoWeave folder to this PC, then run setup.bat again.
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_test_pc.ps1"
if errorlevel 1 (
    echo.
    echo [ERROR] Setup did not finish successfully. Please check the messages above.
    echo If PowerShell itself is blocked on this PC, ask your administrator.
    pause
    exit /b 1
)
exit /b 0
