@echo off
title Cassiopea Pipeline
cd /d "%~dp0"
echo Starting Cassiopea Pipeline...
venv\Scripts\python.exe scripts\run_ui.py
if %errorlevel% neq 0 (
    echo.
    echo Application exited with an error. Check the output above.
    pause
)
