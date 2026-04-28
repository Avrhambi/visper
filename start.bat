@echo off
setlocal EnableDelayedExpansion
title Whisper STT

echo.
echo  ===================================================
echo    Whisper STT  —  Starting
echo  ===================================================
echo.

:: ── Check Python ─────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo  [!] Python is not installed.
    echo.
    echo  Please install Python 3.10 or newer:
    echo    1. Go to:  https://www.python.org/downloads/
    echo    2. Download the Windows installer
    echo    3. Run it and CHECK "Add Python to PATH"
    echo    4. Come back and double-click start.bat again
    echo.
    echo  Need help? A message has been copied to your clipboard.
    echo  Paste it into ChatGPT, Claude, or any AI chatbot for guided help.
    echo.
    echo I need to install Python 3.10 or newer on Windows to run a speech-to-text app. Please walk me through the installation step by step, including how to check "Add Python to PATH" during setup. | clip
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
    echo  [!] Python 3.10 or newer is required  (you have !PYVER!)
    echo.
    echo  Download the latest Python from:
    echo    https://www.python.org/downloads/
    echo.
    echo  Need help? A message has been copied to your clipboard.
    echo  Paste it into ChatGPT, Claude, or any AI chatbot for guided help.
    echo.
    echo I have Python !PYVER! installed on Windows but need Python 3.10 or newer for a speech-to-text app. Please walk me through upgrading Python, including how to check "Add Python to PATH" during setup. | clip
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  [OK] Python %%v found

:: ── Create venv if missing ───────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo  [Setup] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  [!] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment ready.
)

call .venv\Scripts\activate.bat

:: ── First-time setup ─────────────────────────────────────────────────
if not exist "benchmark_results.json" (
    echo.
    echo  [Setup] First-time setup — this will take several minutes.
    echo          The model ^(~1.5 GB^) will be downloaded. Please wait...
    echo.
    python install.py
    if errorlevel 1 (
        echo.
        echo  [!] Setup failed. Read the messages above for details.
        pause
        exit /b 1
    )
    echo.
)

:: ── Launch ───────────────────────────────────────────────────────────
echo  [Start] Starting server...
echo.
echo  ===================================================
echo    http://localhost:8000  will open in your browser
echo    Close this window to stop the server.
echo  ===================================================
echo.

start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

stt-server
