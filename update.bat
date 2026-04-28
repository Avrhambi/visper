@echo off
setlocal
title Whisper STT — Update

echo.
echo  ===================================================
echo    Whisper STT  —  Update
echo  ===================================================
echo.

:: ── Check git ────────────────────────────────────────────────────────
where git >nul 2>&1
if errorlevel 1 (
    echo  [!] git is not installed — cannot pull updates automatically.
    echo.
    echo  Option A — Install git, then run update.bat again:
    echo    https://git-scm.com/download/win
    echo.
    echo  Option B — Download the latest ZIP directly from GitHub:
    echo    https://github.com/Avrhambi/local-whisper-he/archive/refs/heads/master.zip
    echo    Extract it over your current folder, then run start.bat.
    echo.
    pause
    exit /b 1
)

:: ── Pull latest changes ───────────────────────────────────────────────
echo  [Update] Downloading latest changes...
git pull
if errorlevel 1 (
    echo.
    echo  [!] Update failed.  Possible reasons:
    echo       - No internet connection
    echo       - You have local edits that conflict
    echo.
    echo  If you have no local edits, run:
    echo    git reset --hard origin/master
    echo  then double-click update.bat again.
    echo.
    pause
    exit /b 1
)

:: ── Update Python packages ────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo  [!] Virtual environment not found — run start.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo  [Update] Updating dependencies...
pip install -r requirements.txt -q
pip install -e ".[server]" -q
if errorlevel 1 (
    echo  [!] Dependency update failed.
    pause
    exit /b 1
)

echo.
echo  [OK] Update complete!
echo       Double-click start.bat to launch the updated version.
echo.
pause
