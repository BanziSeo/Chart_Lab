@echo off
REM --------------- Git auto-update ---------------

REM 1) move to script folder
cd /d "%~dp0"

REM 2) optional – change console to UTF-8; safe even without it
chcp 65001 > nul

echo.
git status
echo.

set /p MSG="Commit message (leave blank to cancel): "
if "%MSG%"=="" (
    echo Cancelled.
    pause
    goto :eof
)

echo.
echo Adding all changes...
git add -A

echo Committing...
git commit -m "%MSG%"
if errorlevel 1 (
    echo Nothing to commit.
    pause
    goto :eof
)

echo.
echo Pushing to origin/main...
git push origin main
if errorlevel 1 (
    echo [ERROR] Push failed – check network / credentials.
) else (
    echo Push succeeded!  Streamlit Cloud will redeploy automatically.
)
pause
