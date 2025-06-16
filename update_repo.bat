@echo off
REM ────────── Git auto-update script ──────────
setlocal enabledelayedexpansion

echo --------------- Git status ---------------
git status -s

:: ── 커밋 메시지 입력
set /p MSG=Commit message (leave blank to cancel): 
if "%MSG%"=="" (
    echo [CANCEL] Empty message
    goto:eof
)

:: ── 변경 파일 스테이지 & 커밋
echo Adding changes...
git add -A

echo Committing...
git commit -m "%MSG%"

:: ── 원격 최신 변경 반영 후 푸시
echo Rebasing latest remote...
git pull --rebase origin main
if errorlevel 1 (
    echo [ERROR] Pull failed – resolve conflicts, then run again.
    pause
    goto:eof
)

echo Pushing...
git push origin main
if errorlevel 1 (
    echo [ERROR] Push failed – check network / credentials.
    pause
    goto:eof
)

echo.
echo [OK] Repo updated successfully.
pause
