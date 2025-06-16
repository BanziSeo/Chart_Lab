@echo off
title Chart Trainer

REM 현재 bat 파일이 있는 디렉터리
set "ROOT=%~dp0"
cd /d "%ROOT%"

REM 1. python.exe 찾기
where python >nul 2>nul || (
    echo [ERROR] Python 3.x 가 PATH 에 없습니다.
    pause
    exit /b
)

REM 2. 가상환경
if not exist venv (
    python -m venv venv
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r "%ROOT%Chart_Lab\requirements.txt"
) else (
    call venv\Scripts\activate
)

REM 3. 스트림릿 실행
python -m streamlit run "%ROOT%Chart_Lab\app.py"
pause
