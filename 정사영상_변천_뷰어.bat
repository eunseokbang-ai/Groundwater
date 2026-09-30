@echo off
chcp 65001 > nul
title 익산 정사영상 변천 뷰어
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [오류] 가상환경 파이썬을 찾을 수 없습니다: %PY%
    pause
    exit /b 1
)

echo [1/2] 과거정사영상 폴더를 확인하고 새로 추가되거나 바뀐 영상만 타일로 만듭니다...
"%PY%" -W ignore webapp\ortho_history\build_tiles.py
if errorlevel 1 (
    echo.
    echo [경고] 타일 생성 중 오류가 있었습니다. 기존 타일로 뷰어를 엽니다.
)

echo.
echo [2/2] 웹앱을 실행합니다.
"%PY%" webapp\ortho_history\serve.py
pause
