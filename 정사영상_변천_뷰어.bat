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

echo [1/3] 과거정사영상 폴더를 확인하고 새로 추가되거나 바뀐 영상만 타일로 만듭니다...
"%PY%" -W ignore webapp\ortho_history\build_tiles.py
if errorlevel 1 (
    echo.
    echo [경고] 타일 생성 중 오류가 있었습니다. 기존 타일로 뷰어를 엽니다.
)

echo.
echo [2/3] DEM으로 지표수 흐름을 계산합니다...
if exist "DEM(5m)" (
    "%PY%" -W ignore webapp\ortho_history\build_terrain.py
    if errorlevel 1 echo [경고] 지표수 계산 중 오류가 있었습니다. 지형 기능 없이 뷰어를 엽니다.
    if exist "DEM(5m)\dem(0.5m).tif" "%PY%" -W ignore webapp\ortho_history\build_terrain_hr.py
) else (
    echo DEM 폴더가 없어 건너뜁니다.
)

echo.
echo [3/3] 웹앱을 실행합니다.
"%PY%" webapp\ortho_history\serve.py
pause
