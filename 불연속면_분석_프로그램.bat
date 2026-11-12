@echo off
chcp 65001 > nul
title 암벽 불연속면 분석 프로그램
cd /d "%~dp0"

rem 8501 포트는 다른 앱(독도 대시보드)이 쓰고 있어 8502를 사용합니다.
set "PORT=8502"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [오류] 가상환경 파이썬을 찾을 수 없습니다: %PY%
    pause
    exit /b 1
)

netstat -ano | findstr /r /c:":%PORT% .*LISTENING" > nul
if not errorlevel 1 (
    echo 프로그램이 이미 실행 중입니다. 브라우저로 엽니다: http://localhost:%PORT%
    start "" "http://localhost:%PORT%"
    timeout /t 3 > nul
    exit /b 0
)

if not exist "data\raw\demo_site\rgb_photogrammetry\pointcloud.ply" (
    echo [1/2] 데모용 합성 암벽 자료를 만듭니다...
    "%PY%" -W ignore scripts\generate_demo_data.py
    if errorlevel 1 (
        echo [경고] 데모 자료 생성에 실패했습니다. 실제 현장 자료 폴더로는 계속 사용할 수 있습니다.
    )
) else (
    echo [1/2] 데모 자료 확인 완료.
)

echo.
echo [2/2] 분석 프로그램을 실행합니다. 잠시 후 브라우저가 열립니다: http://localhost:%PORT%
echo       왼쪽에서 현장 자료 폴더를 고르고 "분석 실행"을 누르세요. 결과는 그 폴더의 분석결과에 저장됩니다.
echo       종료하려면 이 창을 닫으세요.
echo.
"%PY%" -m streamlit run src\groundwater\viz\dashboard\app.py --server.port %PORT% --browser.gatherUsageStats false
pause
