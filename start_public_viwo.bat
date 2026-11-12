@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title codesearch public API (port 8085)

set "PY=C:\ProgramData\miniconda3\python.exe"
if not exist "%PY%" (
    where python >nul 2>nul && set "PY=python"
)

set "TARGET_DIR=F:\AI\viwo\virtual-world"
set "PORT=8085"

echo ============================================================
echo  Starting codesearch public read-only API on port %PORT%
echo  Target: %TARGET_DIR%
echo ============================================================
echo.

"%PY%" serve_public_viwo.py --root "%TARGET_DIR%" --port %PORT% --host 0.0.0.0
pause
