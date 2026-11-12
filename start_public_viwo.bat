@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title codesearch public API (port 8085)

set "PY=C:\ProgramData\miniconda3\python.exe"
if not exist "%PY%" (
    where python >nul 2>nul && set "PY=python"
)

set "PORT=8085"
set "KEY_ARG="
if not "%~1"=="" (
    set "KEY_ARG=--key %~1"
)

echo ============================================================
echo  Starting codesearch multi-index public API on port %PORT%
echo  Indexes: viwo, hogwarts
if not "%~1"=="" (
    echo  Password Auth: ENABLED
) else (
    echo  Password Auth: DISABLED (Pass password as argument to enable: start_public_viwo.bat yourpassword)
)
echo ============================================================
echo.

"%PY%" serve_public_viwo.py --port %PORT% --host 0.0.0.0 %KEY_ARG%
pause
