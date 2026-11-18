@echo off
setlocal
title codesearch workbench - F:\AI\codesearchtest

set "TARGET_DIR=F:\AI\codesearchtest"
set "SCRIPT_DIR=F:\AI\codesearch"
set "PY=C:\ProgramData\miniconda3\python.exe"
set "PYTHONPATH=%SCRIPT_DIR%\src"
set "CODESEARCH_ROOT=%TARGET_DIR%"

echo ============================================================
echo  codesearch test workbench  -  %TARGET_DIR%
echo ============================================================
echo.

"%PY%" -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1',5002)); s.close(); sys.exit(0 if r==0 else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [!] Something is already running on http://127.0.0.1:5002
    echo     Open http://127.0.0.1:5002 in your browser, or close the old window first.
    pause
    exit /b 1
)

echo [1/3] Indexing any new or changed files in %TARGET_DIR%...
if /i "%~1"=="--embed" (
    "%PY%" -u -m codesearch -r "%TARGET_DIR%" index
) else (
    "%PY%" -u -m codesearch -r "%TARGET_DIR%" index --no-embed
    echo       (Embeddings skipped by default for fast startup; pass --embed or run codesearch embed --limit N)
)
echo.
echo [2/3] Refreshing vector packs...
"%PY%" -u -m codesearch -r "%TARGET_DIR%" pack
echo.
echo [3/3] Starting workbench UI on http://127.0.0.1:5002
echo       Keep this window open and open http://127.0.0.1:5002 in your browser.
echo       Press Ctrl+C to stop.
echo.
"%PY%" -u -m codesearch -r "%TARGET_DIR%" serve
pause
