@echo off
setlocal enabledelayedexpansion
title codesearch

rem ---------------------------------------------------------------------------------
rem  The real script lives HERE, outside any synced folder. A batch file inside a
rem  Google Drive folder can be rewritten by the sync client while cmd.exe is still
rem  reading it, and cmd.exe reads scripts by byte offset -- so it resumes at a
rem  stale offset and tries to execute fragments of the file.
rem
rem  Usage:  start_codesearch.bat  [target-folder]
rem  Default target is the folder this script sits in.
rem ---------------------------------------------------------------------------------

set "SCRIPT_DIR=F:\AI\codesearch"
set "TARGET_DIR=%~1"
if "%TARGET_DIR%"=="" set "TARGET_DIR=%SCRIPT_DIR%"
rem A trailing backslash would escape the closing quote when the path is passed to
rem python as "C:\some\path\" and swallow the next argument. Strip it.
if "%TARGET_DIR:~-1%"=="\" set "TARGET_DIR=%TARGET_DIR:~0,-1%"

set "PY=C:\ProgramData\miniconda3\python.exe"
set "HF_PY=F:\AI\hfenv\Scripts\python.exe"
set "PYTHONPATH=F:\AI\codesearch\src"
set "CODESEARCH_IMAGE_MODEL_PATH=F:\AI\hfenv\hf_models\embeddinggemma-2"
set "HF_HOME=F:\AI\hfenv\hf_models\embeddinggemma-2"
set "SIDECAR_LOG=%SCRIPT_DIR%\image_server.log"

echo ============================================================
echo  codesearch  -  %TARGET_DIR%
echo ============================================================
echo.

rem --- single instance guard -------------------------------------------------------
"%PY%" -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1',5002)); s.close(); sys.exit(0 if r==0 else 1)" >nul 2>&1
if not errorlevel 1 goto ui_busy

"%PY%" -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1',1235)); s.close(); sys.exit(0 if r==0 else 1)" >nul 2>&1
if not errorlevel 1 goto sidecar_running

rem --- start the sidecar -----------------------------------------------------------
echo [1/4] Starting image embedding sidecar - embeddinggemma-2 on CUDA...
echo       log: %SIDECAR_LOG%
start "" /b "%HF_PY%" -u -m codesearch.image_server > "%SIDECAR_LOG%" 2>&1
goto wait_sidecar

:sidecar_running
echo [1/4] Image embedding sidecar already running on port 1235 - reusing it.

:wait_sidecar
echo.
echo [2/4] Waiting for the image model to be ready...
set /a TRIES=0
:wait_sidecar_loop
"%PY%" -c "import requests,sys; sys.exit(0 if requests.get('http://127.0.0.1:1235/health',timeout=3).ok else 1)" >nul 2>&1
if not errorlevel 1 goto sidecar_ready
set /a TRIES+=1
if !TRIES! geq 90 goto sidecar_failed
if !TRIES! equ 1 echo       loading the model - takes about 30-60 seconds
ping -n 4 127.0.0.1 >nul
goto wait_sidecar_loop

:sidecar_failed
echo.
echo ERROR - the image sidecar did not start. Last 20 log lines:
echo ----------------------------------------
powershell -NoProfile -Command "Get-Content -Tail 20 '%SIDECAR_LOG%'" 2>nul
echo ----------------------------------------
pause
exit /b 1

:sidecar_ready
echo       sidecar is up on http://127.0.0.1:1235

rem --- index + embed ---------------------------------------------------------------
echo.
echo [3/4] Indexing and embedding %TARGET_DIR%
echo       reads every file, then embeds text and image pixels
echo       Ctrl+C to stop. Progress is printed below.
echo.
"%PY%" -u -m codesearch -r "%TARGET_DIR%" index --images
if errorlevel 1 echo WARNING - the index step reported errors, see above.

rem --- web ui ----------------------------------------------------------------------
echo.
echo [4/4] Starting the codesearch web UI on http://127.0.0.1:5002
echo       Ctrl+C stops the UI. The sidecar keeps running.
echo.
"%PY%" -u -m codesearch -r "%TARGET_DIR%" serve
pause
exit /b 0

:ui_busy
echo ERROR - something is already serving port 5002.
echo        Close the other codesearch window first, then run this again.
pause
exit /b 1
