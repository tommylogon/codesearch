@echo off
setlocal enabledelayedexpansion
title codesearch - F:\AI\code

set "TARGET_DIR=F:\AI\code"
set "SCRIPT_DIR=F:\AI\codesearch"
set "PY=C:\ProgramData\miniconda3\python.exe"
set "HF_PY=F:\AI\hfenv\Scripts\python.exe"
set "PYTHONPATH=%SCRIPT_DIR%\src"
set "CODESEARCH_ROOT=%TARGET_DIR%"
set "CODESEARCH_IMAGE_MODEL_PATH=F:\AI\hfenv\hf_models\embeddinggemma-2"
set "HF_HOME=F:\AI\hfenv\hf_models\embeddinggemma-2"
set "SIDECAR_LOG=%SCRIPT_DIR%\image_server.log"

echo ============================================================
echo  codesearch  -  %TARGET_DIR%
echo ============================================================
echo.

rem --- check image sidecar on port 1235 ----------------------------
"%PY%" -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1',1235)); s.close(); sys.exit(0 if r==0 else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [1/3] Image embedding sidecar is already active on port 1235.
) else (
    echo [1/3] Starting image embedding sidecar on CUDA...
    start "" /b "%HF_PY%" -u -m codesearch.image_server > "%SIDECAR_LOG%" 2>&1
)

rem --- index + embed target ----------------------------------------
echo.
echo [2/3] Indexing and embedding %TARGET_DIR%...
echo       (Ctrl+C to stop at any time)
echo.
"%PY%" -u -m codesearch -r "%TARGET_DIR%" index

rem --- web UI ------------------------------------------------------
echo.
echo [3/3] Starting web UI on http://127.0.0.1:5002...
"%PY%" -u -m codesearch -r "%TARGET_DIR%" serve
pause
