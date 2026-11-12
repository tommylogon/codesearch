@echo off
title codesearch image server & sidecar
echo Starting codesearch image embedding sidecar (embeddinggemma-2 on CUDA)...
set CODESEARCH_IMAGE_MODEL_PATH=F:\AI\hfenv\hf_models\embeddinggemma-2
set HF_HOME=F:\AI\hfenv\hf_models\embeddinggemma-2
start "codesearch-image-server" /min F:\AI\hfenv\Scripts\python.exe -m codesearch.image_server
echo Sidecar started on port 1235.
echo.
echo Starting codesearch web UI on port 5002...
python -m codesearch -r "F:\AI\characters" serve
pause

