@echo off
set CODESEARCH_IMAGE_EMBED_ENDPOINT=http://127.0.0.1:1235/v1
set CODESEARCH_IMAGE_EMBED_MODEL=embeddinggemma-2
set CODESEARCH_IMAGE_PORT=1235
set HF_HOME=F:\AI\hfenv
set HF_HOME=F:\AI\hfenv\hf_models\embeddinggemma-2
python -m codesearch.image_server