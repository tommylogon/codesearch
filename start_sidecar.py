# Start the image embedding sidecar on CUDA (hfenv) with flask installed
import os, sys
sys.path.insert(0, "F:\\AI\\codesearch\\src")
os.environ["PYTHONPATH"] = "F:\\AI\\codesearch\\src;" + os.environ.get("PYTHONPATH", "")
os.environ["CODESEARCH_IMAGE_EMBED_ENDPOINT"] = "http://127.0.0.1:1235/v1"
os.environ["CODESEARCH_IMAGE_EMBED_MODEL"] = "embeddinggemma-2"
os.environ["CODESEARCH_IMAGE_PORT"] = "1235"
os.environ["HF_HOME"] = "F:\\AI\\hfenv\\hf_models\\embeddinggemma-2"

from codesearch.image_server import main
main()