"""Multimodal embedding sidecar for codesearch.

Loads google/embeddinggemma-2 and exposes an OpenAI-compatible /v1/embeddings endpoint
that accepts both strings and OpenAI-style image_url data URIs. Runs on CUDA with bf16.

Start:  python -m codesearch.image_server
Env:    HF_HOME, CODESEARCH_IMAGE_EMBED_ENDPOINT, CODESEARCH_IMAGE_EMBED_MODEL, CUDA_VISIBLE_DEVICES
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, request
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

# Load config from the main package so env vars are respected
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codesearch import config  # noqa: E402

app = Flask(__name__)

MODEL: SentenceTransformer | None = None
DEVICE: str = "cpu"
MODEL_NAME: str = config.IMAGE_EMBED_MODEL
MODEL_PATH: str = os.environ.get("CODESEARCH_IMAGE_MODEL_PATH", "")


def load_model() -> None:
    global MODEL, DEVICE
    if MODEL is not None:
        return

    # torch 2.7.0+cu126 (NOT 2.11.0+cu128 — that build fails to import c10.dll in this conda-based Python)
    if torch.cuda.is_available():
        DEVICE = "cuda"
        torch.set_float32_matmul_precision("high")
    else:
        DEVICE = "cpu"

    # google/embeddinggemma-2 is a sentence-transformers model; we disable audio_config
    # which isn't needed for image+text and avoids a warning about missing audio encoder.
    model_source = MODEL_PATH if MODEL_PATH else MODEL_NAME
    MODEL = SentenceTransformer(
        model_source,
        device=DEVICE,
        model_kwargs={"torch_dtype": torch.bfloat16 if DEVICE == "cuda" else torch.float32},
        config_kwargs={"audio_config": None},
    )
    print(f"[image_server] loaded {MODEL_NAME} on {DEVICE}", flush=True)


def _data_uri_to_pil(uri: str) -> Image.Image:
    """data:image/png;base64,....  -> PIL Image"""
    if not uri.startswith("data:"):
        raise ValueError("expected data URI")
    header, b64 = uri.split(",", 1)
    mime = header.split(";")[0].split(":", 1)[1] if ";" in header else header.split(":")[1]
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")


def _embed_items(items: list) -> tuple[np.ndarray, list[int], list[dict]]:
    """Embed a mixed list of strings and image data URIs.

    Returns (vectors, original_indices, failures). A single undecodable image must not
    fail the whole batch -- the client would lose the other seven. Items that cannot be
    decoded are reported separately and the rest are still embedded.
    """
    kept: list[tuple[int, str | Image.Image]] = []
    failures: list[dict] = []
    for idx, item in enumerate(items):
        try:
            if isinstance(item, str):
                kept.append((idx, item))
            elif isinstance(item, dict) and item.get("type") == "image_url":
                kept.append((idx, _data_uri_to_pil(item["image_url"]["url"])))
            else:
                raise ValueError(f"unsupported input item type {type(item)}")
        except Exception as exc:
            failures.append({"index": idx, "error": f"{type(exc).__name__}: {exc}"})

    if not kept:
        return np.empty((0, 0), dtype=np.float32), [], failures

    embeddings = MODEL.encode(
        [v for _, v in kept],
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    indices = [i for i, _ in kept]
    return embeddings.astype(np.float32), indices, failures


@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    start = time.time()
    payload = request.get_json(force=True, silent=True) or {}
    model = payload.get("model", MODEL_NAME)
    if model != MODEL_NAME:
        return jsonify({"error": f"model {model!r} not served (loaded: {MODEL_NAME})"}), 400

    inputs = payload.get("input")
    if inputs is None:
        return jsonify({"error": "missing 'input' field"}), 400
    if isinstance(inputs, str):
        inputs = [inputs]

    try:
        vecs, indices, failures = _embed_items(inputs)
    except Exception as exc:
        # A whole-batch failure (out of memory, a model-level error) must be visible.
        traceback.print_exc()
        print(f"[image_server] /embeddings FAILED for {len(inputs)} items: {exc}", flush=True)
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    data = [{"object": "embedding", "embedding": vec.tolist(), "index": idx}
            for vec, idx in zip(vecs, indices)]
    elapsed = time.time() - start
    for f in failures:
        print(f"[image_server] item {f['index']} failed: {f['error']}", flush=True)
    print(f"[image_server] /embeddings {len(inputs)} items in {elapsed:.3f}s "
          f"({len(data)} ok, {len(failures)} failed)", flush=True)
    return jsonify({"object": "list", "data": data, "model": MODEL_NAME,
                    "errors": failures,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0}})


@app.route("/v1/models", methods=["GET"])
def models():
    return jsonify({"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "codesearch"}]})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": DEVICE})


def main() -> None:
    load_model()
    port = int(os.environ.get("CODESEARCH_IMAGE_PORT", "1235"))
    app.run(host="127.0.0.1", port=port, threaded=True)


if __name__ == "__main__":
    main()