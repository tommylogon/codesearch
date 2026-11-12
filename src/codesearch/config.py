"""Central configuration for codesearch.

Everything tunable lives here: scan scope, storage paths, the embedding endpoint
and the retrieval weights. Environment variables override the endpoint settings.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("CODESEARCH_ROOT", Path.cwd())).resolve()
DATA_DIR = REPO_ROOT / ".codesearch"
DB_PATH = DATA_DIR / "codesearch.db"


def set_root(root: Path | str) -> Path:
    """Switch the active target directory (and default .codesearch/codesearch.db path)."""
    global REPO_ROOT, DATA_DIR, DB_PATH
    REPO_ROOT = Path(root).expanduser().resolve()
    DATA_DIR = REPO_ROOT / ".codesearch"
    DB_PATH = DATA_DIR / "codesearch.db"
    os.environ["CODESEARCH_ROOT"] = str(REPO_ROOT)
    return REPO_ROOT

# language by extension. Anything not listed is not indexed.
LANGUAGES = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".md": "markdown",
    ".txt": "text",
    ".srt": "text",
    ".vtt": "text",
    ".html": "html",
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
}

# Images become one `file` node each, language "image". Their path is still embedded by
# the text model (so "goblins/arix.png" is keyword- and meaning-searchable), and their
# pixels are embedded by a separate multimodal model into their own vector space.
IMAGE_LANGUAGES = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".bmp": "image",
}

IGNORE_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env", "node_modules", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".kilo", ".codesearch", ".playwright-mcp",
    "dist", "build", "review-verify", "audit", "playwright-report", "test-results",
    "saves", "downloads", ".obsidian", "chunks", ".vscode",
}
# repo-relative path prefixes that are skipped (vendored third-party code and runtime log exports).
IGNORE_PREFIXES = ("static/js/vendor/", "static/vendor/", "static/icons/", "data/exports/")
IGNORE_FILES = {
    "package-lock.json", ".live_state.json", "autosave.json", "sandbox_autosave.json",
    "embeddings.json", ".tasks-index.json",
}
# Generated manifests: a file that points at other files and repeats their content.
# Indexed once as a manifest, not once per repeated copy. fnmatch on the file name.
IGNORE_FILE_PATTERNS = ("character_index.json", "index.json", "transcript_full.jsonl")

# A .js file sitting next to a .ts file with the same stem is compiled output.
SKIP_JS_WITH_TS_SIBLING = True
# Files above this size are generated dumps, not source. They are reported, not indexed.
# Size caps in BYTES; set to None to disable the cap entirely.
MAX_FILE_BYTES = 100_000_000
MAX_JSON_BYTES = 100_000_000
# A JSON document is walked key by key, one row per key path. These bound the work per
# document. When a bound is hit the parser reports it; nothing is silently dropped.
MAX_JSON_NODES = 50_000
MAX_JSON_DEPTH = 64

# Embedding endpoint (LM Studio, OpenAI compatible).
EMBED_ENDPOINT = os.environ.get("CODESEARCH_EMBED_ENDPOINT",
                                os.environ.get("LM_STUDIO_ENDPOINT", "http://localhost:1234/v1"))
EMBED_MODEL = os.environ.get("CODESEARCH_EMBED_MODEL", "text-embedding-qwen3-embedding-0.6b")
EMBED_QUERY_INSTRUCTION = ""
EMBED_BATCH = 32
EMBED_WORKERS = 4
# The embedding model has a finite context window. This bounds the text SENT TO THE
# MODEL only; the full text stays in the database and is what the UI and CLI show.
EMBED_MAX_CHARS = int(os.environ.get("CODESEARCH_EMBED_MAX_CHARS", 32000))
EMBED_TIMEOUT = 320

# The four vector indexes, each answering a different question.
PARTS = ("filename", "function_name", "function", "comment")
PART_WEIGHTS = {"function_name": 1.0, "function": 1.0, "comment": 0.8, "filename": 0.6}

# The fifth index: pixels. It lives in a different vector space (a multimodal model sees
# images; the code-text model cannot), so it is never compared against the text parts.
# Each part contributes a ranked list and RRF fuses the lists, not the vectors.
IMAGE_PART = "image"
ALL_PARTS = PARTS + (IMAGE_PART,)
PART_WEIGHTS[IMAGE_PART] = 0.9

# The image embedder is a local HTTP sidecar (see image_server.py) so codesearch itself
# keeps its light dependency set: no torch, no transformers.
IMAGE_EMBED_ENDPOINT = os.environ.get("CODESEARCH_IMAGE_EMBED_ENDPOINT",
                                      "http://127.0.0.1:1235/v1")
IMAGE_EMBED_MODEL = os.environ.get("CODESEARCH_IMAGE_EMBED_MODEL", "embeddinggemma-2")
IMAGE_BATCH = 8
# Overlapping requests hides base64/JSON cost behind the GPU's forward pass.
IMAGE_WORKERS = 3
IMAGE_TIMEOUT = 320
IMAGE_MAX_BYTES = 40_000_000
# Cross-modal cosines live in a much narrower band than text-vs-text ones: a text query
# about cars still scored ~0.51 against a goblin portrait, so the text floor (0.45) would
# let pure noise into the image list. Images get their own, stricter floor.
IMAGE_MIN_SIMILARITY = 0.58

# Which model produced which part's vectors. A part is only loaded for the model that
# actually built it, so changing the image model invalidates that part instead of mixing
# vectors from two different spaces into one matrix.
PART_MODELS = {p: EMBED_MODEL for p in PARTS}
PART_MODELS[IMAGE_PART] = IMAGE_EMBED_MODEL

# FTS5 bm25 column weights, in column order of nodes_fts.
FTS_COLUMNS = ("name", "words", "signature", "docstring", "rel_path", "source")
BM25_WEIGHTS = (10.0, 8.0, 6.0, 4.0, 3.0, 1.0)

RRF_K = 60
FTS_WEIGHT = 1.6
CANDIDATES = 100          # per-list candidates before fusion
# A vector hit below this cosine similarity is noise and earns no rank credit. Without a
# floor, RRF pays every one of the top-N candidates even when nothing is actually similar.
MIN_SIMILARITY = 0.45
# A node present in multiple vector parts takes its best part at full weight plus this
# fraction of any corroborating parts, so a 3-part method does not drown out a 1-part
# comment or file node by 3x.
MULTI_PART_DECAY = 0.25

# CALLS resolution: a bare callee name matching more nodes than this is too
# ambiguous (get, update, ...) and produces no edge.
MAX_AMBIGUOUS_TARGETS = 3

WEB_PORT = 5002
